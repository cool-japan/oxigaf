//! # Eye Gaze Estimation and Control
//!
//! Provides a comprehensive gaze control system for the `OxiGAF` FLAME head model,
//! including Listing's law quaternion computation, I-VT/I-DT saccade/fixation
//! detection, natural blink synthesis, vergence estimation, and statistics.
//!
//! ## Overview
//!
//! - [`GazeDirection`] — azimuth / elevation / vergence representation
//! - [`GazeFrame`]     — per-frame binocular gaze + blink data
//! - [`GazeController`] — stateful ring-buffer gaze manager
//! - [`gz_listing_rotation`] — Listing's law quaternion computation
//! - [`gz_detect_saccades`] / [`gz_detect_fixations`] — I-VT classifier
//! - [`gz_synthesize_blinks`] — natural blink generation (xorshift64 + exponential ISI)

use thiserror::Error;

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
    /// Phase of the blink at detection onset.
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
    /// Whether Listing's law is enforced when computing rotations.
    pub listing_enforcement: bool,
}

impl Default for GazeControllerConfig {
    fn default() -> Self {
        Self {
            fps: 60.0,
            saccade_velocity_threshold_dps: 30.0,
            fixation_min_duration_ms: 100.0,
            blink_rate_per_min: 15.0,
            blink_duration_ms: 150.0,
            listing_enforcement: true,
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
const HISTORY_CAP: usize = 1_000;

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
        let fix_min = self.config.fixation_min_duration_ms;
        let blink_thr = 0.5_f32;

        self.saccades = gz_detect_saccades(&self.history, fps, vthr, fix_min);
        self.fixations = gz_detect_fixations(&self.history, fps, vthr, fix_min);
        self.blinks = gz_detect_blinks(&self.history, fps, blink_thr);
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

// ---------------------------------------------------------------------------
// Internal PRNG (xorshift64)
// ---------------------------------------------------------------------------

/// Advance the xorshift64 state and return the next pseudo-random `u64`.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    if s == 0 {
        s = 1;
    }
    *state = s;
    s
}

/// Hash a seed with a splitmix64-style step to produce a well-distributed
/// initial xorshift64 state even for small integer seeds.
#[inline]
fn gz_seed_hash(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9_u64);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB_u64);
    x = x ^ (x >> 31);
    if x == 0 {
        1
    } else {
        x
    }
}

/// Map a `u64` to `[0, 1)` uniformly using the full 64-bit range.
#[inline]
fn xorshift64_f32(state: &mut u64) -> f32 {
    let raw = xorshift64(state);
    // Divide by 2^64 to get uniform [0, 1).
    // Use f64 intermediate for precision, then cast to f32.
    #[allow(clippy::cast_precision_loss)]
    let v = (raw as f64) * (1.0_f64 / u64::MAX as f64);
    v as f32
}

// ---------------------------------------------------------------------------
// Listing's law
// ---------------------------------------------------------------------------

/// Compute the rotation axis that satisfies Listing's law.
///
/// For a gaze direction `target_dir` reached from `primary`, Listing's law
/// states that the rotation axis lies in Listing's plane — i.e., it is
/// perpendicular to `primary`.  The axis is the component of
/// `primary × target_dir` projected onto Listing's plane (the plane whose
/// normal is `primary`).
///
/// # Errors
///
/// Returns [`GazeControllerError::ZeroVector`] when `primary` or `target_dir`
/// is the zero vector, or when `primary` and `target_dir` are (anti-)parallel
/// so that no unique axis exists.
pub fn gz_listing_axis(
    primary: [f32; 3],
    target_dir: [f32; 3],
) -> Result<[f32; 3], GazeControllerError> {
    let pn = gz_vec3_norm(primary);
    let tn = gz_vec3_norm(target_dir);
    if pn < 1e-7 {
        return Err(GazeControllerError::ZeroVector(
            "primary gaze direction is zero".into(),
        ));
    }
    if tn < 1e-7 {
        return Err(GazeControllerError::ZeroVector(
            "target direction is zero".into(),
        ));
    }

    let p = gz_vec3_normalize(primary);
    let t = gz_vec3_normalize(target_dir);

    // Cross product p × t gives the candidate rotation axis.
    let axis = gz_vec3_cross(p, t);
    let axis_norm = gz_vec3_norm(axis);

    if axis_norm < 1e-7 {
        return Err(GazeControllerError::ZeroVector(
            "primary and target are (anti-)parallel — no unique Listing axis".into(),
        ));
    }

    // Project axis onto Listing's plane (remove component along primary).
    let dot_ap = gz_vec3_dot(axis, p);
    let listing_axis = [
        axis[0] - dot_ap * p[0],
        axis[1] - dot_ap * p[1],
        axis[2] - dot_ap * p[2],
    ];

    let la_norm = gz_vec3_norm(listing_axis);
    if la_norm < 1e-7 {
        // Fallback: return the raw cross product normalised.
        return Ok(gz_vec3_scale(axis, 1.0 / axis_norm));
    }

    Ok(gz_vec3_scale(listing_axis, 1.0 / la_norm))
}

/// Compute a unit quaternion `[qx, qy, qz, qw]` satisfying Listing's law.
///
/// The quaternion represents the rotation from `primary` to `target_dir`,
/// constrained so that the rotation axis lies in Listing's plane.
///
/// # Errors
///
/// Propagates [`GazeControllerError::ZeroVector`] from [`gz_listing_axis`].
/// Returns the identity quaternion when target equals primary.
pub fn gz_listing_rotation(
    primary: [f32; 3],
    target_dir: [f32; 3],
) -> Result<[f32; 4], GazeControllerError> {
    let pn = gz_vec3_norm(primary);
    let tn = gz_vec3_norm(target_dir);
    if pn < 1e-7 {
        return Err(GazeControllerError::ZeroVector(
            "primary is zero vector".into(),
        ));
    }
    if tn < 1e-7 {
        return Err(GazeControllerError::ZeroVector(
            "target_dir is zero vector".into(),
        ));
    }

    let p = gz_vec3_normalize(primary);
    let t = gz_vec3_normalize(target_dir);

    // Angle between primary and target.
    let cos_angle = gz_vec3_dot(p, t).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();

    if angle < 1e-7 {
        // Identity quaternion: already at target.
        return Ok([0.0, 0.0, 0.0, 1.0]);
    }

    let axis = gz_listing_axis(primary, target_dir)?;

    let half = 0.5 * angle;
    let (sin_h, cos_h) = half.sin_cos();
    let q = [axis[0] * sin_h, axis[1] * sin_h, axis[2] * sin_h, cos_h];
    Ok(gz_quat_normalise(q))
}

// ---------------------------------------------------------------------------
// Angular velocity
// ---------------------------------------------------------------------------

/// Compute angular velocity (deg/s) between consecutive cyclopean gaze frames.
///
/// The velocity at index `i` is the angular distance between frame `i` and `i+1`
/// divided by the inter-frame interval.  The output has `frames.len() - 1` elements.
/// An empty or single-frame input produces an empty vector.
#[must_use]
pub fn gz_angular_velocity(frames: &[GazeFrame], fps: f32) -> Vec<f32> {
    if frames.len() < 2 {
        return Vec::new();
    }
    let dt_s = if fps > 0.0 { 1.0 / fps } else { 1.0 };
    let mut vel = Vec::with_capacity(frames.len() - 1);
    for w in frames.windows(2) {
        let a = w[0].cyclopean_gaze();
        let b = w[1].cyclopean_gaze();
        let da = gz_angular_distance_deg(&a, &b);
        vel.push(da / dt_s);
    }
    vel
}

// ---------------------------------------------------------------------------
// Saccade / fixation / blink detection
// ---------------------------------------------------------------------------

/// Detect saccades using the I-VT (velocity-threshold) algorithm.
///
/// A saccade is a contiguous run of velocity samples that exceed
/// `velocity_threshold_dps`.  Runs shorter than `min_duration_ms` are
/// discarded.
#[must_use]
pub fn gz_detect_saccades(
    frames: &[GazeFrame],
    fps: f32,
    velocity_threshold_dps: f32,
    min_duration_ms: f32,
) -> Vec<SaccadeEvent> {
    if frames.len() < 2 {
        return Vec::new();
    }
    let vel = gz_angular_velocity(frames, fps);
    let ms_per_frame = if fps > 0.0 { 1000.0 / fps } else { 1000.0 };
    let mut events = Vec::new();

    let mut i = 0_usize;
    while i < vel.len() {
        if vel[i] > velocity_threshold_dps {
            // Start of a saccade.
            let start = i;
            while i < vel.len() && vel[i] > velocity_threshold_dps {
                i += 1;
            }
            let end = i; // exclusive
            let n_frames = end - start;
            let duration_ms = n_frames as f32 * ms_per_frame;
            if duration_ms < min_duration_ms {
                continue;
            }
            // Compute amplitude: angular distance from first to last frame in window.
            let first_gaze = frames[start].cyclopean_gaze();
            let last_gaze = frames[end].cyclopean_gaze();
            let amplitude_deg = gz_angular_distance_deg(&first_gaze, &last_gaze);
            // Peak velocity.
            let peak_velocity_dps = vel[start..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            events.push(SaccadeEvent {
                amplitude_deg,
                peak_velocity_dps,
                duration_ms,
                start_step: frames[start].step,
                end_step: frames[end].step,
            });
        } else {
            i += 1;
        }
    }
    events
}

/// Detect fixations using the I-VT algorithm (complement of saccades).
///
/// A fixation is a contiguous run of velocity samples at or below
/// `velocity_threshold_dps`.  Runs shorter than `min_duration_ms` are
/// discarded.
#[must_use]
pub fn gz_detect_fixations(
    frames: &[GazeFrame],
    fps: f32,
    velocity_threshold_dps: f32,
    min_duration_ms: f32,
) -> Vec<FixationEvent> {
    if frames.len() < 2 {
        return Vec::new();
    }
    let vel = gz_angular_velocity(frames, fps);
    let ms_per_frame = if fps > 0.0 { 1000.0 / fps } else { 1000.0 };
    let mut events = Vec::new();

    let mut i = 0_usize;
    while i < vel.len() {
        if vel[i] <= velocity_threshold_dps {
            let start = i;
            while i < vel.len() && vel[i] <= velocity_threshold_dps {
                i += 1;
            }
            let end = i; // exclusive
                         // Include the last frame (index end) in the fixation window.
            let n_frames = end - start + 1;
            let duration_ms = n_frames as f32 * ms_per_frame;
            if duration_ms < min_duration_ms {
                continue;
            }
            let slice_gazes: Vec<GazeDirection> = frames[start..=end.min(frames.len() - 1)]
                .iter()
                .map(GazeFrame::cyclopean_gaze)
                .collect();
            let dispersion_deg = gz_dispersion(&slice_gazes);
            let centroid_az =
                slice_gazes.iter().map(|g| g.azimuth).sum::<f32>() / slice_gazes.len() as f32;
            let centroid_el =
                slice_gazes.iter().map(|g| g.elevation).sum::<f32>() / slice_gazes.len() as f32;
            events.push(FixationEvent {
                duration_ms,
                dispersion_deg,
                centroid_az,
                centroid_el,
                start_step: frames[start].step,
                end_step: frames[end.min(frames.len() - 1)].step,
            });
        } else {
            i += 1;
        }
    }
    events
}

/// Detect blink events by threshold-crossing of the mean blink value.
///
/// A blink is detected when the mean blink (average of left and right)
/// transitions from below `threshold` to at or above `threshold`.
#[must_use]
pub fn gz_detect_blinks(frames: &[GazeFrame], fps: f32, threshold: f32) -> Vec<BlinkEvent> {
    if frames.len() < 2 {
        return Vec::new();
    }
    let ms_per_frame = if fps > 0.0 { 1000.0 / fps } else { 1000.0 };
    let mut events = Vec::new();
    let mut in_blink = false;
    let mut blink_start = 0_usize;

    for (idx, frame) in frames.iter().enumerate() {
        let v = frame.mean_blink();
        if !in_blink && v >= threshold {
            in_blink = true;
            blink_start = idx;
        } else if in_blink && v < threshold {
            in_blink = false;
            let n_frames = idx - blink_start;
            let duration_ms = n_frames as f32 * ms_per_frame;
            events.push(BlinkEvent {
                duration_ms,
                phase: BlinkPhase::Closing,
                start_step: frames[blink_start].step,
            });
        }
    }
    // Handle blink still active at end of recording.
    if in_blink {
        let n_frames = frames.len() - blink_start;
        let duration_ms = n_frames as f32 * ms_per_frame;
        events.push(BlinkEvent {
            duration_ms,
            phase: BlinkPhase::Closed,
            start_step: frames[blink_start].step,
        });
    }
    events
}

/// Compute the I-DT dispersion metric (max range of azimuth + elevation) in degrees.
///
/// Returns `0.0` for an empty or single-element slice.
#[must_use]
pub fn gz_dispersion(gaze_slice: &[GazeDirection]) -> f32 {
    if gaze_slice.len() < 2 {
        return 0.0;
    }
    let az_vals: Vec<f32> = gaze_slice.iter().map(|g| g.azimuth).collect();
    let el_vals: Vec<f32> = gaze_slice.iter().map(|g| g.elevation).collect();

    let az_min = az_vals.iter().copied().fold(f32::INFINITY, f32::min);
    let az_max = az_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let el_min = el_vals.iter().copied().fold(f32::INFINITY, f32::min);
    let el_max = el_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let dispersion_rad = (az_max - az_min) + (el_max - el_min);
    dispersion_rad.to_degrees()
}

// ---------------------------------------------------------------------------
// Blink model
// ---------------------------------------------------------------------------

/// Compute the blink amplitude at time `t_ms` within a blink of `duration_ms`.
///
/// Uses a cosine model: rises from `0` → `1` over the first half of the
/// duration, then falls from `1` → `0` over the second half.
///
/// Returns values in `[0, 1]`.
#[must_use]
pub fn gz_blink_waveform(t_ms: f32, duration_ms: f32) -> f32 {
    if duration_ms <= 0.0 {
        return 0.0;
    }
    let t_norm = (t_ms / duration_ms).clamp(0.0, 1.0);
    // Full cosine envelope: 0→1→0 over [0,1].
    0.5 * (1.0 - (std::f32::consts::TAU * t_norm).cos())
}

/// Synthesise a natural blink sequence over `duration_steps` frames.
///
/// Uses an xorshift64 PRNG with exponential inter-blink intervals to produce
/// a realistic, variable blink cadence.  Each blink is rendered with
/// [`gz_blink_waveform`] over a fixed 150 ms window.
///
/// Returns a `Vec<f32>` of length `duration_steps` with values in `[0, 1]`.
#[must_use]
pub fn gz_synthesize_blinks(
    duration_steps: usize,
    fps: f32,
    rate_per_min: f32,
    seed: u64,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; duration_steps];
    if duration_steps == 0 || fps <= 0.0 || rate_per_min <= 0.0 {
        return out;
    }
    let mean_interval_ms = 60_000.0 / rate_per_min;
    let blink_dur_ms = 150.0_f32;
    let ms_per_step = 1000.0 / fps;
    let total_ms = duration_steps as f32 * ms_per_step;
    // Hash seed to ensure well-distributed initial state even for small values.
    let mut prng = gz_seed_hash(if seed == 0 { 1 } else { seed });

    // Walk through time, placing blinks with exponentially-distributed ISIs.
    let mut t_ms = gz_exponential_sample(&mut prng, mean_interval_ms);

    while t_ms < total_ms {
        let blink_start_step = (t_ms / ms_per_step) as usize;
        // Render blink waveform into output buffer.
        let blink_dur_steps = ((blink_dur_ms / ms_per_step).ceil() as usize).max(1);
        for k in 0..blink_dur_steps {
            let step = blink_start_step + k;
            if step >= duration_steps {
                break;
            }
            let local_t_ms = k as f32 * ms_per_step;
            let v = gz_blink_waveform(local_t_ms, blink_dur_ms);
            // Accumulate by maximum to handle rare overlapping blinks.
            if v > out[step] {
                out[step] = v;
            }
        }
        // Advance time by blink duration + exponentially-distributed ISI.
        let isi_ms = gz_exponential_sample(&mut prng, mean_interval_ms);
        t_ms += blink_dur_ms + isi_ms;
    }
    out
}

/// Sample from an exponential distribution with `mean` using xorshift64.
/// Guards against zero uniform sample.
#[inline]
fn gz_exponential_sample(prng: &mut u64, mean: f32) -> f32 {
    let u = xorshift64_f32(prng).max(1e-7_f32);
    -u.ln() * mean
}

// ---------------------------------------------------------------------------
// Vergence
// ---------------------------------------------------------------------------

/// Estimate the vergence (fixation) distance in metres from inter-ocular disparity.
///
/// Uses the approximation: `distance = (iod_m * 0.5) / tan(half_convergence_angle)`.
///
/// # Errors
///
/// Returns [`GazeControllerError::NonFinite`] when the computed distance is
/// not finite (e.g. when eyes are parallel — divergence is zero).
pub fn gz_vergence_from_iod(
    left_dir: &GazeDirection,
    right_dir: &GazeDirection,
    iod_mm: f32,
) -> Result<f32, GazeControllerError> {
    let iod_m = iod_mm / 1000.0;
    // Horizontal angular disparity (convergence angle) in radians.
    let disparity_rad = (right_dir.azimuth - left_dir.azimuth).abs();
    if disparity_rad < 1e-9 {
        // Eyes are parallel → object at infinity.
        return Ok(0.0);
    }
    let half_angle = 0.5 * disparity_rad;
    let dist = (iod_m * 0.5) / half_angle.tan();
    if !dist.is_finite() {
        return Err(GazeControllerError::NonFinite(
            "vergence distance (disparity near zero)".into(),
        ));
    }
    Ok(dist.abs())
}

/// Compute the convergence angle in degrees for a fixation at `vergence_dist_m`.
///
/// `iod_m` is the inter-ocular distance in metres.
/// Returns `0.0` when `vergence_dist_m` is zero (optical infinity).
#[must_use]
pub fn gz_convergence_angle_deg(vergence_dist_m: f32, iod_m: f32) -> f32 {
    if vergence_dist_m <= 0.0 || iod_m <= 0.0 {
        return 0.0;
    }
    let half_iod = iod_m * 0.5;
    let half_angle_rad = (half_iod / vergence_dist_m).atan();
    (2.0 * half_angle_rad).to_degrees()
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute aggregate statistics over a controller's current history and events.
#[must_use]
pub fn gz_compute_stats(controller: &GazeController, fps: f32) -> GazeStats {
    let n_frames = controller.history.len();
    let saccades = controller.saccades();
    let fixations = controller.fixations();
    let blinks = controller.blink_events();

    let mean_fixation_dur_ms = if fixations.is_empty() {
        0.0
    } else {
        fixations.iter().map(|f| f.duration_ms).sum::<f32>() / fixations.len() as f32
    };

    let mean_saccade_amplitude_deg = if saccades.is_empty() {
        0.0
    } else {
        saccades.iter().map(|s| s.amplitude_deg).sum::<f32>() / saccades.len() as f32
    };

    let duration_min = if fps > 0.0 && n_frames > 0 {
        n_frames as f32 / fps / 60.0
    } else {
        0.0
    };

    let blink_rate_per_min = if duration_min > 0.0 {
        blinks.len() as f32 / duration_min
    } else {
        0.0
    };

    let mean_vergence_m = if n_frames == 0 {
        0.0
    } else {
        let sum: f32 = controller
            .history
            .iter()
            .map(|f| f.cyclopean_gaze().vergence)
            .sum();
        sum / n_frames as f32
    };

    GazeStats {
        n_frames,
        n_saccades: saccades.len(),
        n_fixations: fixations.len(),
        n_blinks: blinks.len(),
        mean_fixation_dur_ms,
        mean_saccade_amplitude_deg,
        blink_rate_per_min,
        mean_vergence_m,
    }
}

/// Format a [`GazeStats`] summary as a human-readable string.
#[must_use]
pub fn gz_format_stats(stats: &GazeStats) -> String {
    format!(
        "GazeStats {{ frames: {}, saccades: {}, fixations: {}, blinks: {}, \
         mean_fix_dur: {:.1} ms, mean_sacc_amp: {:.1} deg, \
         blink_rate: {:.1}/min, mean_vergence: {:.3} m }}",
        stats.n_frames,
        stats.n_saccades,
        stats.n_fixations,
        stats.n_blinks,
        stats.mean_fixation_dur_ms,
        stats.mean_saccade_amplitude_deg,
        stats.blink_rate_per_min,
        stats.mean_vergence_m,
    )
}

// ---------------------------------------------------------------------------
// Private vector math helpers
// ---------------------------------------------------------------------------

#[inline]
fn gz_vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn gz_vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn gz_vec3_norm(v: [f32; 3]) -> f32 {
    gz_vec3_dot(v, v).sqrt()
}

#[inline]
fn gz_vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let n = gz_vec3_norm(v);
    if n < 1e-12 {
        v
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

#[inline]
fn gz_vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
fn gz_quat_normalise(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

/// Angular distance in degrees between two `GazeDirection`s (using Cartesian dot-product).
#[inline]
fn gz_angular_distance_deg(a: &GazeDirection, b: &GazeDirection) -> f32 {
    let va = a.to_cartesian();
    let vb = b.to_cartesian();
    let cos_a = gz_vec3_dot(va, vb).clamp(-1.0, 1.0);
    cos_a.acos().to_degrees()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_frame(step: u64, az: f32, el: f32, blink: f32, ts: f64) -> GazeFrame {
        let gaze = GazeDirection::new(az, el, 0.0);
        GazeFrame {
            step,
            left_gaze: gaze,
            right_gaze: gaze,
            blink_left: blink,
            blink_right: blink,
            timestamp_ms: ts,
        }
    }

    fn default_config() -> GazeControllerConfig {
        GazeControllerConfig::default()
    }

    // -----------------------------------------------------------------------
    // GazeDirection
    // -----------------------------------------------------------------------

    #[test]
    fn test_gaze_direction_primary() {
        let p = GazeDirection::primary();
        assert_eq!(p.azimuth, 0.0);
        assert_eq!(p.elevation, 0.0);
        assert_eq!(p.vergence, 0.0);
    }

    #[test]
    fn test_gaze_direction_default_matches_primary() {
        let d = GazeDirection::default();
        let p = GazeDirection::primary();
        assert_eq!(d.azimuth, p.azimuth);
        assert_eq!(d.elevation, p.elevation);
    }

    #[test]
    fn test_gaze_direction_to_cartesian_primary() {
        let p = GazeDirection::primary();
        let c = p.to_cartesian();
        // primary gaze: az=0, el=0 → (0, 0, 1)
        assert!((c[0]).abs() < 1e-5, "x should be ~0, got {}", c[0]);
        assert!((c[1]).abs() < 1e-5, "y should be ~0, got {}", c[1]);
        assert!((c[2] - 1.0).abs() < 1e-5, "z should be ~1, got {}", c[2]);
    }

    #[test]
    fn test_gaze_direction_to_cartesian_right() {
        // az = pi/2, el = 0 → gaze to the right
        let g = GazeDirection::new(PI / 2.0, 0.0, 0.0);
        let c = g.to_cartesian();
        assert!((c[0] - 1.0).abs() < 1e-5, "x should be ~1, got {}", c[0]);
        assert!((c[1]).abs() < 1e-5, "y should be ~0, got {}", c[1]);
    }

    #[test]
    fn test_gaze_direction_to_cartesian_unit_length() {
        let g = GazeDirection::new(0.3, -0.2, 0.5);
        let c = g.to_cartesian();
        let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "Cartesian vector should be unit, len={len}"
        );
    }

    // -----------------------------------------------------------------------
    // GazeFrame helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_gaze_frame_cyclopean() {
        let f = make_frame(0, 0.1, -0.1, 0.0, 0.0);
        let c = f.cyclopean_gaze();
        assert!((c.azimuth - 0.1).abs() < 1e-6);
        assert!((c.elevation - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_gaze_frame_mean_blink() {
        let mut f = make_frame(0, 0.0, 0.0, 0.0, 0.0);
        f.blink_left = 0.6;
        f.blink_right = 0.4;
        assert!((f.mean_blink() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gaze_frame_monocular() {
        let g = GazeDirection::new(0.2, 0.1, 1.5);
        let f = GazeFrame::monocular(5, g, 83.3);
        assert_eq!(f.step, 5);
        assert_eq!(f.blink_left, 0.0);
        assert_eq!(f.blink_right, 0.0);
        assert!((f.left_gaze.azimuth - 0.2).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Listing's law
    // -----------------------------------------------------------------------

    #[test]
    fn test_listing_axis_primary_forward() {
        // primary = [0,0,1], target slightly to the right [sin(0.1), 0, cos(0.1)]
        let primary = [0.0_f32, 0.0, 1.0];
        let angle = 0.1_f32;
        let target = [angle.sin(), 0.0, angle.cos()];
        let axis = gz_listing_axis(primary, target).expect("listing axis");
        // Axis should be in Listing's plane → dot with primary ≈ 0.
        let dot = axis[0] * primary[0] + axis[1] * primary[1] + axis[2] * primary[2];
        assert!(dot.abs() < 1e-5, "axis dot primary should be ~0, got {dot}");
        // Axis should be approximately [0, 1, 0] for rightward gaze (rotation around Y).
        // The sign may differ; check absolute value.
        assert!(
            (axis[1].abs() - 1.0).abs() < 0.05,
            "expect ~Y axis, got {axis:?}"
        );
    }

    #[test]
    fn test_listing_axis_zero_primary_error() {
        let result = gz_listing_axis([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(result.is_err(), "should fail for zero primary");
    }

    #[test]
    fn test_listing_axis_zero_target_error() {
        let result = gz_listing_axis([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        assert!(result.is_err(), "should fail for zero target");
    }

    #[test]
    fn test_listing_axis_antiparallel_error() {
        let result = gz_listing_axis([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]);
        assert!(result.is_err(), "antiparallel should fail");
    }

    #[test]
    fn test_listing_rotation_identity_for_equal_dirs() {
        let primary = [0.0_f32, 0.0, 1.0];
        let q = gz_listing_rotation(primary, primary).expect("identity listing rotation");
        // Should be identity quaternion [0, 0, 0, 1].
        assert!((q[3] - 1.0).abs() < 1e-5, "w should be ~1, got {}", q[3]);
        assert!(q[0].abs() < 1e-5);
        assert!(q[1].abs() < 1e-5);
        assert!(q[2].abs() < 1e-5);
    }

    #[test]
    fn test_listing_rotation_unit_quaternion() {
        let primary = [0.0_f32, 0.0, 1.0];
        let target = [0.1_f32.sin(), 0.0, 0.1_f32.cos()];
        let q = gz_listing_rotation(primary, target).expect("listing rotation");
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "quaternion should be unit, norm={norm}"
        );
    }

    #[test]
    fn test_listing_rotation_zero_primary_error() {
        let result = gz_listing_rotation([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_listing_rotation_zero_target_error() {
        let result = gz_listing_rotation([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_listing_rotation_axis_in_listing_plane() {
        // The rotation axis of the quaternion should be perpendicular to primary.
        let primary = [0.0_f32, 0.0, 1.0];
        let target = [0.0_f32, 0.2_f32.sin(), 0.2_f32.cos()];
        let q = gz_listing_rotation(primary, target).expect("listing rotation");
        // Axis component is [qx, qy, qz].
        let dot = q[0] * primary[0] + q[1] * primary[1] + q[2] * primary[2];
        assert!(
            dot.abs() < 1e-4,
            "rotation axis not in Listing plane, dot={dot}"
        );
    }

    // -----------------------------------------------------------------------
    // Angular velocity
    // -----------------------------------------------------------------------

    #[test]
    fn test_angular_velocity_empty() {
        let vel = gz_angular_velocity(&[], 60.0);
        assert!(vel.is_empty());
    }

    #[test]
    fn test_angular_velocity_single_frame() {
        let frames = vec![make_frame(0, 0.0, 0.0, 0.0, 0.0)];
        let vel = gz_angular_velocity(&frames, 60.0);
        assert!(vel.is_empty());
    }

    #[test]
    fn test_angular_velocity_zero_for_identical() {
        let frames: Vec<GazeFrame> = (0..5)
            .map(|i| make_frame(i, 0.0, 0.0, 0.0, i as f64 / 60.0 * 1000.0))
            .collect();
        let vel = gz_angular_velocity(&frames, 60.0);
        for v in &vel {
            assert!(
                v.abs() < 1e-5,
                "velocity should be zero for identical frames, got {v}"
            );
        }
    }

    #[test]
    fn test_angular_velocity_nonzero_for_moving() {
        // Jump of ~10 degrees between frame 0 and frame 1.
        let deg10 = (10.0_f32).to_radians();
        let frames = vec![
            make_frame(0, 0.0, 0.0, 0.0, 0.0),
            make_frame(1, deg10, 0.0, 0.0, 1000.0 / 60.0),
        ];
        let vel = gz_angular_velocity(&frames, 60.0);
        assert_eq!(vel.len(), 1);
        // At 60fps with 10 deg jump: velocity ≈ 10 * 60 = 600 dps.
        assert!(vel[0] > 500.0, "expected high velocity, got {}", vel[0]);
    }

    #[test]
    fn test_angular_velocity_output_length() {
        let frames: Vec<GazeFrame> = (0..10).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
        let vel = gz_angular_velocity(&frames, 60.0);
        assert_eq!(vel.len(), frames.len() - 1);
    }

    // -----------------------------------------------------------------------
    // Saccade detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_saccades_empty() {
        let s = gz_detect_saccades(&[], 60.0, 30.0, 10.0);
        assert!(s.is_empty());
    }

    #[test]
    fn test_detect_saccades_all_slow() {
        // Tiny movement: should not be detected as saccade.
        let frames: Vec<GazeFrame> = (0..20)
            .map(|i| make_frame(i, 0.001 * i as f32, 0.0, 0.0, 0.0))
            .collect();
        let s = gz_detect_saccades(&frames, 60.0, 30.0, 10.0);
        assert!(
            s.is_empty(),
            "no saccade expected for slow movement, got {:?} saccades",
            s.len()
        );
    }

    #[test]
    fn test_detect_saccades_large_jump() {
        let fps = 100.0_f32;
        let threshold = 30.0_f32; // dps
                                  // A 20-frame run at 5 deg/frame × 100fps = 500 dps (above threshold).
        let step_deg = 5.0_f32.to_radians();
        let mut frames: Vec<GazeFrame> = Vec::new();
        // 10 static frames.
        for i in 0..10 {
            frames.push(make_frame(i, 0.0, 0.0, 0.0, 0.0));
        }
        // 20 fast frames.
        for i in 0..20 {
            frames.push(make_frame(10 + i, step_deg * i as f32, 0.0, 0.0, 0.0));
        }
        let s = gz_detect_saccades(&frames, fps, threshold, 1.0);
        assert!(!s.is_empty(), "expected saccade detection");
        assert!(s[0].peak_velocity_dps > threshold);
    }

    #[test]
    fn test_detect_saccades_min_duration_filter() {
        // Single-frame saccade should be filtered if min_duration_ms is high.
        let fps = 60.0;
        let jump = (30.0_f32).to_radians();
        let frames = vec![
            make_frame(0, 0.0, 0.0, 0.0, 0.0),
            make_frame(1, jump, 0.0, 0.0, 0.0),
            make_frame(2, jump, 0.0, 0.0, 0.0),
        ];
        // min_duration_ms = 1000 ms → single frame saccade (16.67 ms) should be filtered.
        let s = gz_detect_saccades(&frames, fps, 30.0, 1000.0);
        assert!(
            s.is_empty(),
            "saccade should be filtered by min_duration_ms"
        );
    }

    // -----------------------------------------------------------------------
    // Fixation detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_fixations_empty() {
        let f = gz_detect_fixations(&[], 60.0, 30.0, 100.0);
        assert!(f.is_empty());
    }

    #[test]
    fn test_detect_fixations_stationary_sequence() {
        // All frames at same position → should be a fixation.
        let fps = 60.0;
        let frames: Vec<GazeFrame> = (0..60)
            .map(|i| make_frame(i, 0.0, 0.0, 0.0, i as f64 * 1000.0 / f64::from(fps)))
            .collect();
        let f = gz_detect_fixations(&frames, fps, 30.0, 100.0);
        assert!(
            !f.is_empty(),
            "expected at least one fixation for stationary gaze"
        );
    }

    #[test]
    fn test_detect_fixations_centroid_accuracy() {
        // 30 frames at (az=0.1, el=0.05) → centroid should match.
        let fps = 60.0;
        let az = 0.1_f32;
        let el = 0.05_f32;
        let frames: Vec<GazeFrame> = (0..30).map(|i| make_frame(i, az, el, 0.0, 0.0)).collect();
        let f = gz_detect_fixations(&frames, fps, 30.0, 100.0);
        if !f.is_empty() {
            assert!(
                (f[0].centroid_az - az).abs() < 0.01,
                "centroid_az expected {}, got {}",
                az,
                f[0].centroid_az
            );
            assert!(
                (f[0].centroid_el - el).abs() < 0.01,
                "centroid_el expected {}, got {}",
                el,
                f[0].centroid_el
            );
        }
    }

    // -----------------------------------------------------------------------
    // Blink detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_blinks_empty() {
        let b = gz_detect_blinks(&[], 60.0, 0.5);
        assert!(b.is_empty());
    }

    #[test]
    fn test_detect_blinks_no_blink() {
        let frames: Vec<GazeFrame> = (0..20).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
        let b = gz_detect_blinks(&frames, 60.0, 0.5);
        assert!(b.is_empty(), "no blink expected, got {}", b.len());
    }

    #[test]
    fn test_detect_blinks_single_blink() {
        let fps = 60.0;
        // Construct a sequence: open, blink, open.
        let mut frames: Vec<GazeFrame> =
            (0..10).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
        for i in 10..15_u64 {
            frames.push(make_frame(i, 0.0, 0.0, 0.8, 0.0)); // blink
        }
        for i in 15..25_u64 {
            frames.push(make_frame(i, 0.0, 0.0, 0.0, 0.0)); // open
        }
        let b = gz_detect_blinks(&frames, fps, 0.5);
        assert_eq!(b.len(), 1, "expected 1 blink, got {}", b.len());
        assert_eq!(b[0].start_step, 10);
    }

    #[test]
    fn test_detect_blinks_threshold_below_not_detected() {
        let fps = 60.0;
        // Blink value is 0.4 (below 0.5 threshold).
        let mut frames: Vec<GazeFrame> =
            (0..10).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
        for i in 10..15_u64 {
            frames.push(make_frame(i, 0.0, 0.0, 0.4, 0.0));
        }
        let b = gz_detect_blinks(&frames, fps, 0.5);
        assert!(b.is_empty(), "blink below threshold should not be detected");
    }

    #[test]
    fn test_detect_blinks_still_open_at_end() {
        // Blink starts but recording ends during blink → BlinkPhase::Closed.
        let fps = 60.0;
        let mut frames: Vec<GazeFrame> =
            (0..5).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
        for i in 5..10_u64 {
            frames.push(make_frame(i, 0.0, 0.0, 0.9, 0.0));
        }
        let b = gz_detect_blinks(&frames, fps, 0.5);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].phase, BlinkPhase::Closed);
    }

    // -----------------------------------------------------------------------
    // Dispersion
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispersion_empty() {
        assert_eq!(gz_dispersion(&[]), 0.0);
    }

    #[test]
    fn test_dispersion_single() {
        let g = GazeDirection::new(0.1, 0.2, 0.0);
        assert_eq!(gz_dispersion(&[g]), 0.0);
    }

    #[test]
    fn test_dispersion_identical() {
        let g = GazeDirection::new(0.1, 0.2, 0.0);
        let slice = vec![g; 10];
        assert!(
            gz_dispersion(&slice) < 1e-4,
            "identical gaze → near-zero dispersion"
        );
    }

    #[test]
    fn test_dispersion_spread() {
        let g1 = GazeDirection::new(0.0, 0.0, 0.0);
        let g2 = GazeDirection::new((10.0_f32).to_radians(), 0.0, 0.0);
        let d = gz_dispersion(&[g1, g2]);
        assert!(
            (d - 10.0).abs() < 0.1,
            "expected ~10 deg dispersion, got {d}"
        );
    }

    // -----------------------------------------------------------------------
    // Blink waveform
    // -----------------------------------------------------------------------

    #[test]
    fn test_blink_waveform_zero_at_start() {
        assert!((gz_blink_waveform(0.0, 100.0)).abs() < 1e-5);
    }

    #[test]
    fn test_blink_waveform_zero_at_end() {
        assert!((gz_blink_waveform(100.0, 100.0)).abs() < 1e-5);
    }

    #[test]
    fn test_blink_waveform_peak_at_half() {
        let v = gz_blink_waveform(50.0, 100.0);
        assert!(
            (v - 1.0).abs() < 1e-5,
            "waveform should peak at 1.0, got {v}"
        );
    }

    #[test]
    fn test_blink_waveform_monotone_first_half() {
        let duration = 200.0;
        let mut prev = 0.0_f32;
        for i in 0..=50_usize {
            let t = i as f32 * 2.0; // 0..100ms (first half)
            let v = gz_blink_waveform(t, duration);
            assert!(
                v >= prev - 1e-5,
                "waveform should monotonically increase in first half at t={t}"
            );
            prev = v;
        }
    }

    #[test]
    fn test_blink_waveform_monotone_second_half() {
        let duration = 200.0;
        let mut prev = 1.0_f32;
        for i in 50_usize..=100 {
            let t = i as f32 * 2.0; // 100..200ms (second half)
            let v = gz_blink_waveform(t, duration);
            assert!(
                v <= prev + 1e-5,
                "waveform should monotonically decrease in second half at t={t}"
            );
            prev = v;
        }
    }

    #[test]
    fn test_blink_waveform_zero_duration() {
        assert_eq!(gz_blink_waveform(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_blink_waveform_out_of_range() {
        // t > duration should clamp to 0 (end of blink).
        let v = gz_blink_waveform(999.0, 100.0);
        assert!(
            v.abs() < 1e-5,
            "waveform beyond duration should be ~0, got {v}"
        );
    }

    // -----------------------------------------------------------------------
    // Blink synthesis
    // -----------------------------------------------------------------------

    #[test]
    fn test_synthesize_blinks_output_length() {
        let out = gz_synthesize_blinks(600, 60.0, 15.0, 42);
        assert_eq!(out.len(), 600);
    }

    #[test]
    fn test_synthesize_blinks_values_in_range() {
        let out = gz_synthesize_blinks(600, 60.0, 15.0, 42);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "blink value out of range at {i}: {v}"
            );
        }
    }

    #[test]
    fn test_synthesize_blinks_has_blinks() {
        // At 60fps, 10min duration, 15 blinks/min → expect ~150 blinks.
        let n_steps = 60 * 60 * 10; // 10 min at 60fps
        let out = gz_synthesize_blinks(n_steps, 60.0, 15.0, 1234);
        let n_peaks: usize = out
            .windows(3)
            .filter(|w| w[1] > w[0] && w[1] > w[2] && w[1] > 0.5)
            .count();
        // Expect at least 50 and no more than 400 blink peaks for 10 min.
        assert!(
            (50..=400).contains(&n_peaks),
            "expected 50-400 blink peaks in 10min, got {n_peaks}"
        );
    }

    #[test]
    fn test_synthesize_blinks_zero_steps() {
        let out = gz_synthesize_blinks(0, 60.0, 15.0, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_synthesize_blinks_different_seeds_differ() {
        let a = gz_synthesize_blinks(600, 60.0, 15.0, 111);
        let b = gz_synthesize_blinks(600, 60.0, 15.0, 222);
        // At least some values should differ.
        let differs = a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6);
        assert!(
            differs,
            "different seeds should produce different sequences"
        );
    }

    // -----------------------------------------------------------------------
    // GazeController
    // -----------------------------------------------------------------------

    #[test]
    fn test_controller_new_ok() {
        let config = default_config();
        let ctrl = GazeController::new(config).expect("controller creation should succeed");
        assert!(ctrl.history.is_empty());
    }

    #[test]
    fn test_controller_new_invalid_fps() {
        let mut config = default_config();
        config.fps = -1.0;
        assert!(GazeController::new(config).is_err());
    }

    #[test]
    fn test_controller_push_frame() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        ctrl.push_frame(make_frame(0, 0.0, 0.0, 0.0, 0.0));
        assert_eq!(ctrl.history.len(), 1);
    }

    #[test]
    fn test_controller_history_cap() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        for i in 0..1200_u64 {
            ctrl.push_frame(make_frame(i, 0.0, 0.0, 0.0, 0.0));
        }
        assert!(
            ctrl.history.len() <= HISTORY_CAP,
            "history exceeds cap: {}",
            ctrl.history.len()
        );
    }

    #[test]
    fn test_controller_history_cap_exact() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        for i in 0..1000_u64 {
            ctrl.push_frame(make_frame(i, 0.0, 0.0, 0.0, 0.0));
        }
        assert_eq!(ctrl.history.len(), HISTORY_CAP);
        // Push one more.
        ctrl.push_frame(make_frame(1000, 0.0, 0.0, 0.0, 0.0));
        assert_eq!(ctrl.history.len(), HISTORY_CAP);
    }

    #[test]
    fn test_controller_mean_gaze_empty_error() {
        let config = default_config();
        let ctrl = GazeController::new(config).expect("new");
        assert!(ctrl.mean_gaze(5).is_err());
    }

    #[test]
    fn test_controller_mean_gaze_zero_n_error() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        ctrl.push_frame(make_frame(0, 0.1, 0.0, 0.0, 0.0));
        assert!(ctrl.mean_gaze(0).is_err());
    }

    #[test]
    fn test_controller_mean_gaze_correct() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        ctrl.push_frame(make_frame(0, 0.1, 0.0, 0.0, 0.0));
        ctrl.push_frame(make_frame(1, 0.3, 0.0, 0.0, 0.0));
        let mg = ctrl.mean_gaze(2).expect("mean gaze");
        assert!(
            (mg.azimuth - 0.2).abs() < 1e-5,
            "expected az=0.2, got {}",
            mg.azimuth
        );
    }

    #[test]
    fn test_controller_mean_gaze_n_clamped_to_history() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        ctrl.push_frame(make_frame(0, 0.0, 0.0, 0.0, 0.0));
        // n_last=10 with 1 frame → should use 1 frame without error.
        let mg = ctrl.mean_gaze(10).expect("mean gaze clamped");
        assert!((mg.azimuth).abs() < 1e-5);
    }

    #[test]
    fn test_controller_update_events_no_panic() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        for i in 0..30_u64 {
            ctrl.push_frame(make_frame(i, 0.0, 0.0, 0.0, i as f64));
        }
        ctrl.update_events(); // should not panic
    }

    #[test]
    fn test_controller_current_fixation_empty() {
        let config = default_config();
        let ctrl = GazeController::new(config).expect("new");
        assert!(ctrl.current_fixation().is_none());
    }

    #[test]
    fn test_controller_is_blinking_empty() {
        let config = default_config();
        let ctrl = GazeController::new(config).expect("new");
        assert!(!ctrl.is_blinking());
    }

    // -----------------------------------------------------------------------
    // Vergence
    // -----------------------------------------------------------------------

    #[test]
    fn test_vergence_from_iod_parallel_eyes() {
        let g = GazeDirection::new(0.0, 0.0, 0.0);
        // Parallel eyes → vergence = 0 (infinity).
        let v = gz_vergence_from_iod(&g, &g, 64.0).expect("vergence");
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_vergence_from_iod_known_geometry() {
        // IOD = 64 mm = 0.064 m. Object at 1 m.
        // Convergence angle ≈ 2 * atan(0.032 / 1.0) ≈ 3.66 deg ≈ 0.0639 rad.
        // Left eye converges inward (positive az), right eye converges inward (negative az).
        let iod_mm = 64.0_f32;
        let dist_m = 1.0_f32;
        let half_iod = iod_mm / 1000.0 / 2.0;
        let half_angle = (half_iod / dist_m).atan();
        let left = GazeDirection::new(-half_angle, 0.0, 0.0);
        let right = GazeDirection::new(half_angle, 0.0, 0.0);
        let v = gz_vergence_from_iod(&left, &right, iod_mm).expect("vergence");
        assert!((v - dist_m).abs() < 0.01, "expected ~1m, got {v}");
    }

    #[test]
    fn test_convergence_angle_deg_infinity() {
        let angle = gz_convergence_angle_deg(0.0, 0.064);
        assert_eq!(angle, 0.0);
    }

    #[test]
    fn test_convergence_angle_deg_one_meter() {
        // At 1m with 64mm IOD: ~3.67 deg.
        let angle = gz_convergence_angle_deg(1.0, 0.064);
        assert!(
            angle > 3.0 && angle < 4.5,
            "expected ~3.67 deg, got {angle}"
        );
    }

    #[test]
    fn test_convergence_angle_deg_increases_as_distance_decreases() {
        let a1 = gz_convergence_angle_deg(2.0, 0.064);
        let a2 = gz_convergence_angle_deg(0.5, 0.064);
        assert!(
            a2 > a1,
            "closer object should have larger convergence angle"
        );
    }

    #[test]
    fn test_convergence_angle_zero_iod() {
        let angle = gz_convergence_angle_deg(1.0, 0.0);
        assert_eq!(angle, 0.0);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_stats_empty_controller() {
        let config = default_config();
        let ctrl = GazeController::new(config).expect("new");
        let stats = gz_compute_stats(&ctrl, 60.0);
        assert_eq!(stats.n_frames, 0);
        assert_eq!(stats.n_saccades, 0);
        assert_eq!(stats.n_fixations, 0);
        assert_eq!(stats.n_blinks, 0);
        assert_eq!(stats.mean_fixation_dur_ms, 0.0);
    }

    #[test]
    fn test_compute_stats_frame_count() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        for i in 0..30_u64 {
            ctrl.push_frame(make_frame(i, 0.0, 0.0, 0.0, 0.0));
        }
        ctrl.update_events();
        let stats = gz_compute_stats(&ctrl, 60.0);
        assert_eq!(stats.n_frames, 30);
    }

    #[test]
    fn test_compute_stats_mean_vergence() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        // 10 frames with vergence 2.0.
        for i in 0..10_u64 {
            let gaze = GazeDirection::new(0.0, 0.0, 2.0);
            ctrl.push_frame(GazeFrame::monocular(i, gaze, 0.0));
        }
        let stats = gz_compute_stats(&ctrl, 60.0);
        assert!(
            (stats.mean_vergence_m - 2.0).abs() < 1e-5,
            "mean vergence should be 2.0"
        );
    }

    #[test]
    fn test_format_stats_contains_fields() {
        let stats = GazeStats {
            n_frames: 100,
            n_saccades: 5,
            n_fixations: 3,
            n_blinks: 2,
            mean_fixation_dur_ms: 200.0,
            mean_saccade_amplitude_deg: 8.5,
            blink_rate_per_min: 12.0,
            mean_vergence_m: 1.5,
        };
        let s = gz_format_stats(&stats);
        assert!(s.contains("100"), "should contain frame count");
        assert!(s.contains("saccades"), "should mention saccades");
        assert!(s.contains("blinks"), "should mention blinks");
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispersion_two_equal_points() {
        let g = GazeDirection::new(0.0, 0.0, 0.0);
        assert!(gz_dispersion(&[g, g]) < 1e-5);
    }

    #[test]
    fn test_angular_velocity_two_identical_frames() {
        let frames = vec![
            make_frame(0, 0.0, 0.0, 0.0, 0.0),
            make_frame(1, 0.0, 0.0, 0.0, 0.0),
        ];
        let vel = gz_angular_velocity(&frames, 60.0);
        assert_eq!(vel.len(), 1);
        assert!(vel[0].abs() < 1e-5);
    }

    #[test]
    fn test_blink_synthesis_rate_reasonable() {
        // 60 fps, 1 min (3600 frames), 10 blinks/min → ~10 blinks.
        let out = gz_synthesize_blinks(3600, 60.0, 10.0, 99);
        // Count transitions from below-0.5 to above-0.5.
        let blink_starts = out.windows(2).filter(|w| w[0] < 0.5 && w[1] >= 0.5).count();
        // Allow large tolerance due to exponential ISI variance.
        assert!(
            (3..=25).contains(&blink_starts),
            "expected 3-25 blinks for 10/min rate in 1min, got {blink_starts}"
        );
    }

    #[test]
    fn test_controller_history_ordering_preserved() {
        let config = default_config();
        let mut ctrl = GazeController::new(config).expect("new");
        for i in 0..10_u64 {
            ctrl.push_frame(make_frame(i, i as f32 * 0.01, 0.0, 0.0, 0.0));
        }
        // History should preserve insertion order.
        for (idx, frame) in ctrl.history.iter().enumerate() {
            assert_eq!(
                frame.step, idx as u64,
                "frame ordering mismatch at index {idx}"
            );
        }
    }

    #[test]
    fn test_listing_rotation_angle_correctness() {
        // primary=[0,0,1], target=[sin(30°), 0, cos(30°)].
        // Expected rotation angle around Y = 30°.
        let primary = [0.0_f32, 0.0, 1.0];
        let deg30 = (30.0_f32).to_radians();
        let target = [deg30.sin(), 0.0, deg30.cos()];
        let q = gz_listing_rotation(primary, target).expect("listing rotation");
        // The rotation angle = 2 * acos(qw).
        let angle = 2.0 * q[3].clamp(-1.0, 1.0).acos();
        assert!(
            (angle - deg30).abs() < 1e-4,
            "expected rotation angle ~30 deg ({deg30} rad), got {angle} rad"
        );
    }

    #[test]
    fn test_vergence_symmetry() {
        // Swapping left and right should give same result.
        let left = GazeDirection::new(-0.02, 0.0, 0.0);
        let right = GazeDirection::new(0.02, 0.0, 0.0);
        let v1 = gz_vergence_from_iod(&left, &right, 64.0).expect("v1");
        let v2 = gz_vergence_from_iod(&right, &left, 64.0).expect("v2");
        assert!((v1 - v2).abs() < 1e-6, "vergence should be symmetric");
    }
}
