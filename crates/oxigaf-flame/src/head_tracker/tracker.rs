//! The live, stateful [`HeadTracker`]: rolling pose history with configurable
//! temporal filtering (none / EMA / SMA / One Euro) and outlier gating.

use super::functions::{interpolate_missing_frames, rotation_delta};
use super::one_euro::one_euro_step;
use super::types::{
    HeadTrackFrame, HeadTrackerConfig, HeadTrackerError, HeadTrajectory, TrackingFilter,
};

// ---------------------------------------------------------------------------
// HeadTracker
// ---------------------------------------------------------------------------

/// Per-channel One Euro filter state (previous filtered value and previous
/// derivative estimate).
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct OneEuroState {
    pub(super) x_prev: Option<f32>,
    pub(super) dx_prev: Option<f32>,
}

/// Real-time head tracker that maintains rolling pose history and applies filters.
pub struct HeadTracker {
    /// Active configuration.
    pub config: HeadTrackerConfig,
    /// Raw (unfiltered) frame history.
    history: Vec<HeadTrackFrame>,
    /// Filtered frame history (parallel to `history`).
    filtered_history: Vec<HeadTrackFrame>,
    /// One Euro filter state per pose channel, in `[yaw, pitch, roll, tx,
    /// ty, tz]` order.
    one_euro: [OneEuroState; 6],
    /// Timestamp of the last frame filtered by [`TrackingFilter::OneEuro`],
    /// used to derive the real inter-frame `dt` the filter's frequency
    /// adaptivity depends on.
    one_euro_prev_ts: Option<f32>,
    /// Most recent frame accepted as valid (post confidence/outlier
    /// gating in [`Self::update`]), used as the outlier-detection baseline.
    last_valid_raw: Option<HeadTrackFrame>,
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
            one_euro: [OneEuroState::default(); 6],
            one_euro_prev_ts: None,
            last_valid_raw: None,
        })
    }

    /// Push a new frame, apply the configured filter, trim history, and return
    /// a reference to the filtered frame just recorded.
    ///
    /// Before filtering, `frame.is_valid` is downgraded to `false`
    /// (whatever the caller passed) when either:
    /// - `frame.confidence < config.confidence_threshold`, or
    /// - the frame is not the first accepted one and its rotation has
    ///   jumped by more than `config.outlier_threshold` radians from the
    ///   last frame accepted as valid.
    pub fn update(&mut self, frame: HeadTrackFrame) -> &HeadTrackFrame {
        let mut incoming = frame;
        if incoming.confidence < self.config.confidence_threshold {
            incoming.is_valid = false;
        }
        if incoming.is_valid {
            if let Some(prev) = &self.last_valid_raw {
                if rotation_delta(&incoming, prev) > self.config.outlier_threshold {
                    incoming.is_valid = false;
                }
            }
        }
        if incoming.is_valid {
            self.last_valid_raw = Some(incoming.clone());
        }

        let filtered = self.apply_filter(&incoming);
        self.history.push(incoming);
        self.filtered_history.push(filtered);

        // Trim both histories to max_history. Clamped to >= 1 so the
        // element just pushed to `filtered_history` is never drained,
        // however `config.max_history` (a public field) happens to be set.
        let max = self.config.max_history.max(1);
        if self.history.len() > max {
            let drain = self.history.len() - max;
            self.history.drain(0..drain);
        }
        if self.filtered_history.len() > max {
            let drain = self.filtered_history.len() - max;
            self.filtered_history.drain(0..drain);
        }

        // `filtered_history` was just pushed to above and the trim never
        // removes the most recent element, so indexing the last position
        // directly is always in-bounds -- no fallible path to paper over.
        let last_idx = self.filtered_history.len() - 1;
        &self.filtered_history[last_idx]
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
                // Average exactly `window` *raw* samples ending at (and
                // including) the incoming frame. `self.history` holds raw
                // frames and, at this point in `update`, does not yet
                // contain `frame` itself -- averaging `filtered_history`
                // here previously made this a cascaded IIR filter with an
                // unbounded effective time constant instead of a bounded
                // FIR moving average, and used `window + 1` samples.
                let window = (*window).max(1);
                let hist_len = self.history.len();
                let count = window.min(hist_len + 1);
                let hist_count = count - 1;
                let start = hist_len - hist_count;
                let (mut yaw, mut pitch, mut roll, mut tx, mut ty, mut tz) =
                    (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for f in &self.history[start..] {
                    yaw += f.yaw;
                    pitch += f.pitch;
                    roll += f.roll;
                    tx += f.tx;
                    ty += f.ty;
                    tz += f.tz;
                }
                yaw += frame.yaw;
                pitch += frame.pitch;
                roll += frame.roll;
                tx += frame.tx;
                ty += frame.ty;
                tz += frame.tz;
                let n = count as f32;
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
                // Real inter-frame dt, not a fixed 30 fps assumption: the
                // filter's entire value is frequency adaptivity based on
                // actual sample timing.
                let dt_ms = match self.one_euro_prev_ts {
                    Some(prev_ts) => (frame.timestamp_ms - prev_ts).max(f32::EPSILON),
                    // No previous sample yet: every channel below takes the
                    // first-sample branch regardless of `dt_ms`.
                    None => 1.0,
                };
                self.one_euro_prev_ts = Some(frame.timestamp_ms);

                let channels = [
                    frame.yaw,
                    frame.pitch,
                    frame.roll,
                    frame.tx,
                    frame.ty,
                    frame.tz,
                ];
                let mut out = [0.0f32; 6];
                for ((o, &x), st) in out
                    .iter_mut()
                    .zip(channels.iter())
                    .zip(self.one_euro.iter_mut())
                {
                    *o = one_euro_step(x, dt_ms, st, min_cutoff, beta);
                }
                HeadTrackFrame {
                    yaw: out[0],
                    pitch: out[1],
                    roll: out[2],
                    tx: out[3],
                    ty: out[4],
                    tz: out[5],
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
        self.one_euro = [OneEuroState::default(); 6];
        self.one_euro_prev_ts = None;
        self.last_valid_raw = None;
    }

    /// Snapshot the raw (unfiltered) history as a standalone
    /// [`HeadTrajectory`] for the trajectory-level analysis functions in
    /// this module.
    ///
    /// # Errors
    ///
    /// Returns [`HeadTrackerError::EmptyTrajectory`] when no frames have
    /// been pushed yet.
    pub fn to_trajectory(&self) -> Result<HeadTrajectory, HeadTrackerError> {
        HeadTrajectory::new(self.history.clone())
    }

    /// Interpolate gaps in the tracked history, honouring
    /// `config.max_gap_frames` -- the field this method exists to make
    /// useful (see [`interpolate_missing_frames`]).
    ///
    /// # Errors
    ///
    /// Returns [`HeadTrackerError::EmptyTrajectory`] when no frames have
    /// been pushed yet.
    pub fn interpolate_gaps(&self) -> Result<HeadTrajectory, HeadTrackerError> {
        interpolate_missing_frames(&self.to_trajectory()?, self.config.max_gap_frames)
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
