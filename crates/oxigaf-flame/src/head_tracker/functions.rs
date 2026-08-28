//! Offline trajectory-analysis algorithms: temporal smoothing (EMA/SMA/One
//! Euro), gap interpolation, pose-jump / anomaly detection, rotation
//! velocity, aggregate statistics, motion segmentation, coverage scoring,
//! slicing, and resampling.

use super::one_euro::{alpha_from_cutoff, one_euro_cutoff};
use super::types::{HeadTrackFrame, HeadTrackerError, HeadTrajectory, TrajectoryStats};

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
/// For each maximal run of consecutive invalid frames no longer than
/// `max_gap_frames`:
/// - Both a left and a right valid neighbor exist: linear interpolation
///   between them, `is_valid` set to `true`.
/// - Only one side has a valid neighbor: filled with that neighbor's pose
///   (nearest-neighbor extrapolation), `is_valid` set to `true`.
/// - No valid neighbor anywhere in the trajectory: left unchanged
///   (`is_valid` stays `false`) -- there is no data to interpolate from, so
///   reporting these frames as tracked would misrepresent a total tracking
///   failure as a perfectly-still, fully-covered trajectory.
///
/// A run longer than `max_gap_frames` is always left unchanged
/// (`is_valid` stays `false`), regardless of neighbor availability.
///
/// # Errors
///
/// Returns [`HeadTrackerError::EmptyTrajectory`] when the trajectory is empty.
pub fn interpolate_missing_frames(
    trajectory: &HeadTrajectory,
    max_gap_frames: usize,
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

        if run_end - run_start > max_gap_frames {
            // Gap exceeds the caller's configured willingness to bridge
            // it; leave these frames invalid and move on.
            continue;
        }

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
                // No valid frame anywhere in the trajectory to interpolate
                // or extrapolate from: leave this run exactly as it was
                // (still invalid) rather than fabricating a "successfully
                // tracked, perfectly still" result.
            }
        }
    }
    HeadTrajectory::new(out)
}

// ---------------------------------------------------------------------------
// Anomaly detection
// ---------------------------------------------------------------------------

/// Euclidean magnitude of the yaw/pitch/roll delta between two frames
/// (radians), used both for offline jump detection and for
/// [`HeadTracker::update`]'s live outlier gate.
#[inline]
pub(super) fn rotation_delta(a: &HeadTrackFrame, b: &HeadTrackFrame) -> f32 {
    let d_yaw = a.yaw - b.yaw;
    let d_pitch = a.pitch - b.pitch;
    let d_roll = a.roll - b.roll;
    (d_yaw * d_yaw + d_pitch * d_pitch + d_roll * d_roll).sqrt()
}

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
            if rotation_delta(frame, prev) > threshold_rad {
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
/// Returns a vector of `(start_frame_idx, end_frame_idx, is_motion)` tuples
/// where the indices refer to positions within `trajectory.frames`.
/// Segments cover every frame index from `0` to `trajectory.len() - 1`:
/// consecutive segments share their boundary frame (the frame at which a
/// transition is detected belongs to both the segment that just ended and
/// the one that starts there), so one segment's `end_frame_idx` always
/// equals the next segment's `start_frame_idx`, and the last segment's
/// `end_frame_idx` is always `trajectory.len() - 1`.
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
