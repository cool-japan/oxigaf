//! Gaze algorithm functions: Listing's law, angular velocity, I-VT
//! saccade/fixation detection, blink detection and synthesis, vergence
//! estimation, and aggregate statistics — plus the private vector/quaternion
//! math helpers they share.

use super::prng::{gz_seed_hash, xorshift64_f32};
use super::types::{
    BlinkEvent, BlinkPhase, FixationEvent, GazeController, GazeControllerError, GazeDirection,
    GazeFrame, GazeStats, SaccadeEvent,
};

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
/// That projection is mathematically a no-op here: `cross(primary,
/// target_dir)` is already perpendicular to both inputs, hence already in
/// Listing's plane (this only ever removes float noise). A single
/// primary→target rotation is thus unconditionally Listing-compliant, with
/// no "unconstrained" variant to gate behind a config flag — hence no
/// `listing_enforcement` field on [`super::types::GazeControllerConfig`].
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
            let end = i; // exclusive (velocity-index space)
                         // A run of k velocity samples spans k+1 frames; matches
                         // `gz_detect_fixations`'s convention for the same array.
            let n_frames = end - start + 1;
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
/// transitions from below `threshold` to at or above `threshold`. See
/// [`BlinkEvent::phase`] for what `phase` means on the returned events.
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
            // The value just crossed back below `threshold`: opening.
            events.push(BlinkEvent {
                duration_ms,
                phase: BlinkPhase::Opening,
                start_step: frames[blink_start].step,
            });
        }
    }
    // Still active at end of recording: derive the phase, don't assume Closed.
    if in_blink {
        let n_frames = frames.len() - blink_start;
        let duration_ms = n_frames as f32 * ms_per_frame;
        events.push(BlinkEvent {
            duration_ms,
            phase: trailing_blink_phase(frames, threshold),
            start_step: frames[blink_start].step,
        });
    }
    events
}

/// Trend of the last two blink values: rising → Closing, falling →
/// Opening, unchanging at/above `threshold` → Closed.
fn trailing_blink_phase(frames: &[GazeFrame], threshold: f32) -> BlinkPhase {
    let n = frames.len();
    if n < 2 {
        return BlinkPhase::Closed;
    }
    let last = frames[n - 1].mean_blink();
    let prev = frames[n - 2].mean_blink();
    let delta = last - prev;
    const FLAT_EPS: f32 = 1e-3;
    if delta > FLAT_EPS {
        BlinkPhase::Closing
    } else if delta < -FLAT_EPS {
        BlinkPhase::Opening
    } else if last >= threshold {
        BlinkPhase::Closed
    } else {
        // Flat-and-below-threshold shouldn't reach here (caller only
        // invokes this while `in_blink`), but avoid mislabelling as Closed.
        BlinkPhase::Opening
    }
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
/// Uses an xorshift64 PRNG with exponential inter-blink intervals for a
/// realistic, variable cadence. Each blink is rendered with
/// [`gz_blink_waveform`] over a `blink_duration_ms` window (non-positive →
/// default 150 ms). Returns `duration_steps` values in `[0, 1]`.
#[must_use]
pub fn gz_synthesize_blinks(
    duration_steps: usize,
    fps: f32,
    rate_per_min: f32,
    blink_duration_ms: f32,
    seed: u64,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; duration_steps];
    if duration_steps == 0 || fps <= 0.0 || rate_per_min <= 0.0 {
        return out;
    }
    let mean_interval_ms = 60_000.0 / rate_per_min;
    let blink_dur_ms = if blink_duration_ms > 0.0 {
        blink_duration_ms
    } else {
        150.0_f32
    };
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
