//! Animation export for OxiGAF Gaussian avatar sequences.
//!
//! This module provides tools for exporting, loading, resampling, and
//! processing Gaussian avatar animation sequences. Each frame contains all
//! Gaussian parameters at a given time step.
//!
//! Gaussians are described by flat `Vec<f32>` arrays:
//! - `positions`: N×3 (x, y, z)
//! - `rotations`: N×4 quaternion [qx, qy, qz, qw]
//! - `scales`: N×3 log-scale
//! - `opacities`: N (logit-space)
//! - `sh_coefficients`: N×C (spherical harmonics)
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::animation_export::{
//!     AnimationFrame, AnimationSequence, AnimExportConfig,
//!     export_animation_json, load_animation_json, format_animation_summary,
//! };
//!
//! let n = 100usize;
//! let frame = AnimationFrame {
//!     step: 0,
//!     timestamp_ms: 0.0,
//!     positions: vec![0.0f32; n * 3],
//!     rotations: vec![0.0f32; n * 4],
//!     scales: vec![0.0f32; n * 3],
//!     opacities: vec![0.0f32; n],
//!     sh_coefficients: vec![],
//! };
//! let seq = AnimationSequence::new(vec![frame], 30.0).expect("valid");
//! println!("{}", format_animation_summary(&seq));
//! ```

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// AnimationError
// ---------------------------------------------------------------------------

/// Errors that can occur during animation export operations.
#[derive(Debug, Error)]
pub enum AnimationError {
    /// Animation has no frames.
    #[error("Empty animation: no frames")]
    EmptyAnimation,

    /// Frame count mismatch.
    #[error("Frame count mismatch: expected {expected}, got {got}")]
    FrameCountMismatch { expected: usize, got: usize },

    /// Gaussian count mismatch between frames.
    #[error("Gaussian count mismatch: frame 0 has {n0} but frame {idx} has {ni}")]
    GaussianCountMismatch { n0: usize, idx: usize, ni: usize },

    /// Invalid FPS value.
    #[error("Invalid FPS {fps}: must be > 0")]
    InvalidFps { fps: f32 },

    /// Frame range is invalid for the animation length.
    #[error("Invalid frame range [{start}, {end}) for animation of {len} frames")]
    InvalidFrameRange {
        start: usize,
        end: usize,
        len: usize,
    },

    /// Cannot interpolate between frames with different Gaussian counts.
    #[error(
        "Interpolation error: cannot interpolate between frames with different Gaussian counts"
    )]
    InterpolationMismatch,

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// AnimationFrame
// ---------------------------------------------------------------------------

/// A single frame of a Gaussian avatar animation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationFrame {
    /// Training step or frame index.
    pub step: usize,
    /// Time in milliseconds.
    pub timestamp_ms: f64,
    /// N×3 position array (x, y, z).
    pub positions: Vec<f32>,
    /// N×4 rotation array as quaternion [qx, qy, qz, qw].
    pub rotations: Vec<f32>,
    /// N×3 log-scale array.
    pub scales: Vec<f32>,
    /// N opacity values in logit-space.
    pub opacities: Vec<f32>,
    /// N×C spherical harmonics coefficients (may be empty).
    pub sh_coefficients: Vec<f32>,
}

impl AnimationFrame {
    /// Returns the number of Gaussians in this frame.
    #[must_use]
    pub fn n_gaussians(&self) -> usize {
        self.positions.len() / 3
    }

    /// Validates that all arrays have consistent sizes.
    pub fn validate(&self) -> Result<(), AnimationError> {
        let n = self.n_gaussians();
        if self.rotations.len() != n * 4 {
            return Err(AnimationError::GaussianCountMismatch {
                n0: n,
                idx: 0,
                ni: self.rotations.len() / 4,
            });
        }
        if self.scales.len() != n * 3 {
            return Err(AnimationError::GaussianCountMismatch {
                n0: n,
                idx: 0,
                ni: self.scales.len() / 3,
            });
        }
        if self.opacities.len() != n {
            return Err(AnimationError::GaussianCountMismatch {
                n0: n,
                idx: 0,
                ni: self.opacities.len(),
            });
        }
        Ok(())
    }

    /// Creates an AnimationFrame with only positions filled; other arrays are empty.
    #[must_use]
    pub fn from_positions_only(step: usize, timestamp_ms: f64, positions: Vec<f32>) -> Self {
        Self {
            step,
            timestamp_ms,
            positions,
            rotations: Vec::new(),
            scales: Vec::new(),
            opacities: Vec::new(),
            sh_coefficients: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// AnimationMeta
// ---------------------------------------------------------------------------

/// Metadata describing an animation sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationMeta {
    /// Number of frames in the sequence.
    pub n_frames: usize,
    /// Number of Gaussians per frame.
    pub n_gaussians: usize,
    /// Frames per second.
    pub fps: f32,
    /// Total duration in milliseconds.
    pub duration_ms: f64,
    /// Whether any spherical harmonics coefficients are present.
    pub has_sh: bool,
    /// Spherical harmonics degree (0, 1, 2, or 3).
    pub sh_degree: usize,
    /// Creation timestamp string (ISO-8601 or placeholder).
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// AnimExportConfig
// ---------------------------------------------------------------------------

/// Configuration for animation export.
#[derive(Debug, Clone)]
pub struct AnimExportConfig {
    /// Frames per second (default: 30.0).
    pub fps: f32,
    /// Include SH coefficients (default: true).
    pub include_sh: bool,
    /// Decimal places for text formats (default: 6).
    pub precision: usize,
}

impl Default for AnimExportConfig {
    fn default() -> Self {
        Self {
            fps: 30.0,
            include_sh: true,
            precision: 6,
        }
    }
}

// ---------------------------------------------------------------------------
// AnimationSequence
// ---------------------------------------------------------------------------

/// A full animation sequence of Gaussian avatar frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationSequence {
    /// Ordered list of animation frames.
    pub frames: Vec<AnimationFrame>,
    /// Sequence metadata.
    pub meta: AnimationMeta,
}

impl AnimationSequence {
    /// Creates a new animation sequence from frames and FPS.
    ///
    /// Returns an error if frames is empty, fps <= 0, or frame sizes are inconsistent.
    pub fn new(frames: Vec<AnimationFrame>, fps: f32) -> Result<Self, AnimationError> {
        if frames.is_empty() {
            return Err(AnimationError::EmptyAnimation);
        }
        if fps <= 0.0 {
            return Err(AnimationError::InvalidFps { fps });
        }
        let meta = build_animation_meta(&frames, fps)?;
        let seq = Self { frames, meta };
        seq.validate()?;
        Ok(seq)
    }

    /// Validates that all frames have the same number of Gaussians.
    pub fn validate(&self) -> Result<(), AnimationError> {
        if self.frames.is_empty() {
            return Err(AnimationError::EmptyAnimation);
        }
        let n0 = self.frames[0].n_gaussians();
        for (idx, frame) in self.frames.iter().enumerate().skip(1) {
            let ni = frame.n_gaussians();
            if ni != n0 {
                return Err(AnimationError::GaussianCountMismatch { n0, idx, ni });
            }
        }
        Ok(())
    }

    /// Returns the total duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        self.frames.len() as f64 / self.meta.fps as f64 * 1000.0
    }

    /// Returns the nearest frame index for a given time in milliseconds, or
    /// `None` if the sequence is empty.
    #[must_use]
    pub fn frame_at_time(&self, time_ms: f64) -> Option<usize> {
        if self.frames.is_empty() {
            return None;
        }
        let fps = self.meta.fps as f64;
        let idx = (time_ms / 1000.0 * fps).round() as i64;
        let clamped = idx.clamp(0, (self.frames.len() as i64) - 1) as usize;
        Some(clamped)
    }
}

// ---------------------------------------------------------------------------
// FrameStats / AnimationStats
// ---------------------------------------------------------------------------

/// Per-frame statistics.
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Frame index.
    pub frame_idx: usize,
    /// Number of Gaussians.
    pub n_gaussians: usize,
    /// Mean opacity value.
    pub mean_opacity: f32,
    /// Mean scale (after exp).
    pub mean_scale: f32,
    /// Position centroid [x, y, z].
    pub position_centroid: [f32; 3],
}

/// Statistics aggregated across the entire animation.
#[derive(Debug, Clone)]
pub struct AnimationStats {
    /// Number of frames.
    pub n_frames: usize,
    /// Number of Gaussians per frame.
    pub n_gaussians: usize,
    /// Frames per second.
    pub fps: f32,
    /// Total duration in milliseconds.
    pub duration_ms: f64,
    /// Mean opacity across all frames.
    pub mean_opacity: f32,
    /// Standard deviation of opacity across all frames.
    pub opacity_std: f32,
    /// Average displacement of centroid across consecutive frames.
    pub position_drift: f32,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Linearly interpolate a scalar.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Linearly interpolate two equal-length slices into a new `Vec<f32>`.
///
/// If the slices differ in length, only the shorter length is produced (via
/// `zip`). Callers that need a hard guarantee of matching lengths (e.g.
/// [`interpolate_frames`]) must validate that themselves — this helper is a
/// low-level numeric primitive, not a validator.
fn lerp_vec(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| lerp(av, bv, t))
        .collect()
}

/// Spherically interpolate one quaternion `[qx, qy, qz, qw]` pair.
///
/// Inputs are expected to be unit quaternions, but this function never
/// divides by an exactly-zero norm: if the interpolated result is
/// (near-)zero — which can only happen for degenerate, non-unit input, since
/// slerp of two genuine unit quaternions is always itself unit-length — the
/// raw (unnormalised) interpolated value is returned instead of NaN.
///
/// Antipodal inputs (`dot(a, b) < 0`) represent the same rotation with
/// opposite sign; `b` is negated first so the interpolation takes the short
/// way around instead of spinning the long way.
fn slerp_quat(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    let mut bx = b[0];
    let mut by = b[1];
    let mut bz = b[2];
    let mut bw = b[3];
    let mut dot = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;

    if dot < 0.0 {
        bx = -bx;
        by = -by;
        bz = -bz;
        bw = -bw;
        dot = -dot;
    }

    let (wa, wb) = if dot > 0.9995 {
        // Nearly identical rotations: sin(theta) below would be near zero,
        // so fall back to (numerically stable) linear blending.
        (1.0 - t, t)
    } else {
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        (
            ((1.0 - t) * theta).sin() / sin_theta,
            (t * theta).sin() / sin_theta,
        )
    };

    let rx = wa * a[0] + wb * bx;
    let ry = wa * a[1] + wb * by;
    let rz = wa * a[2] + wb * bz;
    let rw = wa * a[3] + wb * bw;

    let norm_sq = rx * rx + ry * ry + rz * rz + rw * rw;
    if norm_sq < 1e-12 {
        [rx, ry, rz, rw]
    } else {
        let inv_norm = 1.0 / norm_sq.sqrt();
        [rx * inv_norm, ry * inv_norm, rz * inv_norm, rw * inv_norm]
    }
}

/// Spherically interpolate a rotation array of `[qx, qy, qz, qw]` quaternion
/// groups (see [`slerp_quat`]). Operates on `a.len().min(b.len()) / 4`
/// complete groups; callers needing a strict length match must validate
/// that first (see [`interpolate_frames`]).
fn slerp_vec(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let n = a.len().min(b.len()) / 4;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let qa = [a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]];
        let qb = [b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]];
        out.extend_from_slice(&slerp_quat(&qa, &qb, t));
    }
    out
}

/// Round `value` to `precision` decimal places for lossy text-format export.
///
/// Non-finite inputs, and precisions large enough to make the rounding scale
/// overflow (or the rounded result non-finite), are passed through
/// unchanged — this is a best-effort display aid, not a validated transform.
fn round_to_precision(value: f32, precision: usize) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let scale = 10f32.powi(precision.min(30) as i32);
    if !scale.is_finite() || scale <= 0.0 {
        return value;
    }
    let rounded = (value * scale).round() / scale;
    if rounded.is_finite() {
        rounded
    } else {
        value
    }
}

/// Compute the mean of a slice, returning 0.0 for an empty slice.
fn mean_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Compute the population standard deviation of a slice, returning 0.0 for
/// slices with fewer than 2 elements.
fn std_f32(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean_f32(values);
    let var = values.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / values.len() as f32;
    var.sqrt()
}

/// Infer the SH degree from the number of coefficients per Gaussian.
///
/// Expected coefficients per Gaussian per channel:
/// - degree 0: 1 coeff × 3 channels = 3
/// - degree 1: 4 coeffs × 3 channels = 12
/// - degree 2: 9 coeffs × 3 channels = 27
/// - degree 3: 16 coeffs × 3 channels = 48
fn infer_sh_degree(coeffs_per_gaussian: usize) -> usize {
    match coeffs_per_gaussian {
        0 | 3 => 0,
        12 => 1,
        27 => 2,
        48 => 3,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// build_animation_meta
// ---------------------------------------------------------------------------

/// Builds `AnimationMeta` from a slice of frames and an FPS value.
///
/// Returns an error if `frames` is empty or `fps <= 0`.
pub fn build_animation_meta(
    frames: &[AnimationFrame],
    fps: f32,
) -> Result<AnimationMeta, AnimationError> {
    if frames.is_empty() {
        return Err(AnimationError::EmptyAnimation);
    }
    if fps <= 0.0 {
        return Err(AnimationError::InvalidFps { fps });
    }

    let n_frames = frames.len();
    let n_gaussians = frames[0].n_gaussians();
    let duration_ms = n_frames as f64 / fps as f64 * 1000.0;
    let has_sh = !frames[0].sh_coefficients.is_empty();
    let sh_degree = if let Some(coeffs_per_gaussian) =
        frames[0].sh_coefficients.len().checked_div(n_gaussians)
    {
        infer_sh_degree(coeffs_per_gaussian)
    } else {
        0
    };

    Ok(AnimationMeta {
        n_frames,
        n_gaussians,
        fps,
        duration_ms,
        has_sh,
        sh_degree,
        created_at: "2026-01-01T00:00:00Z".to_string(),
    })
}

// ---------------------------------------------------------------------------
// resample_animation
// ---------------------------------------------------------------------------

/// Resamples the animation to a new FPS using nearest-frame selection.
///
/// Returns an error if `new_fps <= 0`.
pub fn resample_animation(
    sequence: AnimationSequence,
    new_fps: f32,
) -> Result<AnimationSequence, AnimationError> {
    if new_fps <= 0.0 {
        return Err(AnimationError::InvalidFps { fps: new_fps });
    }

    let duration_ms = sequence.duration_ms();
    let new_n = (duration_ms * new_fps as f64 / 1000.0).round() as usize;
    let n = sequence.frames.len();
    let orig_fps = sequence.meta.fps as f64;

    let mut new_frames = Vec::with_capacity(new_n.max(1));
    let count = if new_n == 0 { 1 } else { new_n };

    for i in 0..count {
        let t_ms = i as f64 / new_fps as f64 * 1000.0;
        let orig_idx = ((t_ms / 1000.0 * orig_fps).floor() as usize).min(n.saturating_sub(1));
        let mut frame = sequence.frames[orig_idx].clone();
        frame.timestamp_ms = t_ms;
        new_frames.push(frame);
    }

    AnimationSequence::new(new_frames, new_fps)
}

// ---------------------------------------------------------------------------
// trim_animation
// ---------------------------------------------------------------------------

/// Extracts a sub-sequence of frames `[start_frame, end_frame)` and
/// re-indexes timestamps starting from 0.
///
/// Returns an error if the range is invalid.
pub fn trim_animation(
    sequence: AnimationSequence,
    start_frame: usize,
    end_frame: usize,
) -> Result<AnimationSequence, AnimationError> {
    let len = sequence.frames.len();
    if start_frame >= end_frame || end_frame > len {
        return Err(AnimationError::InvalidFrameRange {
            start: start_frame,
            end: end_frame,
            len,
        });
    }

    let fps = sequence.meta.fps;
    let frames: Vec<AnimationFrame> = sequence.frames[start_frame..end_frame]
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut frame = f.clone();
            frame.timestamp_ms = i as f64 / fps as f64 * 1000.0;
            frame
        })
        .collect();

    AnimationSequence::new(frames, fps)
}

// ---------------------------------------------------------------------------
// reverse_animation
// ---------------------------------------------------------------------------

/// Reverses the frame order and re-assigns timestamps so frame i gets
/// timestamp `i / fps * 1000.0`.
#[must_use]
pub fn reverse_animation(sequence: AnimationSequence) -> AnimationSequence {
    let fps = sequence.meta.fps;
    let mut frames: Vec<AnimationFrame> = sequence.frames.into_iter().rev().collect();
    for (i, frame) in frames.iter_mut().enumerate() {
        frame.timestamp_ms = i as f64 / fps as f64 * 1000.0;
    }
    // Re-build meta to reflect the reversed order (n_frames stays the same).
    // We can safely unwrap here because the sequence was already validated on entry.
    let meta = build_animation_meta(&frames, fps).unwrap_or_else(|_| AnimationMeta {
        n_frames: frames.len(),
        n_gaussians: if frames.is_empty() {
            0
        } else {
            frames[0].n_gaussians()
        },
        fps,
        duration_ms: frames.len() as f64 / fps as f64 * 1000.0,
        has_sh: false,
        sh_degree: 0,
        created_at: "2026-01-01T00:00:00Z".to_string(),
    });
    AnimationSequence { frames, meta }
}

// ---------------------------------------------------------------------------
// concatenate_animations
// ---------------------------------------------------------------------------

/// Concatenates two animations.  Both must have the same number of Gaussians.
///
/// The timestamps of `b` are offset by `a`'s duration.
pub fn concatenate_animations(
    a: AnimationSequence,
    b: AnimationSequence,
) -> Result<AnimationSequence, AnimationError> {
    let na = if a.frames.is_empty() {
        0
    } else {
        a.frames[0].n_gaussians()
    };
    let nb = if b.frames.is_empty() {
        0
    } else {
        b.frames[0].n_gaussians()
    };
    if na != nb {
        return Err(AnimationError::GaussianCountMismatch {
            n0: na,
            idx: 1,
            ni: nb,
        });
    }

    let fps = a.meta.fps;
    let offset_ms = a.duration_ms();
    let mut frames = a.frames;
    for mut frame in b.frames {
        frame.timestamp_ms += offset_ms;
        frames.push(frame);
    }

    AnimationSequence::new(frames, fps)
}

// ---------------------------------------------------------------------------
// interpolate_frames
// ---------------------------------------------------------------------------

/// Linearly interpolates between two frames at parameter `t ∈ [0, 1]`.
///
/// `rotations` are interpolated with spherical (slerp), not linear,
/// interpolation, since they are unit quaternions `[qx, qy, qz, qw]` and a
/// component-wise lerp would neither preserve unit norm nor take the short
/// way around for antipodal pairs.
///
/// Both frames are validated individually (see [`AnimationFrame::validate`])
/// and their `rotations`, `scales`, `opacities`, and `sh_coefficients`
/// arrays must all have matching lengths — a mismatch (e.g. interpolating a
/// fully-populated frame with one built by
/// [`AnimationFrame::from_positions_only`]) returns an error instead of
/// silently producing a corrupted frame.
///
/// # Errors
///
/// Returns [`AnimationError::InterpolationMismatch`] if the frames have
/// different numbers of Gaussians, or if any corresponding array pair
/// differs in length. Returns [`AnimationError::GaussianCountMismatch`] if
/// either frame is internally inconsistent (see
/// [`AnimationFrame::validate`]).
pub fn interpolate_frames(
    frame_a: &AnimationFrame,
    frame_b: &AnimationFrame,
    t: f32,
) -> Result<AnimationFrame, AnimationError> {
    if frame_a.n_gaussians() != frame_b.n_gaussians() {
        return Err(AnimationError::InterpolationMismatch);
    }
    frame_a.validate()?;
    frame_b.validate()?;
    if frame_a.rotations.len() != frame_b.rotations.len()
        || frame_a.scales.len() != frame_b.scales.len()
        || frame_a.opacities.len() != frame_b.opacities.len()
        || frame_a.sh_coefficients.len() != frame_b.sh_coefficients.len()
    {
        return Err(AnimationError::InterpolationMismatch);
    }

    let timestamp_ms =
        frame_a.timestamp_ms + (frame_b.timestamp_ms - frame_a.timestamp_ms) * t as f64;
    let positions = lerp_vec(&frame_a.positions, &frame_b.positions, t);
    let rotations = slerp_vec(&frame_a.rotations, &frame_b.rotations, t);
    let scales = lerp_vec(&frame_a.scales, &frame_b.scales, t);
    let opacities = lerp_vec(&frame_a.opacities, &frame_b.opacities, t);
    let sh_coefficients = lerp_vec(&frame_a.sh_coefficients, &frame_b.sh_coefficients, t);

    Ok(AnimationFrame {
        step: frame_a.step,
        timestamp_ms,
        positions,
        rotations,
        scales,
        opacities,
        sh_coefficients,
    })
}

// ---------------------------------------------------------------------------
// subsample_animation
// ---------------------------------------------------------------------------

/// Keeps every `stride`-th frame from the sequence.
///
/// Returns an error if `stride == 0`.
pub fn subsample_animation(
    sequence: AnimationSequence,
    stride: usize,
) -> Result<AnimationSequence, AnimationError> {
    if stride == 0 {
        return Err(AnimationError::InvalidFrameRange {
            start: 0,
            end: 0,
            len: sequence.frames.len(),
        });
    }

    let fps = sequence.meta.fps;
    let frames: Vec<AnimationFrame> = sequence
        .frames
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % stride == 0)
        .map(|(_, f)| f)
        .collect();

    if frames.is_empty() {
        return Err(AnimationError::EmptyAnimation);
    }

    AnimationSequence::new(frames, fps)
}

// ---------------------------------------------------------------------------
// compute_frame_stats
// ---------------------------------------------------------------------------

/// Computes per-frame statistics.
#[must_use]
pub fn compute_frame_stats(frame: &AnimationFrame) -> FrameStats {
    let n = frame.n_gaussians();
    let frame_idx = frame.step;

    let mean_opacity = mean_f32(&frame.opacities);

    let mean_scale = if frame.scales.is_empty() {
        0.0
    } else {
        let exp_scales: Vec<f32> = frame.scales.iter().map(|&s| s.exp()).collect();
        mean_f32(&exp_scales)
    };

    let position_centroid = if n == 0 {
        [0.0f32; 3]
    } else {
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cz = 0.0f32;
        for i in 0..n {
            cx += frame.positions[i * 3];
            cy += frame.positions[i * 3 + 1];
            cz += frame.positions[i * 3 + 2];
        }
        let nf = n as f32;
        [cx / nf, cy / nf, cz / nf]
    };

    FrameStats {
        frame_idx,
        n_gaussians: n,
        mean_opacity,
        mean_scale,
        position_centroid,
    }
}

// ---------------------------------------------------------------------------
// compute_animation_stats
// ---------------------------------------------------------------------------

/// Computes aggregate statistics across the entire animation.
#[must_use]
pub fn compute_animation_stats(sequence: &AnimationSequence) -> AnimationStats {
    let n_frames = sequence.frames.len();
    let n_gaussians = sequence.meta.n_gaussians;
    let fps = sequence.meta.fps;
    let duration_ms = sequence.duration_ms();

    if n_frames == 0 {
        return AnimationStats {
            n_frames: 0,
            n_gaussians,
            fps,
            duration_ms,
            mean_opacity: 0.0,
            opacity_std: 0.0,
            position_drift: 0.0,
        };
    }

    let frame_stats: Vec<FrameStats> = sequence.frames.iter().map(compute_frame_stats).collect();

    let all_opacities: Vec<f32> = frame_stats.iter().map(|s| s.mean_opacity).collect();
    let mean_opacity = mean_f32(&all_opacities);
    let opacity_std = std_f32(&all_opacities);

    let position_drift = if n_frames < 2 {
        0.0
    } else {
        let mut total_drift = 0.0f32;
        for w in frame_stats.windows(2) {
            let a = &w[0].position_centroid;
            let b = &w[1].position_centroid;
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let dz = b[2] - a[2];
            total_drift += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        total_drift / (n_frames - 1) as f32
    };

    AnimationStats {
        n_frames,
        n_gaussians,
        fps,
        duration_ms,
        mean_opacity,
        opacity_std,
        position_drift,
    }
}

// ---------------------------------------------------------------------------
// export_animation_json / load_animation_json
// ---------------------------------------------------------------------------

/// Exports the animation sequence to a JSON file at `path`, honouring
/// `config`:
///
/// - `include_sh`: when `false`, `sh_coefficients` are cleared from every
///   exported frame (the source `sequence` is not modified).
/// - `precision`: every `f32` value is rounded to this many decimal places
///   before serialisation (see [`round_to_precision`] for the exact,
///   overflow-safe semantics).
/// - `fps`: when `> 0.0`, `meta` is rebuilt using this FPS (via
///   [`build_animation_meta`]) so `meta.fps`/`meta.duration_ms` reflect the
///   requested export rate; otherwise the sequence's own `meta.fps` is kept.
///   Either way, `meta` is rebuilt from the (possibly SH-cleared) frames
///   actually being written, so `has_sh`/`sh_degree` stay consistent with
///   `include_sh`.
pub fn export_animation_json<P: AsRef<Path>>(
    sequence: &AnimationSequence,
    path: P,
    config: &AnimExportConfig,
) -> Result<(), AnimationError> {
    let mut export_seq = sequence.clone();

    if !config.include_sh {
        for frame in &mut export_seq.frames {
            frame.sh_coefficients.clear();
        }
    }

    for frame in &mut export_seq.frames {
        for v in frame.positions.iter_mut() {
            *v = round_to_precision(*v, config.precision);
        }
        for v in frame.rotations.iter_mut() {
            *v = round_to_precision(*v, config.precision);
        }
        for v in frame.scales.iter_mut() {
            *v = round_to_precision(*v, config.precision);
        }
        for v in frame.opacities.iter_mut() {
            *v = round_to_precision(*v, config.precision);
        }
        for v in frame.sh_coefficients.iter_mut() {
            *v = round_to_precision(*v, config.precision);
        }
    }

    let export_fps = if config.fps > 0.0 {
        config.fps
    } else {
        export_seq.meta.fps
    };
    if let Ok(new_meta) = build_animation_meta(&export_seq.frames, export_fps) {
        export_seq.meta = new_meta;
    }

    let json = serde_json::to_string(&export_seq)?;
    fs::write(path, json)?;
    Ok(())
}

/// Loads an animation sequence from a JSON file.
pub fn load_animation_json<P: AsRef<Path>>(path: P) -> Result<AnimationSequence, AnimationError> {
    let data = fs::read_to_string(path)?;
    let sequence: AnimationSequence = serde_json::from_str(&data)?;
    Ok(sequence)
}

// ---------------------------------------------------------------------------
// format_animation_summary
// ---------------------------------------------------------------------------

/// Returns a human-readable summary of the animation.
#[must_use]
pub fn format_animation_summary(sequence: &AnimationSequence) -> String {
    let duration_s = sequence.duration_ms() / 1000.0;
    format!(
        "Animation: {} frames, {} Gaussians, {:.1} fps, {:.2}s duration",
        sequence.meta.n_frames, sequence.meta.n_gaussians, sequence.meta.fps, duration_s,
    )
}

// ---------------------------------------------------------------------------
// loop_animation
// ---------------------------------------------------------------------------

/// Repeats the animation `n_repeats` times, offsetting timestamps each repeat.
///
/// Returns an error if `n_repeats == 0`.
pub fn loop_animation(
    sequence: AnimationSequence,
    n_repeats: usize,
) -> Result<AnimationSequence, AnimationError> {
    if n_repeats == 0 {
        return Err(AnimationError::EmptyAnimation);
    }

    let fps = sequence.meta.fps;
    let base_duration_ms = sequence.duration_ms();
    let mut frames: Vec<AnimationFrame> = Vec::with_capacity(sequence.frames.len() * n_repeats);

    for rep in 0..n_repeats {
        let offset_ms = rep as f64 * base_duration_ms;
        for frame in &sequence.frames {
            let mut f = frame.clone();
            f.timestamp_ms = frame.timestamp_ms + offset_ms;
            frames.push(f);
        }
    }

    AnimationSequence::new(frames, fps)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --------------- Helper builders ----------------------------------------

    fn make_frame(step: usize, timestamp_ms: f64, n: usize) -> AnimationFrame {
        AnimationFrame {
            step,
            timestamp_ms,
            positions: vec![0.0f32; n * 3],
            rotations: vec![0.0f32; n * 4],
            scales: vec![0.0f32; n * 3],
            opacities: vec![0.5f32; n],
            sh_coefficients: Vec::new(),
        }
    }

    fn make_sequence(n_frames: usize, n_gaussians: usize, fps: f32) -> AnimationSequence {
        let frames: Vec<AnimationFrame> = (0..n_frames)
            .map(|i| make_frame(i, i as f64 / fps as f64 * 1000.0, n_gaussians))
            .collect();
        AnimationSequence::new(frames, fps).expect("valid sequence")
    }

    // --------------- AnimationFrame::n_gaussians ----------------------------

    #[test]
    fn test_n_gaussians_zero() {
        let frame = make_frame(0, 0.0, 0);
        assert_eq!(frame.n_gaussians(), 0);
    }

    #[test]
    fn test_n_gaussians_ten() {
        let frame = make_frame(0, 0.0, 10);
        assert_eq!(frame.n_gaussians(), 10);
    }

    #[test]
    fn test_n_gaussians_one_hundred() {
        let frame = make_frame(0, 0.0, 100);
        assert_eq!(frame.n_gaussians(), 100);
    }

    // --------------- AnimationFrame::validate -------------------------------

    #[test]
    fn test_frame_validate_ok() {
        let frame = make_frame(0, 0.0, 50);
        assert!(frame.validate().is_ok());
    }

    #[test]
    fn test_frame_validate_rotation_mismatch() {
        let mut frame = make_frame(0, 0.0, 10);
        frame.rotations.push(0.0); // wrong length
        assert!(frame.validate().is_err());
    }

    #[test]
    fn test_frame_validate_scale_mismatch() {
        let mut frame = make_frame(0, 0.0, 10);
        frame.scales.push(0.0); // wrong length
        assert!(frame.validate().is_err());
    }

    #[test]
    fn test_frame_validate_opacity_mismatch() {
        let mut frame = make_frame(0, 0.0, 10);
        frame.opacities.push(0.0); // wrong length
        assert!(frame.validate().is_err());
    }

    // --------------- AnimationFrame::from_positions_only --------------------

    #[test]
    fn test_from_positions_only() {
        let positions = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let frame = AnimationFrame::from_positions_only(5, 100.0, positions.clone());
        assert_eq!(frame.step, 5);
        assert_eq!(frame.timestamp_ms, 100.0);
        assert_eq!(frame.positions, positions);
        assert!(frame.rotations.is_empty());
        assert!(frame.scales.is_empty());
        assert!(frame.opacities.is_empty());
        assert!(frame.sh_coefficients.is_empty());
    }

    // --------------- AnimationSequence::new ---------------------------------

    #[test]
    fn test_sequence_new_ok() {
        let seq = make_sequence(10, 50, 30.0);
        assert_eq!(seq.meta.n_frames, 10);
        assert_eq!(seq.meta.n_gaussians, 50);
    }

    #[test]
    fn test_sequence_new_empty_error() {
        let result = AnimationSequence::new(vec![], 30.0);
        assert!(matches!(result, Err(AnimationError::EmptyAnimation)));
    }

    #[test]
    fn test_sequence_new_zero_fps_error() {
        let frames = vec![make_frame(0, 0.0, 10)];
        let result = AnimationSequence::new(frames, 0.0);
        assert!(matches!(result, Err(AnimationError::InvalidFps { .. })));
    }

    #[test]
    fn test_sequence_new_negative_fps_error() {
        let frames = vec![make_frame(0, 0.0, 10)];
        let result = AnimationSequence::new(frames, -1.0);
        assert!(matches!(result, Err(AnimationError::InvalidFps { .. })));
    }

    // --------------- AnimationSequence::validate ----------------------------

    #[test]
    fn test_sequence_validate_mismatched_gaussians() {
        let frame0 = make_frame(0, 0.0, 10);
        let frame1 = make_frame(1, 33.3, 20); // different count
        let seq = AnimationSequence {
            frames: vec![frame0, frame1],
            meta: AnimationMeta {
                n_frames: 2,
                n_gaussians: 10,
                fps: 30.0,
                duration_ms: 66.6,
                has_sh: false,
                sh_degree: 0,
                created_at: "2026-01-01T00:00:00Z".to_string(),
            },
        };
        // Manually corrupt: seq is already inconsistent via construction
        let result = seq.validate();
        assert!(matches!(
            result,
            Err(AnimationError::GaussianCountMismatch { .. })
        ));
        drop(seq);
    }

    // --------------- AnimationSequence::duration_ms -------------------------

    #[test]
    fn test_duration_ms_formula() {
        let seq = make_sequence(30, 100, 30.0);
        // 30 frames / 30 fps * 1000 = 1000 ms
        let d = seq.duration_ms();
        assert!((d - 1000.0).abs() < 1e-3, "expected 1000 ms, got {d}");
    }

    // --------------- AnimationSequence::frame_at_time -----------------------

    #[test]
    fn test_frame_at_time_zero() {
        let seq = make_sequence(10, 5, 10.0);
        assert_eq!(seq.frame_at_time(0.0), Some(0));
    }

    #[test]
    fn test_frame_at_time_last() {
        let seq = make_sequence(10, 5, 10.0);
        // 10 frames at 10fps → last frame at 900ms
        assert_eq!(seq.frame_at_time(900.0), Some(9));
    }

    #[test]
    fn test_frame_at_time_clamp_beyond_end() {
        let seq = make_sequence(5, 5, 10.0);
        // Way beyond duration → clamp to last frame
        assert_eq!(seq.frame_at_time(9999.0), Some(4));
    }

    #[test]
    fn test_frame_at_time_middle() {
        let seq = make_sequence(10, 5, 10.0);
        // 500 ms → frame 5
        assert_eq!(seq.frame_at_time(500.0), Some(5));
    }

    // --------------- build_animation_meta -----------------------------------

    #[test]
    fn test_build_meta_duration() {
        let frames = vec![make_frame(0, 0.0, 20), make_frame(1, 100.0, 20)];
        let meta = build_animation_meta(&frames, 10.0).expect("ok");
        // 2 frames / 10 fps * 1000 = 200 ms
        assert!((meta.duration_ms - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_meta_sh_degree_0_no_sh() {
        let frames = vec![make_frame(0, 0.0, 10)];
        let meta = build_animation_meta(&frames, 30.0).expect("ok");
        assert!(!meta.has_sh);
        assert_eq!(meta.sh_degree, 0);
    }

    #[test]
    fn test_build_meta_sh_degree_1() {
        let n = 5usize;
        let mut frame = make_frame(0, 0.0, n);
        frame.sh_coefficients = vec![0.0f32; n * 12]; // degree 1: 12 per gaussian
        let frames = vec![frame];
        let meta = build_animation_meta(&frames, 30.0).expect("ok");
        assert!(meta.has_sh);
        assert_eq!(meta.sh_degree, 1);
    }

    #[test]
    fn test_build_meta_sh_degree_2() {
        let n = 5usize;
        let mut frame = make_frame(0, 0.0, n);
        frame.sh_coefficients = vec![0.0f32; n * 27]; // degree 2
        let frames = vec![frame];
        let meta = build_animation_meta(&frames, 30.0).expect("ok");
        assert_eq!(meta.sh_degree, 2);
    }

    #[test]
    fn test_build_meta_sh_degree_3() {
        let n = 5usize;
        let mut frame = make_frame(0, 0.0, n);
        frame.sh_coefficients = vec![0.0f32; n * 48]; // degree 3
        let frames = vec![frame];
        let meta = build_animation_meta(&frames, 30.0).expect("ok");
        assert_eq!(meta.sh_degree, 3);
    }

    #[test]
    fn test_build_meta_empty_error() {
        let result = build_animation_meta(&[], 30.0);
        assert!(matches!(result, Err(AnimationError::EmptyAnimation)));
    }

    // --------------- resample_animation -------------------------------------

    #[test]
    fn test_resample_invalid_fps() {
        let seq = make_sequence(10, 5, 30.0);
        let result = resample_animation(seq, 0.0);
        assert!(matches!(result, Err(AnimationError::InvalidFps { .. })));
    }

    #[test]
    fn test_resample_same_fps() {
        let seq = make_sequence(30, 10, 30.0);
        let resampled = resample_animation(seq, 30.0).expect("ok");
        assert_eq!(resampled.meta.n_frames, 30);
    }

    #[test]
    fn test_resample_to_half_fps() {
        // 30 frames at 30fps = 1000ms; at 15fps = 15 frames
        let seq = make_sequence(30, 10, 30.0);
        let resampled = resample_animation(seq, 15.0).expect("ok");
        assert_eq!(resampled.meta.fps, 15.0);
        // 1000ms * 15fps / 1000 = 15 frames
        assert_eq!(resampled.meta.n_frames, 15);
    }

    #[test]
    fn test_resample_to_double_fps() {
        // 10 frames at 10fps = 1000ms; at 20fps = 20 frames
        let seq = make_sequence(10, 5, 10.0);
        let resampled = resample_animation(seq, 20.0).expect("ok");
        assert_eq!(resampled.meta.fps, 20.0);
        assert_eq!(resampled.meta.n_frames, 20);
    }

    // --------------- trim_animation -----------------------------------------

    #[test]
    fn test_trim_valid() {
        let seq = make_sequence(20, 10, 30.0);
        let trimmed = trim_animation(seq, 5, 15).expect("ok");
        assert_eq!(trimmed.meta.n_frames, 10);
        // first frame timestamp should be 0
        assert!((trimmed.frames[0].timestamp_ms).abs() < 1e-6);
    }

    #[test]
    fn test_trim_invalid_range_start_ge_end() {
        let seq = make_sequence(10, 5, 30.0);
        let result = trim_animation(seq, 5, 5);
        assert!(matches!(
            result,
            Err(AnimationError::InvalidFrameRange { .. })
        ));
    }

    #[test]
    fn test_trim_invalid_range_end_exceeds_len() {
        let seq = make_sequence(10, 5, 30.0);
        let result = trim_animation(seq, 0, 15);
        assert!(matches!(
            result,
            Err(AnimationError::InvalidFrameRange { .. })
        ));
    }

    #[test]
    fn test_trim_full_range() {
        let seq = make_sequence(10, 5, 30.0);
        let trimmed = trim_animation(seq, 0, 10).expect("ok");
        assert_eq!(trimmed.meta.n_frames, 10);
    }

    // --------------- reverse_animation --------------------------------------

    #[test]
    fn test_reverse_swaps_frames() {
        let seq = make_sequence(5, 10, 30.0);
        let first_step = seq.frames[0].step;
        let last_step = seq.frames[4].step;
        let reversed = reverse_animation(seq);
        assert_eq!(reversed.frames[0].step, last_step);
        assert_eq!(reversed.frames[4].step, first_step);
    }

    #[test]
    fn test_reverse_timestamps_re_indexed() {
        let fps = 10.0f32;
        let seq = make_sequence(5, 10, fps);
        let reversed = reverse_animation(seq);
        // Frame 0 timestamp must be 0
        assert!((reversed.frames[0].timestamp_ms).abs() < 1e-6);
        // Frame 1 timestamp must be 100ms (1/10fps*1000)
        assert!((reversed.frames[1].timestamp_ms - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_reverse_preserves_frame_count() {
        let seq = make_sequence(7, 5, 24.0);
        let reversed = reverse_animation(seq);
        assert_eq!(reversed.meta.n_frames, 7);
    }

    // --------------- concatenate_animations ---------------------------------

    #[test]
    fn test_concatenate_total_frames() {
        let a = make_sequence(5, 10, 30.0);
        let b = make_sequence(3, 10, 30.0);
        let cat = concatenate_animations(a, b).expect("ok");
        assert_eq!(cat.meta.n_frames, 8);
    }

    #[test]
    fn test_concatenate_timestamps_continuous() {
        let a = make_sequence(3, 5, 10.0);
        let b = make_sequence(3, 5, 10.0);
        let a_dur = a.duration_ms();
        let cat = concatenate_animations(a, b).expect("ok");
        // Frame 3 should start at a's duration
        assert!((cat.frames[3].timestamp_ms - a_dur).abs() < 1e-3);
    }

    #[test]
    fn test_concatenate_mismatch_error() {
        let a = make_sequence(3, 10, 30.0);
        let b = make_sequence(3, 20, 30.0);
        let result = concatenate_animations(a, b);
        assert!(matches!(
            result,
            Err(AnimationError::GaussianCountMismatch { .. })
        ));
    }

    // --------------- interpolate_frames -------------------------------------

    #[test]
    fn test_interpolate_t0_equals_a() {
        let n = 4usize;
        let frame_a = AnimationFrame {
            step: 0,
            timestamp_ms: 0.0,
            positions: vec![1.0f32; n * 3],
            rotations: vec![0.0f32; n * 4],
            scales: vec![0.0f32; n * 3],
            opacities: vec![0.3f32; n],
            sh_coefficients: Vec::new(),
        };
        let frame_b = AnimationFrame {
            step: 1,
            timestamp_ms: 100.0,
            positions: vec![3.0f32; n * 3],
            rotations: vec![1.0f32; n * 4],
            scales: vec![1.0f32; n * 3],
            opacities: vec![0.7f32; n],
            sh_coefficients: Vec::new(),
        };
        let interp = interpolate_frames(&frame_a, &frame_b, 0.0).expect("ok");
        assert!((interp.positions[0] - 1.0).abs() < 1e-6);
        assert!((interp.opacities[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_t1_equals_b() {
        let n = 4usize;
        let frame_a = AnimationFrame {
            step: 0,
            timestamp_ms: 0.0,
            positions: vec![1.0f32; n * 3],
            rotations: vec![0.0f32; n * 4],
            scales: vec![0.0f32; n * 3],
            opacities: vec![0.3f32; n],
            sh_coefficients: Vec::new(),
        };
        let frame_b = AnimationFrame {
            step: 1,
            timestamp_ms: 100.0,
            positions: vec![3.0f32; n * 3],
            rotations: vec![1.0f32; n * 4],
            scales: vec![1.0f32; n * 3],
            opacities: vec![0.7f32; n],
            sh_coefficients: Vec::new(),
        };
        let interp = interpolate_frames(&frame_a, &frame_b, 1.0).expect("ok");
        assert!((interp.positions[0] - 3.0).abs() < 1e-6);
        assert!((interp.opacities[0] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_t_half() {
        let n = 2usize;
        let frame_a = AnimationFrame {
            step: 0,
            timestamp_ms: 0.0,
            positions: vec![0.0f32; n * 3],
            rotations: vec![0.0f32; n * 4],
            scales: vec![0.0f32; n * 3],
            opacities: vec![0.0f32; n],
            sh_coefficients: Vec::new(),
        };
        let frame_b = AnimationFrame {
            step: 1,
            timestamp_ms: 200.0,
            positions: vec![2.0f32; n * 3],
            rotations: vec![0.0f32; n * 4],
            scales: vec![0.0f32; n * 3],
            opacities: vec![1.0f32; n],
            sh_coefficients: Vec::new(),
        };
        let interp = interpolate_frames(&frame_a, &frame_b, 0.5).expect("ok");
        assert!((interp.positions[0] - 1.0).abs() < 1e-6);
        assert!((interp.timestamp_ms - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_count_mismatch() {
        let frame_a = make_frame(0, 0.0, 10);
        let frame_b = make_frame(1, 100.0, 20);
        let result = interpolate_frames(&frame_a, &frame_b, 0.5);
        assert!(matches!(result, Err(AnimationError::InterpolationMismatch)));
    }

    #[test]
    fn test_interpolate_rotations_stay_unit_norm_90_degrees() {
        // identity -> 90-degree rotation about Z.
        let half = std::f32::consts::FRAC_PI_4; // 45 degrees in radians
        let frame_a = AnimationFrame {
            step: 0,
            timestamp_ms: 0.0,
            positions: vec![0.0f32; 3],
            rotations: vec![0.0, 0.0, 0.0, 1.0],
            scales: vec![0.0f32; 3],
            opacities: vec![0.0f32; 1],
            sh_coefficients: Vec::new(),
        };
        let frame_b = AnimationFrame {
            step: 1,
            timestamp_ms: 100.0,
            positions: vec![0.0f32; 3],
            rotations: vec![0.0, 0.0, half.sin(), half.cos()],
            scales: vec![0.0f32; 3],
            opacities: vec![0.0f32; 1],
            sh_coefficients: Vec::new(),
        };

        for &t in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let interp = interpolate_frames(&frame_a, &frame_b, t).expect("ok");
            let q = &interp.rotations;
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "t={t}: expected unit norm, got {norm} (q={q:?})"
            );
        }
    }

    #[test]
    fn test_interpolate_rotations_no_nan_on_degenerate_zero_quaternions() {
        // A regression guard: an all-zero "quaternion" is not a valid unit
        // quaternion, but interpolate_frames must not produce NaN via a
        // divide-by-zero in the slerp renormalisation step.
        let frame_a = make_frame(0, 0.0, 1);
        let frame_b = frame_a.clone();
        let interp = interpolate_frames(&frame_a, &frame_b, 0.5).expect("ok");
        assert!(interp.rotations.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_interpolate_rotations_antipodal_short_path() {
        // `a` and `-a` represent the same rotation; slerp between them
        // (dot < 0) must take the short way and stay unit-norm.
        let a = [0.0f32, 0.0, 0.707_106_8, 0.707_106_8]; // 90 degrees about Z
        let frame_a = AnimationFrame {
            step: 0,
            timestamp_ms: 0.0,
            positions: vec![0.0f32; 3],
            rotations: a.to_vec(),
            scales: vec![0.0f32; 3],
            opacities: vec![0.0f32; 1],
            sh_coefficients: Vec::new(),
        };
        let frame_b = AnimationFrame {
            step: 1,
            timestamp_ms: 100.0,
            positions: vec![0.0f32; 3],
            rotations: a.iter().map(|v| -v).collect(),
            scales: vec![0.0f32; 3],
            opacities: vec![0.0f32; 1],
            sh_coefficients: Vec::new(),
        };
        let interp = interpolate_frames(&frame_a, &frame_b, 0.5).expect("ok");
        let q = &interp.rotations;
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "got norm {norm}");
        for i in 0..4 {
            assert!(
                (q[i] - a[i]).abs() < 1e-4,
                "component {i}: expected {}, got {}",
                a[i],
                q[i]
            );
        }
    }

    #[test]
    fn test_interpolate_sh_coefficients_length_mismatch_errors() {
        let n = 4usize;
        let mut frame_a = make_frame(0, 0.0, n);
        frame_a.sh_coefficients = vec![0.1f32; n * 12]; // degree 1
        let frame_b = make_frame(1, 100.0, n); // sh_coefficients empty
        let result = interpolate_frames(&frame_a, &frame_b, 0.5);
        assert!(matches!(result, Err(AnimationError::InterpolationMismatch)));
    }

    #[test]
    fn test_interpolate_rejects_internally_inconsistent_frame() {
        // frame_b's n_gaussians() (derived from positions) is 10, but its
        // rotations array is truncated — validate() must catch this rather
        // than silently producing a corrupted interpolated frame.
        let frame_a = make_frame(0, 0.0, 10);
        let mut frame_b = make_frame(1, 100.0, 10);
        frame_b.rotations.truncate(4);
        let result = interpolate_frames(&frame_a, &frame_b, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolate_from_positions_only_frame_errors_instead_of_corrupting() {
        let full = make_frame(0, 0.0, 5);
        let sparse = AnimationFrame::from_positions_only(1, 100.0, vec![0.0f32; 5 * 3]);
        let result = interpolate_frames(&full, &sparse, 0.5);
        assert!(
            result.is_err(),
            "must not silently produce a frame with empty rotations/scales/opacities"
        );
    }

    // --------------- subsample_animation ------------------------------------

    #[test]
    fn test_subsample_stride_1_same() {
        let seq = make_sequence(10, 5, 30.0);
        let sub = subsample_animation(seq, 1).expect("ok");
        assert_eq!(sub.meta.n_frames, 10);
    }

    #[test]
    fn test_subsample_stride_2_half() {
        let seq = make_sequence(10, 5, 30.0);
        let sub = subsample_animation(seq, 2).expect("ok");
        assert_eq!(sub.meta.n_frames, 5);
    }

    #[test]
    fn test_subsample_stride_3() {
        // 9 frames, stride 3 → frames 0, 3, 6 → 3 frames
        let seq = make_sequence(9, 5, 30.0);
        let sub = subsample_animation(seq, 3).expect("ok");
        assert_eq!(sub.meta.n_frames, 3);
    }

    #[test]
    fn test_subsample_stride_zero_error() {
        let seq = make_sequence(5, 5, 30.0);
        let result = subsample_animation(seq, 0);
        assert!(result.is_err());
    }

    // --------------- compute_frame_stats ------------------------------------

    #[test]
    fn test_frame_stats_zero_gaussians() {
        let frame = make_frame(0, 0.0, 0);
        let stats = compute_frame_stats(&frame);
        assert_eq!(stats.n_gaussians, 0);
        assert_eq!(stats.mean_opacity, 0.0);
        assert_eq!(stats.mean_scale, 0.0);
        assert_eq!(stats.position_centroid, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_frame_stats_known_centroid() {
        let n = 2usize;
        let frame = AnimationFrame {
            step: 0,
            timestamp_ms: 0.0,
            positions: vec![
                1.0, 2.0, 3.0, // gaussian 0
                3.0, 4.0, 5.0, // gaussian 1
            ],
            rotations: vec![0.0f32; n * 4],
            scales: vec![0.0f32; n * 3],
            opacities: vec![0.5f32; n],
            sh_coefficients: Vec::new(),
        };
        let stats = compute_frame_stats(&frame);
        assert!((stats.position_centroid[0] - 2.0).abs() < 1e-6);
        assert!((stats.position_centroid[1] - 3.0).abs() < 1e-6);
        assert!((stats.position_centroid[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_frame_stats_mean_opacity() {
        let n = 4usize;
        let frame = AnimationFrame {
            step: 0,
            timestamp_ms: 0.0,
            positions: vec![0.0f32; n * 3],
            rotations: vec![0.0f32; n * 4],
            scales: vec![0.0f32; n * 3],
            opacities: vec![0.0, 0.5, 1.0, 0.5],
            sh_coefficients: Vec::new(),
        };
        let stats = compute_frame_stats(&frame);
        assert!((stats.mean_opacity - 0.5).abs() < 1e-6);
    }

    // --------------- compute_animation_stats --------------------------------

    #[test]
    fn test_animation_stats_single_frame() {
        let seq = make_sequence(1, 10, 30.0);
        let stats = compute_animation_stats(&seq);
        assert_eq!(stats.n_frames, 1);
        assert_eq!(stats.position_drift, 0.0);
    }

    #[test]
    fn test_animation_stats_multiple_frames() {
        let seq = make_sequence(5, 20, 30.0);
        let stats = compute_animation_stats(&seq);
        assert_eq!(stats.n_frames, 5);
        assert!(stats.duration_ms > 0.0);
    }

    // --------------- export / load round-trip --------------------------------

    #[test]
    fn test_json_round_trip() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxigaf_animation_test_round_trip.json");

        let seq = make_sequence(5, 10, 24.0);
        export_animation_json(&seq, &tmp, &AnimExportConfig::default()).expect("export ok");
        let loaded = load_animation_json(&tmp).expect("load ok");

        assert_eq!(loaded.meta.n_frames, seq.meta.n_frames);
        assert_eq!(loaded.meta.n_gaussians, seq.meta.n_gaussians);
        assert_eq!(loaded.frames.len(), seq.frames.len());
        assert!((loaded.meta.fps - seq.meta.fps).abs() < 1e-6);

        // Clean up
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_json_round_trip_with_sh() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxigaf_animation_test_sh.json");

        let n = 3usize;
        let mut frame = make_frame(0, 0.0, n);
        frame.sh_coefficients = vec![0.1f32; n * 12]; // degree 1
        let seq = AnimationSequence::new(vec![frame], 30.0).expect("ok");
        export_animation_json(&seq, &tmp, &AnimExportConfig::default()).expect("export ok");
        let loaded = load_animation_json(&tmp).expect("load ok");
        assert!(loaded.meta.has_sh);
        assert_eq!(loaded.meta.sh_degree, 1);
        assert_eq!(loaded.frames[0].sh_coefficients.len(), n * 12);

        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_export_json_excludes_sh_when_include_sh_false() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxigaf_animation_test_no_sh_export.json");

        let n = 3usize;
        let mut frame = make_frame(0, 0.0, n);
        frame.sh_coefficients = vec![0.5f32; n * 12];
        let seq = AnimationSequence::new(vec![frame], 30.0).expect("ok");

        let config = AnimExportConfig {
            include_sh: false,
            ..Default::default()
        };
        export_animation_json(&seq, &tmp, &config).expect("export ok");
        let loaded = load_animation_json(&tmp).expect("load ok");

        assert!(loaded.frames[0].sh_coefficients.is_empty());
        assert!(!loaded.meta.has_sh);
        assert_eq!(loaded.meta.sh_degree, 0);

        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_export_json_applies_precision_rounding() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxigaf_animation_test_precision_export.json");

        let mut frame = make_frame(0, 0.0, 1);
        frame.positions = vec![1.234_567_9, 0.0, 0.0];
        let seq = AnimationSequence::new(vec![frame], 30.0).expect("ok");

        let config = AnimExportConfig {
            precision: 2,
            ..Default::default()
        };
        export_animation_json(&seq, &tmp, &config).expect("export ok");
        let loaded = load_animation_json(&tmp).expect("load ok");

        assert!(
            (loaded.frames[0].positions[0] - 1.23).abs() < 1e-6,
            "got {}",
            loaded.frames[0].positions[0]
        );

        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_export_json_uses_config_fps_for_meta() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxigaf_animation_test_config_fps_export.json");

        let seq = make_sequence(5, 10, 30.0);
        let config = AnimExportConfig {
            fps: 60.0,
            ..Default::default()
        };
        export_animation_json(&seq, &tmp, &config).expect("export ok");
        let loaded = load_animation_json(&tmp).expect("load ok");

        assert!((loaded.meta.fps - 60.0).abs() < 1e-6);

        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_round_to_precision_zero_places() {
        assert!((round_to_precision(1.7, 0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_round_to_precision_passes_through_non_finite() {
        assert!(round_to_precision(f32::NAN, 4).is_nan());
        assert_eq!(round_to_precision(f32::INFINITY, 4), f32::INFINITY);
    }

    #[test]
    fn test_round_to_precision_large_precision_does_not_panic_or_overflow() {
        // Must not panic, and must never turn a finite input into inf/NaN.
        let r = round_to_precision(1.5, usize::MAX);
        assert!(r.is_finite());
    }

    // --------------- format_animation_summary --------------------------------

    #[test]
    fn test_format_summary_non_empty() {
        let seq = make_sequence(60, 100, 30.0);
        let s = format_animation_summary(&seq);
        assert!(!s.is_empty());
        assert!(s.contains("60 frames"));
        assert!(s.contains("100 Gaussians"));
    }

    #[test]
    fn test_format_summary_contains_fps() {
        let seq = make_sequence(10, 50, 24.0);
        let s = format_animation_summary(&seq);
        assert!(s.contains("24"));
    }

    // --------------- loop_animation ------------------------------------------

    #[test]
    fn test_loop_animation_two_repeats() {
        let seq = make_sequence(5, 10, 30.0);
        let looped = loop_animation(seq, 2).expect("ok");
        assert_eq!(looped.meta.n_frames, 10);
    }

    #[test]
    fn test_loop_animation_three_repeats() {
        let seq = make_sequence(4, 5, 10.0);
        let looped = loop_animation(seq, 3).expect("ok");
        assert_eq!(looped.meta.n_frames, 12);
    }

    #[test]
    fn test_loop_animation_one_repeat() {
        let seq = make_sequence(6, 10, 30.0);
        let looped = loop_animation(seq, 1).expect("ok");
        assert_eq!(looped.meta.n_frames, 6);
    }

    #[test]
    fn test_loop_animation_zero_repeats_error() {
        let seq = make_sequence(5, 10, 30.0);
        let result = loop_animation(seq, 0);
        assert!(matches!(result, Err(AnimationError::EmptyAnimation)));
    }

    #[test]
    fn test_loop_timestamps_offset() {
        let seq = make_sequence(3, 5, 10.0);
        let dur = seq.duration_ms();
        let looped = loop_animation(seq, 2).expect("ok");
        // Frame at index 3 (start of second repeat) should have timestamp = dur
        assert!((looped.frames[3].timestamp_ms - dur).abs() < 1e-3);
    }

    // --------------- AnimationError variants --------------------------------

    #[test]
    fn test_error_empty_animation_display() {
        let e = AnimationError::EmptyAnimation;
        let s = e.to_string();
        assert!(s.contains("Empty"));
    }

    #[test]
    fn test_error_frame_count_mismatch_display() {
        let e = AnimationError::FrameCountMismatch {
            expected: 10,
            got: 5,
        };
        let s = e.to_string();
        assert!(s.contains("10") && s.contains("5"));
    }

    #[test]
    fn test_error_gaussian_count_mismatch_display() {
        let e = AnimationError::GaussianCountMismatch {
            n0: 100,
            idx: 3,
            ni: 50,
        };
        let s = e.to_string();
        assert!(s.contains("100") && s.contains("3") && s.contains("50"));
    }

    #[test]
    fn test_error_invalid_fps_display() {
        let e = AnimationError::InvalidFps { fps: -5.0 };
        let s = e.to_string();
        assert!(s.contains("-5"));
    }

    #[test]
    fn test_error_invalid_frame_range_display() {
        let e = AnimationError::InvalidFrameRange {
            start: 5,
            end: 3,
            len: 10,
        };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("3") && s.contains("10"));
    }

    #[test]
    fn test_error_interpolation_mismatch_display() {
        let e = AnimationError::InterpolationMismatch;
        let s = e.to_string();
        assert!(s.contains("nterpolat"));
    }
}
