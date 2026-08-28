//! Expression animation system for the FLAME model.
//!
//! Provides keyframe-based animation of facial expression blend weights with:
//! - Six easing functions (Linear, `EaseIn`, `EaseOut`, `EaseInOut`, Bounce, `ElasticOut`)
//! - Catmull-Rom spline interpolation between keyframes
//! - Three loop modes (Once, Loop, `PingPong`)
//! - Animation clips with variable playback rate
//! - Animation player for real-time playback control
//!
//! # Quick Start
//!
//! ```rust
//! use oxigaf_flame::expression_animation::{
//!     AnimationClip, AnimationPlayer, EasingFunction, ExpressionKeyframe,
//!     ExpressionTimeline, LoopMode,
//! };
//!
//! let names = vec!["smile".to_string(), "brow_raise".to_string()];
//! let mut timeline = ExpressionTimeline::new(names.clone());
//!
//! timeline.add_keyframe(
//!     ExpressionKeyframe::new(0.0, vec![0.0, 0.0])
//! ).expect("add keyframe 0");
//! timeline.add_keyframe(
//!     ExpressionKeyframe::new(1.0, vec![1.0, 0.5])
//!         .with_easing(EasingFunction::EaseInOut)
//! ).expect("add keyframe 1");
//!
//! let clip = AnimationClip::new("smile_anim", names)
//!     .with_loop_mode(LoopMode::Loop);
//! let mut player = AnimationPlayer::new(clip);
//! player.play();
//! let weights = player.advance(0.5);
//! ```

use std::f32::consts::PI;
use std::fmt;

// ---------------------------------------------------------------------------
// AnimationError
// ---------------------------------------------------------------------------

/// Errors that can occur during expression animation operations.
#[derive(Debug)]
pub enum AnimationError {
    /// The supplied weights vector length does not match the timeline's dimension count.
    DimensionMismatch { expected: usize, got: usize },
    /// The keyframe index is out of range.
    InvalidKeyframeIndex(usize),
    /// The timeline contains no keyframes.
    EmptyTimeline,
    /// A keyframe time value is negative.
    NegativeTime(f32),
}

impl fmt::Display for AnimationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} weights, got {got}"
            ),
            Self::InvalidKeyframeIndex(idx) => {
                write!(f, "keyframe index {idx} is out of range")
            }
            Self::EmptyTimeline => write!(f, "timeline contains no keyframes"),
            Self::NegativeTime(t) => write!(f, "keyframe time {t} is negative"),
        }
    }
}

impl std::error::Error for AnimationError {}

// ---------------------------------------------------------------------------
// EasingFunction
// ---------------------------------------------------------------------------

/// The easing curve applied between two consecutive keyframes.
///
/// Each variant maps a normalized time `t ∈ [0, 1]` to a progress value also
/// in `[0, 1]`, altering the rate of change of the interpolated quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EasingFunction {
    /// Constant-rate interpolation (progress = t).
    Linear,
    /// Cubic ease-in: slow start, fast end (`t³`).
    EaseIn,
    /// Cubic ease-out: fast start, slow end (`1 - (1-t)³`).
    EaseOut,
    /// Cubic ease-in-out: smooth step (`3t² - 2t³`).
    EaseInOut,
    /// Classic four-piece piecewise bounce that settles at the end.
    Bounce,
    /// Spring overshoot with exponential decay.
    ElasticOut,
}

impl EasingFunction {
    /// Apply the easing curve to `t ∈ [0, 1]`, returning a value typically in
    /// `[0, 1]` (`ElasticOut` may momentarily exceed this range before settling).
    #[must_use]
    pub fn apply(&self, t: f32) -> f32 {
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t * t,
            Self::EaseOut => {
                let inv = 1.0 - t;
                1.0 - inv * inv * inv
            }
            Self::EaseInOut => t * t * (3.0 - 2.0 * t),
            Self::Bounce => {
                if t <= 0.0 {
                    return 0.0;
                }
                if t >= 1.0 {
                    return 1.0;
                }
                Self::bounce_out(t)
            }
            Self::ElasticOut => {
                if t <= 0.0 {
                    return 0.0;
                }
                if t >= 1.0 {
                    return 1.0;
                }
                // Standard easeOutElastic: 2^(-10t) * sin((10t - 0.75) * (2π/3)) + 1.
                1.0 + (-10.0 * t).exp2() * ((10.0 * t - 0.75) * (2.0 * PI / 3.0)).sin()
            }
        }
    }

    /// Classic four-segment bounce-out easing.
    ///
    /// The constants `n1 = 7.5625` and `d1 = 2.75` produce four sub-bounces.
    fn bounce_out(t: f32) -> f32 {
        const N1: f32 = 7.5625;
        const D1: f32 = 2.75;

        if t < 1.0 / D1 {
            N1 * t * t
        } else if t < 2.0 / D1 {
            let t2 = t - 1.5 / D1;
            N1 * t2 * t2 + 0.75
        } else if t < 2.5 / D1 {
            let t2 = t - 2.25 / D1;
            N1 * t2 * t2 + 0.9375
        } else {
            let t2 = t - 2.625 / D1;
            N1 * t2 * t2 + 0.984_375
        }
    }
}

// ---------------------------------------------------------------------------
// ExpressionKeyframe
// ---------------------------------------------------------------------------

/// A single keyframe in an expression timeline.
///
/// Each keyframe stores a time stamp, blend weights for every expression
/// dimension, and an easing function that controls the interpolation from
/// this keyframe *to* the next.
#[derive(Debug, Clone)]
pub struct ExpressionKeyframe {
    /// Time in seconds (must be ≥ 0).
    pub time: f32,
    /// Blend weights — one per expression dimension.
    pub weights: Vec<f32>,
    /// Easing to apply from this keyframe to the next.
    pub easing: EasingFunction,
}

impl ExpressionKeyframe {
    /// Create a new keyframe at `time` with `weights` and `Linear` easing.
    #[must_use]
    pub fn new(time: f32, weights: Vec<f32>) -> Self {
        Self {
            time,
            weights,
            easing: EasingFunction::Linear,
        }
    }

    /// Builder helper: replace the easing function and return `self`.
    #[must_use]
    pub fn with_easing(mut self, easing: EasingFunction) -> Self {
        self.easing = easing;
        self
    }

    /// Convenience constructor: all weights zero at `t = 0`.
    #[must_use]
    pub fn at_zero(n: usize) -> Self {
        Self::new(0.0, vec![0.0_f32; n])
    }
}

// ---------------------------------------------------------------------------
// ExpressionTimeline
// ---------------------------------------------------------------------------

/// A sorted sequence of expression keyframes for one animation track.
///
/// The timeline always maintains keyframes sorted by ascending time.
/// Catmull-Rom spline interpolation is used between interior keyframes;
/// linear interpolation is used at the boundary segments where one of the
/// phantom control points would be out of range.
#[derive(Debug, Clone)]
pub struct ExpressionTimeline {
    /// Human-readable name for each expression dimension.
    pub expression_names: Vec<String>,
    /// Keyframes sorted by ascending `time` (maintained invariant).
    keyframes: Vec<ExpressionKeyframe>,
}

impl ExpressionTimeline {
    /// Create a new timeline with a single zero-weight keyframe at `t = 0`.
    #[must_use]
    pub fn new(expression_names: Vec<String>) -> Self {
        let n = expression_names.len();
        Self {
            keyframes: vec![ExpressionKeyframe::at_zero(n)],
            expression_names,
        }
    }

    /// Return the number of expression dimensions.
    #[must_use]
    pub fn num_dimensions(&self) -> usize {
        self.expression_names.len()
    }

    /// Return the number of keyframes.
    #[must_use]
    pub fn num_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    /// Return the time of the last keyframe, or `0.0` if the timeline is empty.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map_or(0.0, |kf| kf.time)
    }

    /// Insert a keyframe in sorted order by time.
    ///
    /// # Errors
    ///
    /// - [`AnimationError::DimensionMismatch`] if `keyframe.weights.len() !=
    ///   self.num_dimensions()`
    /// - [`AnimationError::NegativeTime`] if `keyframe.time < 0.0`
    pub fn add_keyframe(&mut self, keyframe: ExpressionKeyframe) -> Result<(), AnimationError> {
        if keyframe.time < 0.0 {
            return Err(AnimationError::NegativeTime(keyframe.time));
        }
        let expected = self.num_dimensions();
        let got = keyframe.weights.len();
        if got != expected {
            return Err(AnimationError::DimensionMismatch { expected, got });
        }
        // Binary-search for the insertion position to maintain sorted order.
        let pos = self
            .keyframes
            .partition_point(|kf| kf.time <= keyframe.time);
        self.keyframes.insert(pos, keyframe);
        Ok(())
    }

    /// Remove the keyframe at position `idx`.
    ///
    /// # Errors
    ///
    /// - [`AnimationError::InvalidKeyframeIndex`] if `idx >= self.num_keyframes()`
    pub fn remove_keyframe(&mut self, idx: usize) -> Result<(), AnimationError> {
        if idx >= self.keyframes.len() {
            return Err(AnimationError::InvalidKeyframeIndex(idx));
        }
        self.keyframes.remove(idx);
        Ok(())
    }

    /// Evaluate the timeline at time `t`, returning interpolated blend weights.
    ///
    /// - If `t` is before the first keyframe, returns the first keyframe's weights.
    /// - If `t` is after the last keyframe, returns the last keyframe's weights.
    /// - Otherwise, the segment that brackets `t` is interpolated using the
    ///   easing function of its left keyframe: Catmull-Rom spline
    ///   interpolation for an interior segment (using a uniform
    ///   parameterisation — very unevenly spaced keyframes can visibly
    ///   over/undershoot), or plain linear interpolation for a boundary
    ///   segment (the first or last segment, where one of the two phantom
    ///   Catmull-Rom control points would be out of range).
    #[must_use]
    pub fn evaluate(&self, t: f32) -> Vec<f32> {
        let n = self.num_dimensions();
        if self.keyframes.is_empty() {
            return vec![0.0_f32; n];
        }

        // Clamp before first keyframe.
        if t <= self.keyframes[0].time {
            return self.keyframes[0].weights.clone();
        }

        // Clamp after last keyframe.
        let last_idx = self.keyframes.len() - 1;
        if t >= self.keyframes[last_idx].time {
            return self.keyframes[last_idx].weights.clone();
        }

        // Find the segment [i, i+1] that contains t.
        // We know t is strictly between first and last, so at least two keyframes exist.
        let i = self
            .keyframes
            .partition_point(|kf| kf.time <= t)
            .saturating_sub(1);
        // Guard: ensure i+1 is valid.
        let i = i.min(self.keyframes.len() - 2);

        let kf_left = &self.keyframes[i];
        let kf_right = &self.keyframes[i + 1];

        // Compute local t in [0, 1] for this segment.
        let dt = kf_right.time - kf_left.time;
        let local_t = if dt > f32::EPSILON {
            ((t - kf_left.time) / dt).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Apply easing.
        let te = kf_left.easing.apply(local_t);

        let p1 = &kf_left.weights;
        let p2 = &kf_right.weights;

        // At a boundary segment (i == 0, or i+2 >= keyframes.len()) one of
        // the two phantom Catmull-Rom control points would be out of
        // range. Rather than duplicating the nearest endpoint — which
        // silently distorts the curve away from a straight line (matching
        // it only at te = 0, 0.5 and 1) — fall back to true linear
        // interpolation there, per this type's documented contract.
        if i == 0 || i + 2 >= self.keyframes.len() {
            let mut result = vec![0.0_f32; n];
            for (dim, out_val) in result.iter_mut().enumerate().take(n) {
                let q1 = p1.get(dim).copied().unwrap_or(0.0);
                let q2 = p2.get(dim).copied().unwrap_or(0.0);
                *out_val = q1 + (q2 - q1) * te;
            }
            return result;
        }

        // Catmull-Rom control points for an interior segment.
        // p0: the keyframe before i.
        // p1, p2: the two keyframes bracketing `t` (kf_left, kf_right).
        // p3: the keyframe after i+1.
        let p0 = &self.keyframes[i - 1].weights;
        let p3 = &self.keyframes[i + 2].weights;

        // Catmull-Rom per-dimension. Note this uses a uniform
        // parameterisation: with non-uniformly spaced keyframes the
        // tangents implied here are not scaled by the neighboring
        // segments' actual durations, which can visibly over/undershoot
        // for very unevenly spaced keyframes. A centripetal or chordal
        // parameterisation would address this but is not implemented here.
        let mut result = vec![0.0_f32; n];
        let t2 = te * te;
        let t3 = t2 * te;
        for (dim, out_val) in result.iter_mut().enumerate().take(n) {
            let q0 = p0.get(dim).copied().unwrap_or(0.0);
            let q1 = p1.get(dim).copied().unwrap_or(0.0);
            let q2 = p2.get(dim).copied().unwrap_or(0.0);
            let q3 = p3.get(dim).copied().unwrap_or(0.0);

            // q(t) = 0.5 * ( 2*q1 + (-q0+q2)*t + (2*q0-5*q1+4*q2-q3)*t² + (-q0+3*q1-3*q2+q3)*t³ )
            *out_val = 0.5
                * (2.0 * q1
                    + (-q0 + q2) * te
                    + (2.0 * q0 - 5.0 * q1 + 4.0 * q2 - q3) * t2
                    + (-q0 + 3.0 * q1 - 3.0 * q2 + q3) * t3);
        }
        result
    }

    /// Sample the timeline at uniform intervals.
    ///
    /// Returns a `Vec` of `(time, weights)` pairs spanning `[0, total_duration]`
    /// at `fps` samples per second. The first sample is always at `t = 0`.
    ///
    /// Returns an empty `Vec` if `fps <= 0.0` or `total_duration <= 0.0`.
    #[must_use]
    pub fn sample_uniform(&self, fps: f32, total_duration: f32) -> Vec<(f32, Vec<f32>)> {
        if fps <= 0.0 || total_duration <= 0.0 {
            return Vec::new();
        }
        let step = 1.0 / fps;
        // Include a sample for every frame plus possibly an extra at the end.
        let num_samples = (total_duration * fps).ceil() as usize + 1;
        let mut out = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = (i as f32 * step).min(total_duration);
            out.push((t, self.evaluate(t)));
            if t >= total_duration {
                break;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// LoopMode
// ---------------------------------------------------------------------------

/// How an [`AnimationClip`] behaves when the playhead reaches the end.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoopMode {
    /// Play once and stop at the end.
    Once,
    /// Wrap the time modulo the clip duration and play continuously.
    Loop,
    /// Even-numbered passes go forward; odd-numbered passes are mirrored.
    PingPong,
}

// ---------------------------------------------------------------------------
// AnimationClip
// ---------------------------------------------------------------------------

/// A named animation clip containing a timeline plus playback configuration.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    /// Human-readable name of this clip.
    pub name: String,
    /// The underlying keyframe timeline.
    pub timeline: ExpressionTimeline,
    /// How the clip loops (or doesn't) at the end.
    pub loop_mode: LoopMode,
    /// Playback rate multiplier (`1.0` = real-time, `2.0` = double speed).
    pub playback_rate: f32,
}

impl AnimationClip {
    /// Create a new clip with `LoopMode::Once` and `playback_rate = 1.0`.
    ///
    /// The embedded timeline is initialised with a single zero-weight keyframe
    /// at `t = 0`.
    #[must_use]
    pub fn new(name: impl Into<String>, expression_names: Vec<String>) -> Self {
        Self {
            name: name.into(),
            timeline: ExpressionTimeline::new(expression_names),
            loop_mode: LoopMode::Once,
            playback_rate: 1.0,
        }
    }

    /// Builder helper: set the loop mode and return `self`.
    #[must_use]
    pub fn with_loop_mode(mut self, mode: LoopMode) -> Self {
        self.loop_mode = mode;
        self
    }

    /// Builder helper: set the playback rate and return `self`.
    #[must_use]
    pub fn with_playback_rate(mut self, rate: f32) -> Self {
        self.playback_rate = rate;
        self
    }

    /// Effective duration in seconds (timeline duration divided by playback rate).
    ///
    /// Returns `f32::INFINITY` if `playback_rate` is zero to avoid division by
    /// zero.
    #[must_use]
    pub fn effective_duration(&self) -> f32 {
        if self.playback_rate.abs() < f32::EPSILON {
            return f32::INFINITY;
        }
        self.timeline.duration() / self.playback_rate.abs()
    }

    /// Evaluate the clip at global time `t`.
    ///
    /// Applies `playback_rate` and `loop_mode` before delegating to the
    /// timeline's [`ExpressionTimeline::evaluate`].
    #[must_use]
    pub fn evaluate(&self, t: f32) -> Vec<f32> {
        let duration = self.timeline.duration();

        // Map global time → local timeline time via playback_rate.
        let scaled_t = t * self.playback_rate.abs();

        if duration <= f32::EPSILON {
            // Zero-length clip: always return the first keyframe (or zeros).
            return self.timeline.evaluate(0.0);
        }

        let local_t = match self.loop_mode {
            LoopMode::Once => scaled_t.clamp(0.0, duration),
            LoopMode::Loop => {
                let wrapped = scaled_t % duration;
                if wrapped < 0.0 {
                    wrapped + duration
                } else {
                    wrapped
                }
            }
            LoopMode::PingPong => {
                // Determine which pass number we are in.
                let pass = (scaled_t / duration) as u32;
                let frac = (scaled_t % duration).clamp(0.0, duration);
                if pass.is_multiple_of(2) {
                    frac
                } else {
                    duration - frac
                }
            }
        };

        self.timeline.evaluate(local_t)
    }
}

// ---------------------------------------------------------------------------
// AnimationPlayer
// ---------------------------------------------------------------------------

/// A stateful playback controller wrapping an [`AnimationClip`].
///
/// Advance the internal clock with [`AnimationPlayer::advance`] each frame to
/// receive the current interpolated expression weights.
#[derive(Debug, Clone)]
pub struct AnimationPlayer {
    /// The clip being played.
    pub clip: AnimationClip,
    current_time: f32,
    /// Whether the player is currently advancing the playhead.
    pub is_playing: bool,
}

impl AnimationPlayer {
    /// Create a new player for `clip`, starting paused at `t = 0`.
    #[must_use]
    pub fn new(clip: AnimationClip) -> Self {
        Self {
            clip,
            current_time: 0.0,
            is_playing: false,
        }
    }

    /// Begin or resume playback.
    pub fn play(&mut self) {
        self.is_playing = true;
    }

    /// Pause playback without resetting the playhead.
    pub fn pause(&mut self) {
        self.is_playing = false;
    }

    /// Reset the playhead to `t = 0` (does not change the playing state).
    pub fn reset(&mut self) {
        self.current_time = 0.0;
    }

    /// Seek to an absolute time `t` (clamped to `≥ 0`).
    pub fn seek(&mut self, t: f32) {
        self.current_time = t.max(0.0);
    }

    /// Return the current playhead time in seconds.
    #[must_use]
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Advance the playhead by `delta_secs` if playing, then return the
    /// current interpolated expression weights.
    pub fn advance(&mut self, delta_secs: f32) -> Vec<f32> {
        if self.is_playing {
            self.current_time += delta_secs;
        }
        self.clip.evaluate(self.current_time)
    }

    /// Returns `true` when a `Once`-mode clip has reached or passed its end.
    ///
    /// Always returns `false` for `Loop` and `PingPong` clips (they never
    /// finish).
    #[must_use]
    pub fn is_finished(&self) -> bool {
        matches!(self.clip.loop_mode, LoopMode::Once)
            && self.current_time >= self.clip.effective_duration()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // EasingFunction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_easing_linear_identity() {
        let e = EasingFunction::Linear;
        assert!((e.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((e.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((e.apply(1.0) - 1.0).abs() < 1e-6);
        assert!((e.apply(0.25) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_in_midpoint() {
        // t=0.5 → 0.5^3 = 0.125
        let e = EasingFunction::EaseIn;
        assert!((e.apply(0.5) - 0.125).abs() < 1e-4);
        assert!((e.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((e.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_out_midpoint() {
        // t=0.5 → 1 - (1-0.5)^3 = 1 - 0.125 = 0.875
        let e = EasingFunction::EaseOut;
        assert!((e.apply(0.5) - 0.875).abs() < 1e-4);
        assert!((e.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((e.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_in_out_midpoint() {
        // t=0.5 → 3*(0.5)^2 - 2*(0.5)^3 = 0.75 - 0.25 = 0.5
        let e = EasingFunction::EaseInOut;
        assert!((e.apply(0.5) - 0.5).abs() < 1e-4);
        assert!((e.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((e.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_easing_bounce_endpoints() {
        let e = EasingFunction::Bounce;
        assert!((e.apply(0.0) - 0.0).abs() < 1e-4, "bounce at t=0 must be 0");
        assert!((e.apply(1.0) - 1.0).abs() < 1e-4, "bounce at t=1 must be 1");
    }

    #[test]
    fn test_easing_bounce_monotone_at_end() {
        // Final segment: t ≥ 2.5/2.75 ≈ 0.909 → converges to 1.0
        let e = EasingFunction::Bounce;
        let v = e.apply(0.95);
        assert!(v > 0.98 && v <= 1.001, "bounce near end: got {v}");
    }

    #[test]
    fn test_easing_elastic_out_endpoints() {
        let e = EasingFunction::ElasticOut;
        assert!(
            (e.apply(0.0) - 0.0).abs() < 1e-6,
            "elastic at t=0 must be 0, got {}",
            e.apply(0.0)
        );
        assert!(
            (e.apply(1.0) - 1.0).abs() < 1e-6,
            "elastic at t=1 must be 1, got {}",
            e.apply(1.0)
        );
    }

    #[test]
    fn test_easing_elastic_out_mid_range() {
        // At t in (0,1) the formula should produce values that eventually
        // converge to 1; at t=0.5 a reasonable mid-point value is above 1.0
        // (spring overshoot) — just verify it's finite and positive.
        let e = EasingFunction::ElasticOut;
        let v = e.apply(0.5);
        assert!(v.is_finite(), "elastic mid must be finite");
        assert!(v >= 0.0, "elastic mid must be non-negative");
    }

    #[test]
    fn test_easing_elastic_out_eases_in_from_zero() {
        // Regression test: the previous (buggy) formula evaluated to
        // 1 - e^0 * cos(-pi/2) = 1.0 immediately after t=0 (e.g. ~0.979 at
        // t=0.001), popping instantly to the target instead of easing in.
        // The correct curve stays close to 0 just after t=0.
        let e = EasingFunction::ElasticOut;
        let v = e.apply(0.001);
        assert!(
            v < 0.05,
            "elastic-out must ease in from 0 near t=0, got apply(0.001)={v}"
        );
    }

    #[test]
    fn test_easing_elastic_out_overshoots_past_one() {
        // A genuine elastic/spring easing must overshoot its target at some
        // point in (0, 1) before settling back to 1.0.
        let e = EasingFunction::ElasticOut;
        let max = (1..1000)
            .map(|i| e.apply(i as f32 / 1000.0))
            .fold(f32::MIN, f32::max);
        assert!(
            max > 1.05,
            "elastic-out must overshoot above 1.0 somewhere in (0,1), max={max}"
        );
    }

    // -----------------------------------------------------------------------
    // ExpressionKeyframe tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_keyframe_new_and_builder() {
        let kf = ExpressionKeyframe::new(1.0, vec![0.5, 0.3]).with_easing(EasingFunction::EaseIn);
        assert!((kf.time - 1.0).abs() < f32::EPSILON);
        assert_eq!(kf.weights.len(), 2);
        assert!((kf.weights[0] - 0.5).abs() < f32::EPSILON);
        assert_eq!(kf.easing, EasingFunction::EaseIn);
    }

    #[test]
    fn test_keyframe_at_zero() {
        let kf = ExpressionKeyframe::at_zero(4);
        assert!((kf.time - 0.0).abs() < f32::EPSILON);
        assert_eq!(kf.weights.len(), 4);
        assert!(kf.weights.iter().all(|&w| w == 0.0));
        assert_eq!(kf.easing, EasingFunction::Linear);
    }

    // -----------------------------------------------------------------------
    // ExpressionTimeline tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_timeline_add_keyframe_sorted() {
        let names = vec!["a".to_string(), "b".to_string()];
        let mut tl = ExpressionTimeline::new(names);

        // Add keyframes out of order.
        tl.add_keyframe(ExpressionKeyframe::new(2.0, vec![1.0, 1.0]))
            .expect("add 2.0");
        tl.add_keyframe(ExpressionKeyframe::new(1.0, vec![0.5, 0.5]))
            .expect("add 1.0");

        // After insertion the keyframes should be: 0.0, 1.0, 2.0
        assert_eq!(tl.num_keyframes(), 3);
        assert!((tl.keyframes[0].time - 0.0).abs() < f32::EPSILON);
        assert!((tl.keyframes[1].time - 1.0).abs() < f32::EPSILON);
        assert!((tl.keyframes[2].time - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_timeline_evaluate_at_endpoints() {
        let names = vec!["x".to_string()];
        let mut tl = ExpressionTimeline::new(names);
        tl.add_keyframe(ExpressionKeyframe::new(1.0, vec![1.0]))
            .expect("add 1.0");

        // t=0: should return first keyframe weights.
        let at_start = tl.evaluate(0.0);
        assert!(
            (at_start[0] - 0.0).abs() < 1e-4,
            "start: got {}",
            at_start[0]
        );

        // t=1: should return last keyframe weights.
        let at_end = tl.evaluate(1.0);
        assert!((at_end[0] - 1.0).abs() < 1e-4, "end: got {}", at_end[0]);
    }

    #[test]
    fn test_timeline_evaluate_linear_midpoint() {
        // With only two keyframes the single segment [0, 1] is a boundary
        // segment on both sides, which is documented (and, since the fix
        // for the Catmull-Rom-at-boundaries bug, actually implemented) to
        // use plain linear interpolation. Timeline: t=0 → [0.0], t=1 →
        // [2.0]. Evaluate at t=0.5 with Linear easing → local_t=0.5.
        let names = vec!["x".to_string()];
        let mut tl = ExpressionTimeline::new(names);
        tl.add_keyframe(ExpressionKeyframe::new(1.0, vec![2.0]))
            .expect("add 1.0");

        let mid = tl.evaluate(0.5);
        // Linear: 0.0 + (2.0 - 0.0) * 0.5 = 1.0.
        assert!(
            (mid[0] - 1.0).abs() < 1e-4,
            "midpoint with linear easing: got {}",
            mid[0]
        );
    }

    #[test]
    fn test_timeline_evaluate_boundary_segment_is_truly_linear() {
        // Regression test: unlike `test_timeline_evaluate_linear_midpoint`'s
        // t=0.5 (where the old endpoint-duplicated-Catmull-Rom formula
        // happens to coincide with the line, by coincidence of the specific
        // polynomial), t=0.25 exercises a point where the two formulas
        // differ, so this would have caught the doc/behavior mismatch.
        let names = vec!["x".to_string()];
        let mut tl = ExpressionTimeline::new(names);
        tl.add_keyframe(ExpressionKeyframe::new(1.0, vec![2.0]))
            .expect("add 1.0");

        let q = tl.evaluate(0.25);
        // True linear: 0.0 + (2.0 - 0.0) * 0.25 = 0.5.
        assert!(
            (q[0] - 0.5).abs() < 1e-4,
            "boundary segment at t=0.25 must be linear, got {}",
            q[0]
        );
    }

    #[test]
    fn test_timeline_evaluate_interior_segment_uses_catmull_rom() {
        // With >= 4 keyframes, the middle segment (between keyframes 1 and
        // 2) is an interior segment and should still use full Catmull-Rom
        // with real p0/p3 control points (not linear).
        let names = vec!["x".to_string()];
        let mut tl = ExpressionTimeline::new(names); // kf0: t=0, w=0
        tl.add_keyframe(ExpressionKeyframe::new(1.0, vec![1.0]))
            .expect("add kf1");
        tl.add_keyframe(ExpressionKeyframe::new(2.0, vec![1.0]))
            .expect("add kf2");
        tl.add_keyframe(ExpressionKeyframe::new(3.0, vec![0.0]))
            .expect("add kf3");

        // Segment [kf1, kf2] (both weight 1.0) is interior: i=1, i+2=3 < 4.
        // p0 = kf0.weights = [0.0], p1 = p2 = [1.0], p3 = kf3.weights = [0.0].
        // At te=0.5: q = 0.5*(2*1 + (-0+1)*0.5 + (0-5+4-0)*0.25 + (0+3-3+0)*0.125)
        //          = 0.5*(2 + 0.5 - 0.25 + 0) = 0.5*2.25 = 1.125 (overshoot,
        // characteristic of Catmull-Rom, unlike linear which would stay at 1.0).
        let mid = tl.evaluate(1.5);
        assert!(
            (mid[0] - 1.125).abs() < 1e-3,
            "interior segment must use Catmull-Rom (with overshoot), got {}",
            mid[0]
        );
    }

    #[test]
    fn test_timeline_evaluate_clamps_before_start() {
        let names = vec!["y".to_string()];
        let mut tl = ExpressionTimeline::new(names);
        tl.add_keyframe(ExpressionKeyframe::new(1.0, vec![5.0]))
            .expect("add");

        // Negative time → clamp to first keyframe (t=0, weight=0.0).
        let before = tl.evaluate(-1.0);
        assert!((before[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_timeline_evaluate_clamps_after_end() {
        let names = vec!["y".to_string()];
        let mut tl = ExpressionTimeline::new(names);
        tl.add_keyframe(ExpressionKeyframe::new(1.0, vec![5.0]))
            .expect("add");

        // Time beyond last keyframe → clamp to last keyframe weights.
        let after = tl.evaluate(999.0);
        assert!((after[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_timeline_sample_uniform() {
        let names = vec!["z".to_string()];
        let mut tl = ExpressionTimeline::new(names);
        tl.add_keyframe(ExpressionKeyframe::new(1.0, vec![1.0]))
            .expect("add");

        // 2 fps over 1 second → samples at t=0, 0.5, 1.0
        let samples = tl.sample_uniform(2.0, 1.0);
        assert!(
            samples.len() >= 2,
            "expected at least 2 samples, got {}",
            samples.len()
        );
        // First sample at t=0.
        assert!((samples[0].0 - 0.0).abs() < 1e-4);
        // Last sample at t=1.0.
        let last = samples.last().expect("non-empty");
        assert!((last.0 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_timeline_sample_uniform_zero_fps_returns_empty() {
        let names = vec!["z".to_string()];
        let tl = ExpressionTimeline::new(names);
        let samples = tl.sample_uniform(0.0, 1.0);
        assert!(samples.is_empty());
    }

    // -----------------------------------------------------------------------
    // LoopMode / AnimationClip tests
    // -----------------------------------------------------------------------

    fn make_two_keyframe_clip(loop_mode: LoopMode) -> AnimationClip {
        let names = vec!["w".to_string()];
        let mut clip = AnimationClip::new("test", names).with_loop_mode(loop_mode);
        clip.timeline
            .add_keyframe(ExpressionKeyframe::new(1.0, vec![1.0]))
            .expect("add kf");
        clip
    }

    #[test]
    fn test_loop_mode_once() {
        let clip = make_two_keyframe_clip(LoopMode::Once);
        // Past the end: should return last keyframe value.
        let at_end = clip.evaluate(2.0);
        assert!(
            (at_end[0] - 1.0).abs() < 1e-4,
            "once past end: got {}",
            at_end[0]
        );

        // Before start: first keyframe.
        let at_start = clip.evaluate(-1.0);
        // evaluate uses 0.0 as effective local time since scaled_t ≥ 0 → clamp to 0.
        assert!((at_start[0] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_loop_mode_loop() {
        let clip = make_two_keyframe_clip(LoopMode::Loop);
        // At t=0: weight=0, at t=1: weight=1, at t=1.5 wraps to t=0.5 ≈ 0.5 weight.
        let at_half = clip.evaluate(0.5);
        let wrapped = clip.evaluate(1.5); // 1.5 % 1.0 = 0.5
        assert!(
            (at_half[0] - wrapped[0]).abs() < 1e-4,
            "loop wrap: {} vs {}",
            at_half[0],
            wrapped[0]
        );
    }

    #[test]
    fn test_loop_mode_ping_pong() {
        let clip = make_two_keyframe_clip(LoopMode::PingPong);
        // t=0.5 (forward pass): weight should be ≈0.5.
        let forward = clip.evaluate(0.5);
        // t=1.5 (reverse pass): mirrors t=0.5 → weight should also be ≈0.5.
        let reverse = clip.evaluate(1.5); // pass=1 (odd) → mirror
        assert!(
            (forward[0] - reverse[0]).abs() < 1e-4,
            "ping-pong mirror: {} vs {}",
            forward[0],
            reverse[0]
        );
    }

    // -----------------------------------------------------------------------
    // AnimationPlayer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_animation_player_advance() {
        let clip = make_two_keyframe_clip(LoopMode::Once);
        let mut player = AnimationPlayer::new(clip);
        player.play();

        // Advance to t=0.5.
        let weights = player.advance(0.5);
        assert!((player.current_time() - 0.5).abs() < 1e-6);
        // Weight at t=0.5 with linear easing / Catmull-Rom two-keyframe.
        assert!(
            weights[0] > 0.0 && weights[0] < 1.0,
            "mid weight: {}",
            weights[0]
        );
    }

    #[test]
    fn test_animation_player_pause_does_not_advance() {
        let clip = make_two_keyframe_clip(LoopMode::Once);
        let mut player = AnimationPlayer::new(clip);
        player.play();
        player.advance(0.3);
        player.pause();
        player.advance(0.3);
        // Time should still be 0.3 after the paused advance.
        assert!(
            (player.current_time() - 0.3).abs() < 1e-6,
            "paused time: {}",
            player.current_time()
        );
    }

    #[test]
    fn test_animation_player_is_finished() {
        let clip = make_two_keyframe_clip(LoopMode::Once);
        let mut player = AnimationPlayer::new(clip);
        player.play();
        assert!(!player.is_finished(), "should not be finished at start");
        player.advance(2.0); // past the end (duration=1.0)
        assert!(player.is_finished(), "should be finished after end");
    }

    #[test]
    fn test_animation_player_is_not_finished_in_loop_mode() {
        let clip = make_two_keyframe_clip(LoopMode::Loop);
        let mut player = AnimationPlayer::new(clip);
        player.play();
        player.advance(10.0); // well past duration
        assert!(!player.is_finished(), "loop mode should never finish");
    }

    #[test]
    fn test_animation_player_seek_and_reset() {
        let clip = make_two_keyframe_clip(LoopMode::Once);
        let mut player = AnimationPlayer::new(clip);
        player.seek(0.75);
        assert!((player.current_time() - 0.75).abs() < 1e-6);
        player.reset();
        assert!((player.current_time() - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Error handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dimension_mismatch_error() {
        let names = vec!["a".to_string(), "b".to_string()];
        let mut tl = ExpressionTimeline::new(names);
        // Supply 3 weights for a 2-dimensional timeline.
        let result = tl.add_keyframe(ExpressionKeyframe::new(1.0, vec![0.5, 0.5, 0.5]));
        assert!(result.is_err());
        match result {
            Err(AnimationError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, 2);
                assert_eq!(got, 3);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_negative_time_error() {
        let names = vec!["a".to_string()];
        let mut tl = ExpressionTimeline::new(names);
        let result = tl.add_keyframe(ExpressionKeyframe::new(-0.1, vec![0.5]));
        assert!(result.is_err());
        assert!(matches!(result, Err(AnimationError::NegativeTime(_))));
    }

    #[test]
    fn test_invalid_keyframe_index_error() {
        let names = vec!["a".to_string()];
        let mut tl = ExpressionTimeline::new(names);
        let result = tl.remove_keyframe(99);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(AnimationError::InvalidKeyframeIndex(99))
        ));
    }

    #[test]
    fn test_animation_error_display() {
        let e = AnimationError::DimensionMismatch {
            expected: 3,
            got: 5,
        };
        let s = format!("{e}");
        assert!(s.contains('3') && s.contains('5'));

        let e2 = AnimationError::NegativeTime(-1.0);
        let s2 = format!("{e2}");
        assert!(s2.contains("negative") || s2.contains("-1"));
    }
}
