//! Mesh morphing — smooth blending between FLAME mesh poses/shapes.
//!
//! Provides interpolation strategies, morph targets, animation clips, and
//! sequence utilities for transitioning between different FLAME mesh states
//! (vertices, normals, blend shape weights).

use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during mesh morphing operations.
#[derive(Debug, thiserror::Error)]
pub enum MorphError {
    #[error("Mesh A has {a} vertices but mesh B has {b}")]
    VertexCountMismatch { a: usize, b: usize },

    #[error("Sequence is empty")]
    EmptySequence,

    #[error("Invalid blend weight {0}: must be in [0, 1]")]
    InvalidBlendWeight(f32),

    #[error("Morph target index {idx} out of range (have {n})")]
    TargetIndexOutOfRange { idx: usize, n: usize },

    #[error("Weights sum to {sum:.4} but must sum to 1.0 (tolerance {tol:.4})")]
    WeightSumError { sum: f32, tol: f32 },

    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// MorphTarget
// ─────────────────────────────────────────────────────────────────────────────

/// A single morph target: per-vertex displacement from the base mesh.
#[derive(Debug, Clone)]
pub struct MorphTarget {
    /// Human-readable name of this morph target.
    pub name: String,
    /// Per-vertex displacement vectors `[dx, dy, dz]`.
    pub deltas: Vec<[f32; 3]>,
    /// Current blend weight in `[0, 1]`.
    pub weight: f32,
}

impl MorphTarget {
    /// Create a new morph target with the given name and deltas.
    /// Weight starts at `0.0`.
    pub fn new(name: impl Into<String>, deltas: Vec<[f32; 3]>) -> Self {
        Self {
            name: name.into(),
            deltas,
            weight: 0.0,
        }
    }

    /// Create a zero-displacement morph target (all deltas are `[0, 0, 0]`).
    pub fn zero(name: impl Into<String>, n_verts: usize) -> Self {
        Self {
            name: name.into(),
            deltas: vec![[0.0_f32; 3]; n_verts],
            weight: 0.0,
        }
    }

    /// Maximum displacement magnitude across all vertices.
    #[must_use]
    pub fn max_delta(&self) -> f32 {
        self.deltas
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
            .fold(0.0_f32, f32::max)
    }

    /// Root-mean-square displacement magnitude across all vertices.
    #[must_use]
    pub fn rms_delta(&self) -> f32 {
        if self.deltas.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = self
            .deltas
            .iter()
            .map(|d| d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            .sum();
        (sum_sq / self.deltas.len() as f32).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MorphTargetSet
// ─────────────────────────────────────────────────────────────────────────────

/// A collection of morph targets sharing the same base mesh.
#[derive(Debug, Clone)]
pub struct MorphTargetSet {
    /// Base (neutral) vertex positions.
    pub base_vertices: Vec<[f32; 3]>,
    /// All morph targets in this set.
    pub targets: Vec<MorphTarget>,
}

impl MorphTargetSet {
    /// Create a new morph target set from the given base vertices.
    #[must_use]
    pub fn new(base_vertices: Vec<[f32; 3]>) -> Self {
        Self {
            base_vertices,
            targets: Vec::new(),
        }
    }

    /// Add a morph target, returning an error if the vertex count does not match.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn add_target(&mut self, target: MorphTarget) -> Result<(), MorphError> {
        let base_n = self.base_vertices.len();
        let tgt_n = target.deltas.len();
        if tgt_n != base_n {
            return Err(MorphError::VertexCountMismatch {
                a: base_n,
                b: tgt_n,
            });
        }
        self.targets.push(target);
        Ok(())
    }

    /// Set the blend weight of target `idx`, validating the value is in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn set_weight(&mut self, idx: usize, weight: f32) -> Result<(), MorphError> {
        let n = self.targets.len();
        if idx >= n {
            return Err(MorphError::TargetIndexOutOfRange { idx, n });
        }
        if !(0.0..=1.0).contains(&weight) {
            return Err(MorphError::InvalidBlendWeight(weight));
        }
        self.targets[idx].weight = weight;
        Ok(())
    }

    /// Evaluate the blended mesh: `base + Σ(target.deltas * target.weight)`.
    ///
    /// A target whose `deltas` length does not match `base_vertices` is
    /// skipped (matching the free function [`apply_morph_targets`]'s
    /// guard). `add_target` validates this invariant on insertion, but
    /// `targets` is a public `Vec<MorphTarget>` (so a caller can `push`
    /// directly) and `MorphTarget::deltas` is itself public and can be
    /// resized after insertion, so `add_target`'s check alone cannot be
    /// relied on here.
    #[must_use]
    pub fn evaluate(&self) -> Vec<[f32; 3]> {
        let mut result = self.base_vertices.clone();
        let n = result.len();
        for target in &self.targets {
            let w = target.weight;
            if w == 0.0 || target.deltas.len() != n {
                continue;
            }
            for (i, r) in result.iter_mut().enumerate() {
                let d = target.deltas[i];
                r[0] += d[0] * w;
                r[1] += d[1] * w;
                r[2] += d[2] * w;
            }
        }
        result
    }

    /// Number of morph targets in this set.
    #[must_use]
    pub fn n_targets(&self) -> usize {
        self.targets.len()
    }

    /// Sum of all current blend weights.
    #[must_use]
    pub fn total_weights(&self) -> f32 {
        self.targets.iter().map(|t| t.weight).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Interpolation strategy
// ─────────────────────────────────────────────────────────────────────────────

/// Interpolation strategies available for mesh morphing.
#[derive(Debug, Clone, PartialEq)]
pub enum MorphInterpolation {
    /// Simple linear interpolation (LERP).
    Linear,
    /// Cosine-eased interpolation (smooth start and end).
    Cosine,
    /// Cubic Hermite (Catmull-Rom) interpolation.
    CubicHermite,
    /// Step function — no blending; jumps at `t = 0.5`.
    Step,
}

// ─────────────────────────────────────────────────────────────────────────────
// MorphKeyframe / MorphLoopMode / MorphClip
// ─────────────────────────────────────────────────────────────────────────────

/// A single keyframe in a morph animation.
#[derive(Debug, Clone)]
pub struct MorphKeyframe {
    /// Time of this keyframe in seconds.
    pub time: f32,
    /// Vertex positions at this keyframe.
    pub vertices: Vec<[f32; 3]>,
}

/// How a [`MorphClip`] behaves when playback reaches the end.
#[derive(Debug, Clone, PartialEq)]
pub enum MorphLoopMode {
    /// Play once and stop.
    Once,
    /// Loop back to the beginning.
    Loop,
    /// Bounce back and forth (ping-pong).
    PingPong,
}

/// A morph animation clip containing keyframes.
#[derive(Debug, Clone)]
pub struct MorphClip {
    /// Name of the clip.
    pub name: String,
    /// Keyframes sorted ascending by time.
    pub keyframes: Vec<MorphKeyframe>,
    /// Total duration of the clip in seconds.
    pub duration: f32,
    /// Looping behaviour.
    pub loop_mode: MorphLoopMode,
}

impl MorphClip {
    /// Create a new clip from a list of keyframes.
    ///
    /// Keyframes are sorted by time. Returns [`MorphError::EmptySequence`] if
    /// `keyframes` is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn new(
        name: impl Into<String>,
        mut keyframes: Vec<MorphKeyframe>,
    ) -> Result<Self, MorphError> {
        if keyframes.is_empty() {
            return Err(MorphError::EmptySequence);
        }
        keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let duration = keyframes.last().map_or(0.0, |k| k.time);
        Ok(Self {
            name: name.into(),
            keyframes,
            duration,
            loop_mode: MorphLoopMode::Once,
        })
    }

    /// Total duration of the clip.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Number of keyframes.
    #[must_use]
    pub fn n_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    /// Sample the clip at `time` (seconds) using the given interpolation strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn sample(
        &self,
        time: f32,
        interp: &MorphInterpolation,
    ) -> Result<Vec<[f32; 3]>, MorphError> {
        if self.keyframes.is_empty() {
            return Err(MorphError::EmptySequence);
        }

        // Resolve looping
        let resolved_time = self.resolve_time(time);

        // Fast-path: single keyframe
        if self.keyframes.len() == 1 {
            return Ok(self.keyframes[0].vertices.clone());
        }

        // Clamp to [first, last]
        let first_t = self.keyframes[0].time;
        let last_t = self.keyframes[self.keyframes.len() - 1].time;

        if resolved_time <= first_t {
            return Ok(self.keyframes[0].vertices.clone());
        }
        if resolved_time >= last_t {
            return Ok(self.keyframes[self.keyframes.len() - 1].vertices.clone());
        }

        // Binary-search for the surrounding keyframe index
        let hi = self.keyframes.partition_point(|k| k.time <= resolved_time);
        // hi is the first keyframe with time > resolved_time
        // lo is the keyframe just before
        let lo = hi.saturating_sub(1);
        let lo = lo.min(self.keyframes.len() - 2);
        let hi = lo + 1;

        let t0 = self.keyframes[lo].time;
        let t1 = self.keyframes[hi].time;
        let span = t1 - t0;

        let t = if span.abs() < 1e-9 {
            0.0
        } else {
            ((resolved_time - t0) / span).clamp(0.0, 1.0)
        };

        match interp {
            MorphInterpolation::Linear => morph_lerp(
                &self.keyframes[lo].vertices,
                &self.keyframes[hi].vertices,
                t,
            ),
            MorphInterpolation::Cosine => morph_cosine(
                &self.keyframes[lo].vertices,
                &self.keyframes[hi].vertices,
                t,
            ),
            MorphInterpolation::Step => {
                if t < 0.5 {
                    Ok(self.keyframes[lo].vertices.clone())
                } else {
                    Ok(self.keyframes[hi].vertices.clone())
                }
            }
            MorphInterpolation::CubicHermite => {
                // Need 4 surrounding keyframes; clamp at boundaries
                let v0 = if lo == 0 {
                    &self.keyframes[lo].vertices
                } else {
                    &self.keyframes[lo - 1].vertices
                };
                let v1 = &self.keyframes[lo].vertices;
                let v2 = &self.keyframes[hi].vertices;
                let v3 = if hi + 1 >= self.keyframes.len() {
                    &self.keyframes[hi].vertices
                } else {
                    &self.keyframes[hi + 1].vertices
                };
                morph_cubic_hermite(v0, v1, v2, v3, t)
            }
        }
    }

    /// Apply loop mode to compute the effective playback time.
    fn resolve_time(&self, time: f32) -> f32 {
        let dur = self.duration;
        if dur <= 0.0 {
            return 0.0;
        }
        match self.loop_mode {
            MorphLoopMode::Once => time,
            MorphLoopMode::Loop => {
                let t = time % dur;
                if t < 0.0 {
                    t + dur
                } else {
                    t
                }
            }
            MorphLoopMode::PingPong => {
                // period = 2 * duration
                let period = 2.0 * dur;
                let t = time % period;
                let t = if t < 0.0 { t + period } else { t };
                if t <= dur {
                    t
                } else {
                    period - t
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core interpolation functions
// ─────────────────────────────────────────────────────────────────────────────

/// Linear interpolation between two vertex arrays.
///
/// Returns `a` when `t = 0` and `b` when `t = 1`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn morph_lerp(a: &[[f32; 3]], b: &[[f32; 3]], t: f32) -> Result<Vec<[f32; 3]>, MorphError> {
    let n = a.len();
    if b.len() != n {
        return Err(MorphError::VertexCountMismatch { a: n, b: b.len() });
    }
    let t_c = t.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t_c;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(va, vb)| {
            [
                va[0] * one_minus_t + vb[0] * t_c,
                va[1] * one_minus_t + vb[1] * t_c,
                va[2] * one_minus_t + vb[2] * t_c,
            ]
        })
        .collect())
}

/// Cosine-eased interpolation between two vertex arrays.
///
/// Applies `t_ease = (1 − cos(t × π)) / 2` before linear interpolation,
/// yielding smooth acceleration and deceleration.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn morph_cosine(a: &[[f32; 3]], b: &[[f32; 3]], t: f32) -> Result<Vec<[f32; 3]>, MorphError> {
    let n = a.len();
    if b.len() != n {
        return Err(MorphError::VertexCountMismatch { a: n, b: b.len() });
    }
    let t_c = t.clamp(0.0, 1.0);
    let t_ease = (1.0 - (t_c * PI).cos()) / 2.0;
    morph_lerp(a, b, t_ease)
}

/// Cubic Hermite (Catmull-Rom) interpolation between four vertex arrays.
///
/// `v1` and `v2` are the endpoints; `v0` and `v3` are the outer control
/// points used to compute tangents.
///
/// - At `t = 0`, the result equals `v1`.
/// - At `t = 1`, the result equals `v2`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn morph_cubic_hermite(
    v0: &[[f32; 3]],
    v1: &[[f32; 3]],
    v2: &[[f32; 3]],
    v3: &[[f32; 3]],
    t: f32,
) -> Result<Vec<[f32; 3]>, MorphError> {
    let n = v1.len();
    if v0.len() != n || v2.len() != n || v3.len() != n {
        return Err(MorphError::VertexCountMismatch {
            a: n,
            b: if v0.len() != n {
                v0.len()
            } else if v2.len() != n {
                v2.len()
            } else {
                v3.len()
            },
        });
    }
    let t_c = t.clamp(0.0, 1.0);
    let t2 = t_c * t_c;
    let t3 = t2 * t_c;

    // Hermite basis functions
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t_c;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    let mut out = vec![[0.0_f32; 3]; n];
    for i in 0..n {
        for c in 0..3 {
            let p0 = v1[i][c];
            let p1 = v2[i][c];
            // Catmull-Rom tangents
            let m0 = (v2[i][c] - v0[i][c]) / 2.0;
            let m1 = (v3[i][c] - v1[i][c]) / 2.0;
            out[i][c] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
        }
    }
    Ok(out)
}

/// Apply the chosen interpolation strategy between two vertex arrays.
///
/// For [`MorphInterpolation::CubicHermite`], the endpoints are duplicated
/// (`morph_cubic_hermite(a, a, b, b, t)`). This does *not* produce zero
/// tangents: with `v0 = v1 = a` and `v2 = v3 = b`, the Catmull-Rom tangent
/// formula collapses to `m0 = m1 = (b - a) / 2` at both ends (the secant
/// slope), giving the ease curve `a + (b - a) * (1.5*t^2 - t^3 + 0.5*t)`.
/// This is smooth (no oscillation, matching `a` at `t=0` and `b` at `t=1`),
/// but it is *not* the same curve as a true zero-tangent Hermite/smoothstep
/// (`3*t^2 - 2*t^3`) -- the two differ noticeably in the first and last
/// thirds of `[0, 1]` (e.g. at `t = 0.25` this curve gives `~0.2031` vs
/// smoothstep's `~0.1563`).
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn morph_interpolate(
    a: &[[f32; 3]],
    b: &[[f32; 3]],
    t: f32,
    interp: &MorphInterpolation,
) -> Result<Vec<[f32; 3]>, MorphError> {
    match interp {
        MorphInterpolation::Linear => morph_lerp(a, b, t),
        MorphInterpolation::Cosine => morph_cosine(a, b, t),
        MorphInterpolation::CubicHermite => morph_cubic_hermite(a, a, b, b, t),
        MorphInterpolation::Step => {
            if t < 0.5 {
                // Validate sizes before returning
                let n = a.len();
                if b.len() != n {
                    return Err(MorphError::VertexCountMismatch { a: n, b: b.len() });
                }
                Ok(a.to_vec())
            } else {
                let n = a.len();
                if b.len() != n {
                    return Err(MorphError::VertexCountMismatch { a: n, b: b.len() });
                }
                Ok(b.to_vec())
            }
        }
    }
}

/// Blend multiple vertex arrays with given weights.
///
/// `weights` must have the same length as `meshes`, all meshes must share the
/// same vertex count, and the weights must sum to approximately `1.0`
/// (tolerance `1e-4`).
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn morph_blend_n(meshes: &[&[[f32; 3]]], weights: &[f32]) -> Result<Vec<[f32; 3]>, MorphError> {
    if meshes.is_empty() {
        return Err(MorphError::EmptySequence);
    }
    if weights.len() != meshes.len() {
        return Err(MorphError::InvalidParam(format!(
            "meshes.len() = {} but weights.len() = {}",
            meshes.len(),
            weights.len()
        )));
    }

    let n = meshes[0].len();
    for (k, mesh) in meshes.iter().enumerate() {
        if mesh.len() != n {
            return Err(MorphError::VertexCountMismatch {
                a: n,
                b: mesh.len(),
            });
        }
        let w = weights[k];
        if !(-1e-6..=1.0 + 1e-6).contains(&w) {
            return Err(MorphError::InvalidBlendWeight(w));
        }
    }

    let tol = 1e-4_f32;
    let sum: f32 = weights.iter().sum();
    if (sum - 1.0).abs() > tol {
        return Err(MorphError::WeightSumError { sum, tol });
    }

    let mut out = vec![[0.0_f32; 3]; n];
    for (mesh, &w) in meshes.iter().zip(weights.iter()) {
        for (i, v) in mesh.iter().enumerate() {
            out[i][0] += v[0] * w;
            out[i][1] += v[1] * w;
            out[i][2] += v[2] * w;
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Morph target helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a slice of morph targets to `base` vertices and return the result.
///
/// `result[i] = base[i] + Σ(targets[k].deltas[i] × targets[k].weight)`
#[must_use]
pub fn apply_morph_targets(base: &[[f32; 3]], targets: &[MorphTarget]) -> Vec<[f32; 3]> {
    let n = base.len();
    let mut result = base.to_vec();
    for target in targets {
        let w = target.weight;
        if w == 0.0 || target.deltas.len() != n {
            continue;
        }
        for (i, r) in result.iter_mut().enumerate() {
            let d = target.deltas[i];
            r[0] += d[0] * w;
            r[1] += d[1] * w;
            r[2] += d[2] * w;
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequence utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a sliding-window average to a sequence of vertex arrays.
///
/// `window` must be `>= 1` and `<= sequence.len()`.
///
/// For each frame `i`, the smoothed output is the average of frames in the
/// range `[max(0, i − window/2), min(n − 1, i + window/2)]`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn smooth_morph_sequence(
    sequence: &[Vec<[f32; 3]>],
    window: usize,
) -> Result<Vec<Vec<[f32; 3]>>, MorphError> {
    let n = sequence.len();
    if n == 0 {
        return Err(MorphError::EmptySequence);
    }
    if window == 0 {
        return Err(MorphError::InvalidParam("window must be >= 1".to_string()));
    }
    if window > n {
        return Err(MorphError::InvalidParam(format!(
            "window ({window}) must be <= sequence length ({n})"
        )));
    }

    let n_verts = sequence[0].len();
    let half = window / 2;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let count = (hi - lo) as f32;
        let mut avg = vec![[0.0_f32; 3]; n_verts];
        for frame in &sequence[lo..hi] {
            for (j, v) in frame.iter().enumerate() {
                if j < n_verts {
                    avg[j][0] += v[0];
                    avg[j][1] += v[1];
                    avg[j][2] += v[2];
                }
            }
        }
        for v in &mut avg {
            v[0] /= count;
            v[1] /= count;
            v[2] /= count;
        }
        result.push(avg);
    }
    Ok(result)
}

/// Resample a morph sequence to a new number of frames using linear
/// interpolation.
///
/// The first and last frames of the original sequence are preserved exactly.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn resample_morph_sequence(
    sequence: &[Vec<[f32; 3]>],
    n_frames: usize,
) -> Result<Vec<Vec<[f32; 3]>>, MorphError> {
    let n = sequence.len();
    if n == 0 {
        return Err(MorphError::EmptySequence);
    }
    if n_frames == 0 {
        return Err(MorphError::InvalidParam(
            "n_frames must be >= 1".to_string(),
        ));
    }

    let n_verts = sequence[0].len();
    for frame in sequence.iter().skip(1) {
        if frame.len() != n_verts {
            return Err(MorphError::VertexCountMismatch {
                a: n_verts,
                b: frame.len(),
            });
        }
    }

    // A single source frame has no interval to interpolate within: every
    // output frame is simply that frame, regardless of `n_frames`. Without
    // this early return, `n - 2` below underflows (`n == 1`), which panics
    // in debug builds (subtract-with-overflow) and wraps to `usize::MAX` in
    // release, causing an out-of-bounds `&sequence[hi]` panic just after.
    if n == 1 {
        return Ok(vec![sequence[0].clone(); n_frames]);
    }

    if n_frames == 1 {
        // Return the first frame
        return Ok(vec![sequence[0].clone()]);
    }

    let mut result = Vec::with_capacity(n_frames);
    for j in 0..n_frames {
        // Fractional position in [0, n-1]
        let pos = j as f32 / (n_frames - 1) as f32 * (n - 1) as f32;
        let lo = (pos.floor() as usize).min(n - 2);
        let hi = lo + 1;
        let t = pos - lo as f32;

        let frame_lo = &sequence[lo];
        let frame_hi = &sequence[hi];

        let mut out = vec![[0.0_f32; 3]; n_verts];
        let one_minus_t = 1.0 - t;
        for k in 0..n_verts {
            out[k][0] = frame_lo[k][0] * one_minus_t + frame_hi[k][0] * t;
            out[k][1] = frame_lo[k][1] * one_minus_t + frame_hi[k][1] * t;
            out[k][2] = frame_lo[k][2] * one_minus_t + frame_hi[k][2] * t;
        }
        result.push(out);
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Delta / magnitude utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-vertex displacement from `source` to `target`.
///
/// Returns an error if the vertex counts differ.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn compute_morph_delta(
    source: &[[f32; 3]],
    target: &[[f32; 3]],
) -> Result<Vec<[f32; 3]>, MorphError> {
    let n = source.len();
    if target.len() != n {
        return Err(MorphError::VertexCountMismatch {
            a: n,
            b: target.len(),
        });
    }
    Ok(source
        .iter()
        .zip(target.iter())
        .map(|(s, t)| [t[0] - s[0], t[1] - s[1], t[2] - s[2]])
        .collect())
}

/// Compute a sequence of per-vertex deltas, each relative to the first frame.
///
/// The first element of the result is always all-zeros.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn compute_delta_sequence(
    sequence: &[Vec<[f32; 3]>],
) -> Result<Vec<Vec<[f32; 3]>>, MorphError> {
    if sequence.is_empty() {
        return Err(MorphError::EmptySequence);
    }
    let base = &sequence[0];
    let mut result = Vec::with_capacity(sequence.len());
    for frame in sequence {
        result.push(compute_morph_delta(base, frame)?);
    }
    Ok(result)
}

/// Compute per-vertex displacement magnitudes for a delta array.
#[must_use]
pub fn delta_magnitudes(deltas: &[[f32; 3]]) -> Vec<f32> {
    deltas
        .iter()
        .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
        .collect()
}

/// Maximum displacement magnitude in a delta array.
#[must_use]
pub fn max_delta_magnitude(deltas: &[[f32; 3]]) -> f32 {
    delta_magnitudes(deltas).into_iter().fold(0.0_f32, f32::max)
}

/// Mean displacement magnitude across all deltas.
#[must_use]
pub fn mean_delta_magnitude(deltas: &[[f32; 3]]) -> f32 {
    if deltas.is_empty() {
        return 0.0;
    }
    let sum: f32 = delta_magnitudes(deltas).iter().sum();
    sum / deltas.len() as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Display helper
// ─────────────────────────────────────────────────────────────────────────────

/// Format a [`MorphClip`] summary as a human-readable string.
#[must_use]
pub fn format_morph_clip(clip: &MorphClip) -> String {
    let n_verts = clip.keyframes.first().map_or(0, |k| k.vertices.len());
    let loop_str = match clip.loop_mode {
        MorphLoopMode::Once => "once",
        MorphLoopMode::Loop => "loop",
        MorphLoopMode::PingPong => "ping-pong",
    };
    format!(
        "MorphClip '{}': {} keyframe(s), {:.3}s duration, {} vertices, {} loop",
        clip.name,
        clip.keyframes.len(),
        clip.duration,
        n_verts,
        loop_str,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_verts(n: usize, val: f32) -> Vec<[f32; 3]> {
        vec![[val; 3]; n]
    }

    fn verts_approx_eq(a: &[[f32; 3]], b: &[[f32; 3]], eps: f32) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(va, vb)| {
                (va[0] - vb[0]).abs() < eps
                    && (va[1] - vb[1]).abs() < eps
                    && (va[2] - vb[2]).abs() < eps
            })
    }

    // ── morph_lerp ───────────────────────────────────────────────────────────

    #[test]
    fn test_morph_lerp_t0() {
        let a = make_verts(4, 0.0);
        let b = make_verts(4, 1.0);
        let r = morph_lerp(&a, &b, 0.0).expect("lerp t=0");
        assert!(verts_approx_eq(&r, &a, 1e-6));
    }

    #[test]
    fn test_morph_lerp_t1() {
        let a = make_verts(4, 0.0);
        let b = make_verts(4, 1.0);
        let r = morph_lerp(&a, &b, 1.0).expect("lerp t=1");
        assert!(verts_approx_eq(&r, &b, 1e-6));
    }

    #[test]
    fn test_morph_lerp_midpoint() {
        let a = make_verts(3, 0.0);
        let b = make_verts(3, 2.0);
        let r = morph_lerp(&a, &b, 0.5).expect("lerp t=0.5");
        let expected = make_verts(3, 1.0);
        assert!(verts_approx_eq(&r, &expected, 1e-5));
    }

    #[test]
    fn test_morph_lerp_mismatch() {
        let a = make_verts(3, 0.0);
        let b = make_verts(4, 1.0);
        assert!(matches!(
            morph_lerp(&a, &b, 0.5),
            Err(MorphError::VertexCountMismatch { .. })
        ));
    }

    #[test]
    fn test_morph_lerp_clamps_above_1() {
        let a = make_verts(2, 0.0);
        let b = make_verts(2, 4.0);
        let r = morph_lerp(&a, &b, 2.0).expect("lerp clamped");
        assert!(verts_approx_eq(&r, &b, 1e-5));
    }

    #[test]
    fn test_morph_lerp_clamps_below_0() {
        let a = make_verts(2, 0.0);
        let b = make_verts(2, 4.0);
        let r = morph_lerp(&a, &b, -1.0).expect("lerp clamped low");
        assert!(verts_approx_eq(&r, &a, 1e-5));
    }

    // ── morph_cosine ─────────────────────────────────────────────────────────

    #[test]
    fn test_morph_cosine_t0() {
        let a = make_verts(3, 0.0);
        let b = make_verts(3, 1.0);
        let r = morph_cosine(&a, &b, 0.0).expect("cosine t=0");
        assert!(verts_approx_eq(&r, &a, 1e-6));
    }

    #[test]
    fn test_morph_cosine_t1() {
        let a = make_verts(3, 0.0);
        let b = make_verts(3, 1.0);
        let r = morph_cosine(&a, &b, 1.0).expect("cosine t=1");
        assert!(verts_approx_eq(&r, &b, 1e-5));
    }

    #[test]
    fn test_morph_cosine_differs_from_lerp_at_midpoint() {
        let a = make_verts(2, 0.0);
        let b = make_verts(2, 1.0);
        let cos_r = morph_cosine(&a, &b, 0.3).expect("cosine");
        let lin_r = morph_lerp(&a, &b, 0.3).expect("lerp");
        // Cosine easing should differ from linear at non-endpoint t
        let differs = cos_r
            .iter()
            .zip(lin_r.iter())
            .any(|(c, l)| (c[0] - l[0]).abs() > 1e-6);
        assert!(differs, "cosine and lerp should differ at t=0.3");
    }

    #[test]
    fn test_morph_cosine_mismatch() {
        let a = make_verts(3, 0.0);
        let b = make_verts(5, 1.0);
        assert!(matches!(
            morph_cosine(&a, &b, 0.5),
            Err(MorphError::VertexCountMismatch { .. })
        ));
    }

    // ── morph_cubic_hermite ──────────────────────────────────────────────────

    #[test]
    fn test_morph_cubic_hermite_t0_returns_v1() {
        let v0 = make_verts(2, 0.0);
        let v1 = make_verts(2, 1.0);
        let v2 = make_verts(2, 3.0);
        let v3 = make_verts(2, 4.0);
        let r = morph_cubic_hermite(&v0, &v1, &v2, &v3, 0.0).expect("hermite t=0");
        assert!(verts_approx_eq(&r, &v1, 1e-5));
    }

    #[test]
    fn test_morph_cubic_hermite_t1_returns_v2() {
        let v0 = make_verts(2, 0.0);
        let v1 = make_verts(2, 1.0);
        let v2 = make_verts(2, 3.0);
        let v3 = make_verts(2, 4.0);
        let r = morph_cubic_hermite(&v0, &v1, &v2, &v3, 1.0).expect("hermite t=1");
        assert!(verts_approx_eq(&r, &v2, 1e-5));
    }

    #[test]
    fn test_morph_cubic_hermite_midpoint_is_between() {
        let v0 = make_verts(1, 0.0);
        let v1 = make_verts(1, 1.0);
        let v2 = make_verts(1, 3.0);
        let v3 = make_verts(1, 4.0);
        let r = morph_cubic_hermite(&v0, &v1, &v2, &v3, 0.5).expect("hermite mid");
        assert!(
            r[0][0] > 1.0 && r[0][0] < 3.0,
            "mid should be between v1 and v2"
        );
    }

    #[test]
    fn test_morph_cubic_hermite_mismatch() {
        let v0 = make_verts(2, 0.0);
        let v1 = make_verts(2, 1.0);
        let v2 = make_verts(3, 3.0); // mismatched
        let v3 = make_verts(2, 4.0);
        assert!(matches!(
            morph_cubic_hermite(&v0, &v1, &v2, &v3, 0.5),
            Err(MorphError::VertexCountMismatch { .. })
        ));
    }

    // ── morph_interpolate ────────────────────────────────────────────────────

    #[test]
    fn test_morph_interpolate_linear_t0() {
        let a = make_verts(3, 0.0);
        let b = make_verts(3, 1.0);
        let r = morph_interpolate(&a, &b, 0.0, &MorphInterpolation::Linear).expect("lin t=0");
        assert!(verts_approx_eq(&r, &a, 1e-6));
    }

    #[test]
    fn test_morph_interpolate_cosine_t1() {
        let a = make_verts(3, 0.0);
        let b = make_verts(3, 1.0);
        let r = morph_interpolate(&a, &b, 1.0, &MorphInterpolation::Cosine).expect("cos t=1");
        assert!(verts_approx_eq(&r, &b, 1e-5));
    }

    #[test]
    fn test_morph_interpolate_step_low() {
        let a = make_verts(2, 0.0);
        let b = make_verts(2, 1.0);
        let r = morph_interpolate(&a, &b, 0.3, &MorphInterpolation::Step).expect("step low");
        assert!(verts_approx_eq(&r, &a, 1e-6));
    }

    #[test]
    fn test_morph_interpolate_step_high() {
        let a = make_verts(2, 0.0);
        let b = make_verts(2, 1.0);
        let r = morph_interpolate(&a, &b, 0.7, &MorphInterpolation::Step).expect("step high");
        assert!(verts_approx_eq(&r, &b, 1e-6));
    }

    #[test]
    fn test_morph_interpolate_cubic_hermite_t0() {
        let a = make_verts(3, 0.0);
        let b = make_verts(3, 2.0);
        let r =
            morph_interpolate(&a, &b, 0.0, &MorphInterpolation::CubicHermite).expect("cubic t=0");
        assert!(verts_approx_eq(&r, &a, 1e-5));
    }

    #[test]
    fn test_morph_interpolate_cubic_hermite_t1() {
        let a = make_verts(3, 0.0);
        let b = make_verts(3, 2.0);
        let r =
            morph_interpolate(&a, &b, 1.0, &MorphInterpolation::CubicHermite).expect("cubic t=1");
        assert!(verts_approx_eq(&r, &b, 1e-5));
    }

    #[test]
    fn test_morph_interpolate_cubic_hermite_matches_documented_ease_curve() {
        // Verify the doc's claim: duplicated endpoints give tangents
        // (b-a)/2 (not zero), producing a + (b-a)*(1.5t^2 - t^3 + 0.5t),
        // which differs from a true zero-tangent smoothstep (3t^2 - 2t^3).
        let a = make_verts(1, 0.0);
        let b = make_verts(1, 1.0);
        let t = 0.25_f32;
        let r = morph_interpolate(&a, &b, t, &MorphInterpolation::CubicHermite).expect("cubic");
        let expected = 1.5 * t * t - t * t * t + 0.5 * t; // a=0, b=1, so a+(b-a)*f = f
        let smoothstep = 3.0 * t * t - 2.0 * t * t * t;
        assert!(
            (r[0][0] - expected).abs() < 1e-5,
            "expected documented ease curve value {expected}, got {}",
            r[0][0]
        );
        assert!(
            (r[0][0] - smoothstep).abs() > 1e-3,
            "sanity check: the implemented curve must differ from smoothstep at t=0.25"
        );
    }

    // ── morph_blend_n ────────────────────────────────────────────────────────

    #[test]
    fn test_morph_blend_n_equal_weights() {
        let a = make_verts(4, 0.0);
        let b = make_verts(4, 2.0);
        let r = morph_blend_n(&[a.as_slice(), b.as_slice()], &[0.5, 0.5]).expect("blend equal");
        let expected = make_verts(4, 1.0);
        assert!(verts_approx_eq(&r, &expected, 1e-5));
    }

    #[test]
    fn test_morph_blend_n_all_weight_to_first() {
        let a = make_verts(3, 5.0);
        let b = make_verts(3, 10.0);
        let r = morph_blend_n(&[a.as_slice(), b.as_slice()], &[1.0, 0.0]).expect("blend first");
        assert!(verts_approx_eq(&r, &a, 1e-5));
    }

    #[test]
    fn test_morph_blend_n_weight_sum_error() {
        let a = make_verts(3, 0.0);
        let b = make_verts(3, 1.0);
        assert!(matches!(
            morph_blend_n(&[a.as_slice(), b.as_slice()], &[0.5, 0.3]),
            Err(MorphError::WeightSumError { .. })
        ));
    }

    #[test]
    fn test_morph_blend_n_three_meshes() {
        let a = make_verts(2, 0.0);
        let b = make_verts(2, 3.0);
        let c = make_verts(2, 6.0);
        let r = morph_blend_n(
            &[a.as_slice(), b.as_slice(), c.as_slice()],
            &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        )
        .expect("blend three");
        // expected average = 3.0
        assert!((r[0][0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_morph_blend_n_mismatch_lengths() {
        let a = make_verts(3, 0.0);
        let b = make_verts(2, 1.0);
        assert!(matches!(
            morph_blend_n(&[a.as_slice(), b.as_slice()], &[0.5, 0.5]),
            Err(MorphError::VertexCountMismatch { .. })
        ));
    }

    // ── MorphTarget ──────────────────────────────────────────────────────────

    #[test]
    fn test_morph_target_new() {
        let d = vec![[1.0_f32, 0.0, 0.0]; 5];
        let t = MorphTarget::new("smile", d.clone());
        assert_eq!(t.name, "smile");
        assert_eq!(t.deltas.len(), 5);
        assert_eq!(t.weight, 0.0);
    }

    #[test]
    fn test_morph_target_zero() {
        let t = MorphTarget::zero("flat", 10);
        assert_eq!(t.deltas.len(), 10);
        assert_eq!(t.max_delta(), 0.0);
        assert_eq!(t.rms_delta(), 0.0);
    }

    #[test]
    fn test_morph_target_max_delta() {
        let d = vec![[3.0_f32, 4.0, 0.0]; 2]; // magnitude = 5.0
        let t = MorphTarget::new("t", d);
        assert!((t.max_delta() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_morph_target_rms_delta() {
        // All same delta → rms = max
        let d = vec![[3.0_f32, 4.0, 0.0]; 3];
        let t = MorphTarget::new("t", d);
        assert!((t.rms_delta() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_morph_target_rms_empty() {
        let t = MorphTarget::new("empty", vec![]);
        assert_eq!(t.rms_delta(), 0.0);
    }

    // ── MorphTargetSet ───────────────────────────────────────────────────────

    #[test]
    fn test_morph_target_set_add_mismatch() {
        let mut set = MorphTargetSet::new(make_verts(5, 0.0));
        let bad = MorphTarget::new("bad", make_verts(3, 1.0));
        assert!(matches!(
            set.add_target(bad),
            Err(MorphError::VertexCountMismatch { .. })
        ));
    }

    #[test]
    fn test_morph_target_set_add_ok() {
        let mut set = MorphTargetSet::new(make_verts(5, 0.0));
        let good = MorphTarget::new("good", make_verts(5, 1.0));
        set.add_target(good).expect("add ok");
        assert_eq!(set.n_targets(), 1);
    }

    #[test]
    fn test_morph_target_set_set_weight_out_of_range_idx() {
        let mut set = MorphTargetSet::new(make_verts(3, 0.0));
        assert!(matches!(
            set.set_weight(0, 0.5),
            Err(MorphError::TargetIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn test_morph_target_set_set_weight_invalid_value() {
        let mut set = MorphTargetSet::new(make_verts(3, 0.0));
        set.add_target(MorphTarget::new("t", make_verts(3, 1.0)))
            .expect("add");
        assert!(matches!(
            set.set_weight(0, 1.5),
            Err(MorphError::InvalidBlendWeight(_))
        ));
    }

    #[test]
    fn test_morph_target_set_evaluate_zero_weight() {
        let base = make_verts(4, 1.0);
        let mut set = MorphTargetSet::new(base.clone());
        let mut t = MorphTarget::new("t", make_verts(4, 5.0));
        t.weight = 0.0;
        set.add_target(t).expect("add");
        let result = set.evaluate();
        assert!(verts_approx_eq(&result, &base, 1e-6));
    }

    #[test]
    fn test_morph_target_set_evaluate_full_weight() {
        let base = make_verts(3, 0.0);
        let delta = make_verts(3, 2.0);
        let mut set = MorphTargetSet::new(base.clone());
        let mut t = MorphTarget::new("t", delta.clone());
        t.weight = 1.0;
        set.add_target(t).expect("add");
        let result = set.evaluate();
        let expected = make_verts(3, 2.0); // 0 + 1.0 * 2.0
        assert!(verts_approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_morph_target_set_evaluate_skips_mismatched_deltas_len() {
        // `add_target` validates deltas.len() == base_vertices.len(), but
        // `targets` is a public Vec<MorphTarget> that a caller can push
        // into directly, and `MorphTarget::deltas` can itself be resized
        // after insertion -- both bypass that check. `evaluate()` must not
        // panic or read out of bounds when that happens; it should simply
        // skip the mismatched target, exactly like the free function
        // `apply_morph_targets` already does.
        let base = make_verts(4, 1.0);
        let mut set = MorphTargetSet::new(base.clone());
        let mut t = MorphTarget::new("t", make_verts(4, 5.0));
        t.weight = 1.0;
        set.targets.push(t); // bypass add_target's validation
        set.targets[0].deltas = make_verts(2, 9.0); // now mismatched (2 != 4)
        let result = set.evaluate();
        assert!(
            verts_approx_eq(&result, &base, 1e-6),
            "mismatched-length target should be skipped, not panic or corrupt output"
        );
    }

    #[test]
    fn test_morph_target_set_total_weights() {
        let mut set = MorphTargetSet::new(make_verts(2, 0.0));
        let mut t1 = MorphTarget::new("t1", make_verts(2, 1.0));
        t1.weight = 0.3;
        let mut t2 = MorphTarget::new("t2", make_verts(2, 2.0));
        t2.weight = 0.5;
        set.add_target(t1).expect("add t1");
        set.add_target(t2).expect("add t2");
        assert!((set.total_weights() - 0.8).abs() < 1e-6);
    }

    // ── apply_morph_targets ──────────────────────────────────────────────────

    #[test]
    fn test_apply_morph_targets_known_delta() {
        let base = vec![[0.0_f32, 0.0, 0.0]; 2];
        let delta = vec![[1.0_f32, 2.0, 3.0]; 2];
        let mut t = MorphTarget::new("t", delta);
        t.weight = 0.5;
        let result = apply_morph_targets(&base, &[t]);
        assert!((result[0][0] - 0.5).abs() < 1e-6);
        assert!((result[0][1] - 1.0).abs() < 1e-6);
        assert!((result[0][2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_morph_targets_no_targets() {
        let base = make_verts(3, 7.0);
        let result = apply_morph_targets(&base, &[]);
        assert!(verts_approx_eq(&result, &base, 1e-6));
    }

    // ── compute_morph_delta ──────────────────────────────────────────────────

    #[test]
    fn test_compute_morph_delta_known() {
        let src = vec![[1.0_f32, 2.0, 3.0]];
        let tgt = vec![[4.0_f32, 6.0, 8.0]];
        let d = compute_morph_delta(&src, &tgt).expect("delta");
        assert!((d[0][0] - 3.0).abs() < 1e-6);
        assert!((d[0][1] - 4.0).abs() < 1e-6);
        assert!((d[0][2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_morph_delta_mismatch() {
        let src = make_verts(3, 0.0);
        let tgt = make_verts(5, 1.0);
        assert!(matches!(
            compute_morph_delta(&src, &tgt),
            Err(MorphError::VertexCountMismatch { .. })
        ));
    }

    // ── delta_magnitudes / max / mean ────────────────────────────────────────

    #[test]
    fn test_delta_magnitudes_known() {
        let d = vec![[3.0_f32, 4.0, 0.0]]; // magnitude = 5
        let mags = delta_magnitudes(&d);
        assert!((mags[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_delta_magnitude() {
        let d = vec![[1.0_f32, 0.0, 0.0], [3.0, 4.0, 0.0]]; // 1 and 5
        assert!((max_delta_magnitude(&d) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_delta_magnitude() {
        let d = vec![[1.0_f32, 0.0, 0.0], [0.0, 3.0, 4.0]]; // 1 and 5
        let m = mean_delta_magnitude(&d);
        assert!((m - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_delta_magnitude_empty() {
        assert_eq!(mean_delta_magnitude(&[]), 0.0);
    }

    // ── smooth_morph_sequence ────────────────────────────────────────────────

    #[test]
    fn test_smooth_morph_sequence_constant() {
        let seq: Vec<Vec<[f32; 3]>> = (0..5).map(|_| make_verts(3, 2.0)).collect();
        let smoothed = smooth_morph_sequence(&seq, 3).expect("smooth");
        for frame in &smoothed {
            assert!(verts_approx_eq(frame, &make_verts(3, 2.0), 1e-5));
        }
    }

    #[test]
    fn test_smooth_morph_sequence_window_1_unchanged() {
        let seq: Vec<Vec<[f32; 3]>> =
            vec![make_verts(2, 0.0), make_verts(2, 4.0), make_verts(2, 8.0)];
        let smoothed = smooth_morph_sequence(&seq, 1).expect("smooth w=1");
        assert!(verts_approx_eq(&smoothed[0], &seq[0], 1e-5));
        assert!(verts_approx_eq(&smoothed[1], &seq[1], 1e-5));
        assert!(verts_approx_eq(&smoothed[2], &seq[2], 1e-5));
    }

    #[test]
    fn test_smooth_morph_sequence_empty_err() {
        assert!(matches!(
            smooth_morph_sequence(&[], 1),
            Err(MorphError::EmptySequence)
        ));
    }

    #[test]
    fn test_smooth_morph_sequence_window_zero_err() {
        let seq = vec![make_verts(2, 0.0)];
        assert!(matches!(
            smooth_morph_sequence(&seq, 0),
            Err(MorphError::InvalidParam(_))
        ));
    }

    #[test]
    fn test_smooth_morph_sequence_window_too_large_err() {
        let seq = vec![make_verts(2, 0.0)];
        assert!(matches!(
            smooth_morph_sequence(&seq, 5),
            Err(MorphError::InvalidParam(_))
        ));
    }

    // ── resample_morph_sequence ──────────────────────────────────────────────

    #[test]
    fn test_resample_morph_sequence_2_to_4_endpoints() {
        let seq = vec![make_verts(2, 0.0), make_verts(2, 6.0)];
        let resampled = resample_morph_sequence(&seq, 4).expect("resample");
        assert_eq!(resampled.len(), 4);
        assert!(verts_approx_eq(&resampled[0], &make_verts(2, 0.0), 1e-5));
        assert!(verts_approx_eq(&resampled[3], &make_verts(2, 6.0), 1e-5));
    }

    #[test]
    fn test_resample_morph_sequence_same_count() {
        let seq: Vec<Vec<[f32; 3]>> =
            vec![make_verts(2, 1.0), make_verts(2, 3.0), make_verts(2, 5.0)];
        let resampled = resample_morph_sequence(&seq, 3).expect("resample same");
        assert_eq!(resampled.len(), 3);
        assert!(verts_approx_eq(&resampled[0], &seq[0], 1e-5));
        assert!(verts_approx_eq(&resampled[2], &seq[2], 1e-5));
    }

    #[test]
    fn test_resample_morph_sequence_to_1() {
        let seq = vec![make_verts(2, 5.0), make_verts(2, 10.0)];
        let resampled = resample_morph_sequence(&seq, 1).expect("resample 1");
        assert_eq!(resampled.len(), 1);
        assert!(verts_approx_eq(&resampled[0], &seq[0], 1e-5));
    }

    #[test]
    fn test_resample_morph_sequence_empty_err() {
        assert!(matches!(
            resample_morph_sequence(&[], 3),
            Err(MorphError::EmptySequence)
        ));
    }

    #[test]
    fn test_resample_morph_sequence_zero_frames_err() {
        let seq = vec![make_verts(2, 0.0)];
        assert!(matches!(
            resample_morph_sequence(&seq, 0),
            Err(MorphError::InvalidParam(_))
        ));
    }

    #[test]
    fn test_resample_morph_sequence_single_frame_expands_without_panicking() {
        // Upsampling a single keyframe to several frames is legal input; it
        // previously reached `(n - 2)` with `n == 1`, underflowing `usize`
        // (panicking in debug, or wrapping and then indexing out of bounds
        // in release).
        let seq = vec![make_verts(3, 4.0)];
        let resampled = resample_morph_sequence(&seq, 5).expect("resample single frame");
        assert_eq!(resampled.len(), 5);
        for frame in &resampled {
            assert!(verts_approx_eq(frame, &make_verts(3, 4.0), 1e-6));
        }
    }

    #[test]
    fn test_resample_morph_sequence_single_frame_to_single_frame() {
        let seq = vec![make_verts(2, 7.0)];
        let resampled = resample_morph_sequence(&seq, 1).expect("resample");
        assert_eq!(resampled.len(), 1);
        assert!(verts_approx_eq(&resampled[0], &seq[0], 1e-6));
    }

    #[test]
    fn test_resample_morph_sequence_mismatched_frame_lengths_errs() {
        let seq: Vec<Vec<[f32; 3]>> = vec![make_verts(3, 0.0), make_verts(5, 1.0)];
        assert!(matches!(
            resample_morph_sequence(&seq, 4),
            Err(MorphError::VertexCountMismatch { .. })
        ));
    }

    // ── compute_delta_sequence ───────────────────────────────────────────────

    #[test]
    fn test_compute_delta_sequence_first_is_zeros() {
        let seq = vec![make_verts(3, 2.0), make_verts(3, 5.0), make_verts(3, 7.0)];
        let deltas = compute_delta_sequence(&seq).expect("delta_seq");
        let zeros = make_verts(3, 0.0);
        assert!(verts_approx_eq(&deltas[0], &zeros, 1e-6));
    }

    #[test]
    fn test_compute_delta_sequence_second_frame() {
        let seq = vec![make_verts(2, 1.0), make_verts(2, 4.0)];
        let deltas = compute_delta_sequence(&seq).expect("delta_seq 2");
        let expected = make_verts(2, 3.0);
        assert!(verts_approx_eq(&deltas[1], &expected, 1e-5));
    }

    #[test]
    fn test_compute_delta_sequence_empty_err() {
        assert!(matches!(
            compute_delta_sequence(&[]),
            Err(MorphError::EmptySequence)
        ));
    }

    // ── MorphClip ────────────────────────────────────────────────────────────

    #[test]
    fn test_morph_clip_empty_err() {
        assert!(matches!(
            MorphClip::new("empty", vec![]),
            Err(MorphError::EmptySequence)
        ));
    }

    #[test]
    fn test_morph_clip_duration() {
        let kfs = vec![
            MorphKeyframe {
                time: 0.0,
                vertices: make_verts(3, 0.0),
            },
            MorphKeyframe {
                time: 2.0,
                vertices: make_verts(3, 1.0),
            },
        ];
        let clip = MorphClip::new("c", kfs).expect("clip");
        assert!((clip.duration() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_morph_clip_n_keyframes() {
        let kfs = vec![
            MorphKeyframe {
                time: 0.0,
                vertices: make_verts(2, 0.0),
            },
            MorphKeyframe {
                time: 1.0,
                vertices: make_verts(2, 1.0),
            },
            MorphKeyframe {
                time: 2.0,
                vertices: make_verts(2, 2.0),
            },
        ];
        let clip = MorphClip::new("c", kfs).expect("clip");
        assert_eq!(clip.n_keyframes(), 3);
    }

    #[test]
    fn test_morph_clip_sample_t0() {
        let kfs = vec![
            MorphKeyframe {
                time: 0.0,
                vertices: make_verts(3, 0.0),
            },
            MorphKeyframe {
                time: 1.0,
                vertices: make_verts(3, 5.0),
            },
        ];
        let clip = MorphClip::new("c", kfs).expect("clip");
        let r = clip
            .sample(0.0, &MorphInterpolation::Linear)
            .expect("sample t=0");
        assert!(verts_approx_eq(&r, &make_verts(3, 0.0), 1e-5));
    }

    #[test]
    fn test_morph_clip_sample_t_duration() {
        let kfs = vec![
            MorphKeyframe {
                time: 0.0,
                vertices: make_verts(3, 0.0),
            },
            MorphKeyframe {
                time: 1.0,
                vertices: make_verts(3, 5.0),
            },
        ];
        let clip = MorphClip::new("c", kfs).expect("clip");
        let r = clip
            .sample(1.0, &MorphInterpolation::Linear)
            .expect("sample t=dur");
        assert!(verts_approx_eq(&r, &make_verts(3, 5.0), 1e-5));
    }

    #[test]
    fn test_morph_clip_sample_loop_wrap() {
        let kfs = vec![
            MorphKeyframe {
                time: 0.0,
                vertices: make_verts(2, 0.0),
            },
            MorphKeyframe {
                time: 1.0,
                vertices: make_verts(2, 10.0),
            },
        ];
        let mut clip = MorphClip::new("c", kfs).expect("clip");
        clip.loop_mode = MorphLoopMode::Loop;
        // time = 0.5 wraps to 0.5 inside the clip
        let r0 = clip.sample(0.5, &MorphInterpolation::Linear).expect("s0.5");
        // time = 1.5 → 0.5 after wrap
        let r1 = clip.sample(1.5, &MorphInterpolation::Linear).expect("s1.5");
        assert!(verts_approx_eq(&r0, &r1, 1e-4));
    }

    #[test]
    fn test_morph_clip_sample_pingpong_bounce() {
        let kfs = vec![
            MorphKeyframe {
                time: 0.0,
                vertices: make_verts(2, 0.0),
            },
            MorphKeyframe {
                time: 2.0,
                vertices: make_verts(2, 10.0),
            },
        ];
        let mut clip = MorphClip::new("c", kfs).expect("clip");
        clip.loop_mode = MorphLoopMode::PingPong;
        // At t=0.5 (forward), sample should equal t=3.5 (period=4, 4-3.5=0.5)
        let forward = clip.sample(0.5, &MorphInterpolation::Linear).expect("fwd");
        let bounce = clip.sample(3.5, &MorphInterpolation::Linear).expect("bce");
        assert!(verts_approx_eq(&forward, &bounce, 1e-4));
    }

    #[test]
    fn test_morph_clip_sorted_on_creation() {
        // Keyframes provided out-of-order; should be sorted
        let kfs = vec![
            MorphKeyframe {
                time: 1.0,
                vertices: make_verts(2, 5.0),
            },
            MorphKeyframe {
                time: 0.0,
                vertices: make_verts(2, 0.0),
            },
        ];
        let clip = MorphClip::new("c", kfs).expect("clip");
        assert!((clip.keyframes[0].time - 0.0).abs() < 1e-6);
        assert!((clip.keyframes[1].time - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_morph_clip_single_keyframe() {
        let kfs = vec![MorphKeyframe {
            time: 0.5,
            vertices: make_verts(3, 7.0),
        }];
        let clip = MorphClip::new("c", kfs).expect("clip");
        let r = clip
            .sample(0.0, &MorphInterpolation::Linear)
            .expect("single kf");
        assert!(verts_approx_eq(&r, &make_verts(3, 7.0), 1e-5));
    }

    // ── format_morph_clip ────────────────────────────────────────────────────

    #[test]
    fn test_format_morph_clip_non_empty() {
        let kfs = vec![
            MorphKeyframe {
                time: 0.0,
                vertices: make_verts(5, 0.0),
            },
            MorphKeyframe {
                time: 1.0,
                vertices: make_verts(5, 1.0),
            },
        ];
        let clip = MorphClip::new("test_clip", kfs).expect("clip");
        let s = format_morph_clip(&clip);
        assert!(!s.is_empty());
        assert!(s.contains("test_clip"));
    }

    #[test]
    fn test_format_morph_clip_loop_modes() {
        let make = |mode: MorphLoopMode| {
            let kfs = vec![
                MorphKeyframe {
                    time: 0.0,
                    vertices: make_verts(1, 0.0),
                },
                MorphKeyframe {
                    time: 1.0,
                    vertices: make_verts(1, 1.0),
                },
            ];
            let mut c = MorphClip::new("c", kfs).expect("clip");
            c.loop_mode = mode;
            c
        };
        assert!(format_morph_clip(&make(MorphLoopMode::Once)).contains("once"));
        assert!(format_morph_clip(&make(MorphLoopMode::Loop)).contains("loop"));
        assert!(format_morph_clip(&make(MorphLoopMode::PingPong)).contains("ping-pong"));
    }

    // ── error display ────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_vertex_count_mismatch() {
        let e = MorphError::VertexCountMismatch { a: 3, b: 5 };
        assert!(e.to_string().contains('3') && e.to_string().contains('5'));
    }

    #[test]
    fn test_error_display_invalid_blend_weight() {
        let e = MorphError::InvalidBlendWeight(1.5);
        assert!(e.to_string().contains("1.5"));
    }

    #[test]
    fn test_error_display_weight_sum() {
        let e = MorphError::WeightSumError {
            sum: 0.7,
            tol: 1e-4,
        };
        assert!(e.to_string().contains("0.7"));
    }

    #[test]
    fn test_error_display_empty_sequence() {
        let e = MorphError::EmptySequence;
        assert!(!e.to_string().is_empty());
    }
}
