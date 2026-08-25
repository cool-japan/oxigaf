//! # Emotion Recognition from FLAME Expression Parameters
//!
//! This module provides rule-based emotion classification from FLAME facial expression
//! parameters. It maps expression PCA coefficients to the 8 basic emotions (Ekman + Contempt),
//! computes arousal-valence coordinates (Russell's circumplex model), and supports trajectory
//! analysis over sequences.
//!
//! ## Approach
//!
//! Rather than a trained ML model, a geometric rule-based approach is used:
//! 1. Project the first few expression PCA coefficients onto hand-tuned arousal/valence axes.
//! 2. Compute L1 distance from each emotion's canonical circumplex position.
//! 3. Apply temperature-scaled softmax to distances to produce a probability distribution.
//!
//! This is fast, deterministic, and works without any training data.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────

/// Errors that can occur during emotion recognition.
#[derive(Debug, Error)]
pub enum EmotionError {
    /// The expression parameter vector was completely empty.
    #[error("Empty expression parameters")]
    EmptyParams,

    /// The expression parameter vector was shorter than the required minimum.
    #[error("Expression vector length {len} is too short (need at least {min})")]
    ParamsTooShort { len: usize, min: usize },

    /// Confidence values did not sum to 1.0 within tolerance.
    #[error("Confidence values must sum to 1.0, got {sum:.4}")]
    InvalidConfidenceSum { sum: f32 },

    /// Blend weights did not sum to 1.0 within tolerance.
    #[error("Blend weights do not sum to 1.0, got {sum:.4}")]
    InvalidBlendWeights { sum: f32 },

    /// An emotion index was out of the valid range.
    #[error("Emotion index {idx} out of range (max {max})")]
    EmotionIndexOutOfRange { idx: usize, max: usize },
}

// ─────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────

/// The 8 basic emotions (Ekman's six universal emotions + Contempt + Neutral).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasicEmotion {
    /// Neutral / resting face with no discernible expression.
    Neutral,
    /// Happiness — raised cheeks, lip corner pull.
    Happy,
    /// Sadness — inner brow raise, lip corner depression.
    Sad,
    /// Anger — brow lowerer, upper lid raise, lip corner depression.
    Angry,
    /// Surprise — brow raise, jaw drop.
    Surprised,
    /// Fear — brow raise + inner corner, lip stretch.
    Fearful,
    /// Disgust — nose wrinkle, upper lip raise.
    Disgusted,
    /// Contempt — unilateral lip corner raise.
    Contempt,
}

/// A single emotion with its associated confidence score.
#[derive(Debug, Clone)]
pub struct EmotionScore {
    /// The classified emotion.
    pub emotion: BasicEmotion,
    /// Confidence in [0, 1]; all 8 scores sum to 1.0.
    pub confidence: f32,
}

/// A point in Russell's 2D circumplex emotion space.
#[derive(Debug, Clone, Copy)]
pub struct ArousalValence {
    /// Arousal axis in [-1, 1]: calm (−) to excited (+).
    pub arousal: f32,
    /// Valence axis in [-1, 1]: negative affect (−) to positive affect (+).
    pub valence: f32,
}

/// Configuration for the emotion recogniser.
#[derive(Debug, Clone)]
pub struct EmotionConfig {
    /// Number of leading expression PCA coefficients to use (default: 20).
    pub n_params: usize,
    /// Temperature for softmax scoring; lower → more peaked (default: 0.5).
    pub softmax_temperature: f32,
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            n_params: 20,
            softmax_temperature: 0.5,
        }
    }
}

/// Full result of an emotion recognition pass.
#[derive(Debug, Clone)]
pub struct EmotionResult {
    /// All 8 emotions sorted by confidence (highest first).
    pub scores: Vec<EmotionScore>,
    /// The emotion with the highest confidence.
    pub dominant: BasicEmotion,
    /// Confidence of the dominant emotion in [0, 1].
    pub dominant_confidence: f32,
    /// Position in the arousal-valence circumplex.
    pub arousal_valence: ArousalValence,
    /// Overall expression intensity in [0, 1].
    pub intensity: f32,
}

/// Temporal record of emotions across a sequence of frames.
#[derive(Debug, Clone)]
pub struct EmotionTrajectory {
    /// Dominant emotion for each frame.
    pub emotions: Vec<BasicEmotion>,
    /// Confidence of the dominant emotion for each frame.
    pub confidences: Vec<f32>,
    /// Arousal-valence coordinates for each frame.
    pub av_history: Vec<ArousalValence>,
}

// ─────────────────────────────────────────────────────────────────
// BasicEmotion helpers
// ─────────────────────────────────────────────────────────────────

impl BasicEmotion {
    /// Returns an array of all 8 basic emotions in a stable order.
    #[inline]
    #[must_use]
    pub fn all() -> [Self; 8] {
        [
            Self::Neutral,
            Self::Happy,
            Self::Sad,
            Self::Angry,
            Self::Surprised,
            Self::Fearful,
            Self::Disgusted,
            Self::Contempt,
        ]
    }

    /// Returns the human-readable name of the emotion.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Neutral => "Neutral",
            Self::Happy => "Happy",
            Self::Sad => "Sad",
            Self::Angry => "Angry",
            Self::Surprised => "Surprised",
            Self::Fearful => "Fearful",
            Self::Disgusted => "Disgusted",
            Self::Contempt => "Contempt",
        }
    }

    /// Returns the canonical (arousal, valence) position in Russell's circumplex model.
    ///
    /// These are hand-tuned approximate targets:
    /// - arousal in [-1, 1]: calm (−) → excited (+)
    /// - valence in [-1, 1]: negative (−) → positive (+)
    #[inline]
    #[must_use]
    pub fn arousal_valence_target(&self) -> (f32, f32) {
        match self {
            Self::Neutral => (0.0, 0.0),
            Self::Happy => (0.5, 0.8),
            Self::Sad => (-0.5, -0.7),
            Self::Angry => (0.7, -0.6),
            Self::Surprised => (0.6, 0.2),
            Self::Fearful => (0.6, -0.5),
            Self::Disgusted => (0.2, -0.8),
            Self::Contempt => (-0.1, -0.4),
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Private utilities
// ─────────────────────────────────────────────────────────────────

/// Numerically stable softmax with temperature scaling.
///
/// Returns an empty vec when `values` is empty rather than panicking.
fn softmax(values: &[f32], temperature: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let t = if temperature.abs() < 1e-9 {
        1e-9
    } else {
        temperature
    };
    let scaled: Vec<f32> = values.iter().map(|v| v / t).collect();
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    if sum < 1e-12 {
        // Uniform fallback to avoid division by zero
        let uniform = 1.0 / exp_vals.len() as f32;
        return vec![uniform; exp_vals.len()];
    }
    exp_vals.iter().map(|v| v / sum).collect()
}

// ─────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────

/// Compute the overall expression intensity as the normalised RMS of the leading coefficients.
///
/// Uses the first `min(config.n_params, params.len())` coefficients. Result is
/// clamped to [0, 1] by dividing by 3.0 (typical maximum absolute value for
/// FLAME PCA coefficients).
///
/// # Not monotonic in `n_params`
///
/// Because this is a root-mean-*square*, widening the window can *lower* the
/// result: the extra coefficients enlarge the divisor `n` whether or not they
/// contribute to the numerator. With `params = [3, 0, 0, …]`, `n_params = 1`
/// reports `1.0` while `n_params = 8` reports `sqrt(9/8)/3 ≈ 0.354`. That is
/// the intended reading of an RMS — "how strong is the average leading
/// coefficient", not "how much total energy is there" — so do not assume
/// `intensity(n) <= intensity(n + k)`. Use a fixed `n_params` when comparing
/// intensities across frames or subjects.
///
/// # Errors
///
/// Returns [`EmotionError::EmptyParams`] when `params` is empty.
pub fn compute_expression_intensity(
    params: &[f32],
    config: &EmotionConfig,
) -> Result<f32, EmotionError> {
    if params.is_empty() {
        return Err(EmotionError::EmptyParams);
    }
    let n = params.len().min(config.n_params);
    if n == 0 {
        // config.n_params == 0: no coefficients to consider.
        return Ok(0.0);
    }
    let rms = (params[..n].iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
    // Normalise: typical max |coeff| for FLAME PCA is ~3.0
    let normalised = (rms / 3.0).clamp(0.0, 1.0);
    Ok(normalised)
}

/// Safely index a slice, returning 0.0 when `i` is out of bounds for
/// `params` OR `i >= limit`.
#[inline]
fn param_at(params: &[f32], i: usize, limit: usize) -> f32 {
    if i >= limit {
        return 0.0;
    }
    params.get(i).copied().unwrap_or(0.0)
}

/// Project expression PCA coefficients onto the arousal-valence circumplex axes.
///
/// Uses a simple linear combination of 6 expression parameters (indices 0
/// and 2 and 4 for arousal; 1, 3 and 5 for valence) with hand-tuned weights
/// of 0.5/0.3/0.2 each. Real systems would fit these from labelled data.
/// Any of these 6 indices at or beyond `config.n_params` is treated as
/// absent (contributes 0), so setting `config.n_params < 6` progressively
/// drops the higher-indexed terms.
///
/// # Errors
///
/// Returns [`EmotionError::EmptyParams`] when `params` is empty, or
/// [`EmotionError::ParamsTooShort`] when fewer than 2 parameters are provided.
pub fn compute_arousal_valence(
    params: &[f32],
    config: &EmotionConfig,
) -> Result<ArousalValence, EmotionError> {
    if params.is_empty() {
        return Err(EmotionError::EmptyParams);
    }
    if params.len() < 2 {
        return Err(EmotionError::ParamsTooShort {
            len: params.len(),
            min: 2,
        });
    }

    let limit = config.n_params;
    let arousal = param_at(params, 0, limit) * 0.5
        + param_at(params, 2, limit) * 0.3
        + param_at(params, 4, limit) * 0.2;
    let valence = param_at(params, 1, limit) * 0.5
        + param_at(params, 3, limit) * 0.3
        + param_at(params, 5, limit) * 0.2;

    Ok(ArousalValence {
        arousal: arousal.clamp(-1.0, 1.0),
        valence: valence.clamp(-1.0, 1.0),
    })
}

/// Compute confidence scores for all 8 emotions given expression PCA coefficients.
///
/// Algorithm:
/// 1. Project params to arousal-valence.
/// 2. For each emotion, compute L1 distance to its canonical circumplex target.
/// 3. Convert distance to similarity: `1.0 / (1.0 + distance)`.
/// 4. Apply temperature-scaled softmax.
/// 5. Return scores sorted by confidence (highest first).
///
/// # Errors
///
/// Propagates errors from [`compute_arousal_valence`].
pub fn compute_emotion_scores(
    params: &[f32],
    config: &EmotionConfig,
) -> Result<Vec<EmotionScore>, EmotionError> {
    let av = compute_arousal_valence(params, config)?;

    let emotions = BasicEmotion::all();
    let similarities: Vec<f32> = emotions
        .iter()
        .map(|e| {
            let (at, vt) = e.arousal_valence_target();
            let dist = (av.arousal - at).abs() + (av.valence - vt).abs();
            1.0 / (1.0 + dist)
        })
        .collect();

    let probs = softmax(&similarities, config.softmax_temperature);

    let mut scores: Vec<EmotionScore> = emotions
        .iter()
        .zip(probs.iter())
        .map(|(&emotion, &confidence)| EmotionScore {
            emotion,
            confidence,
        })
        .collect();

    scores.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(scores)
}

/// Recognise the dominant emotion from FLAME expression PCA coefficients.
///
/// # Errors
///
/// Returns [`EmotionError::ParamsTooShort`] when fewer than 2 parameters are provided,
/// or [`EmotionError::EmptyParams`] when the slice is empty.
pub fn recognize_emotion(
    params: &[f32],
    config: &EmotionConfig,
) -> Result<EmotionResult, EmotionError> {
    if params.is_empty() {
        return Err(EmotionError::EmptyParams);
    }
    if params.len() < 2 {
        return Err(EmotionError::ParamsTooShort {
            len: params.len(),
            min: 2,
        });
    }

    let intensity = compute_expression_intensity(params, config)?;
    let arousal_valence = compute_arousal_valence(params, config)?;
    let scores = compute_emotion_scores(params, config)?;

    // scores is sorted descending; first entry is dominant
    let dominant = scores.first().ok_or(EmotionError::EmptyParams)?.emotion;
    let dominant_confidence = scores.first().ok_or(EmotionError::EmptyParams)?.confidence;

    Ok(EmotionResult {
        scores,
        dominant,
        dominant_confidence,
        arousal_valence,
        intensity,
    })
}

/// Blend multiple sets of expression parameters using per-vector weights.
///
/// Each entry in `target_emotions` is `(expression_params, weight)`. The weights
/// must sum to approximately 1.0 (within 0.01 tolerance) and all parameter vectors
/// must have the same length.
///
/// # Errors
///
/// - [`EmotionError::InvalidBlendWeights`] if weights do not sum to ~1.0.
/// - [`EmotionError::EmptyParams`] if the input list is empty.
pub fn blend_emotion_params(target_emotions: &[(Vec<f32>, f32)]) -> Result<Vec<f32>, EmotionError> {
    if target_emotions.is_empty() {
        return Err(EmotionError::EmptyParams);
    }

    let weight_sum: f32 = target_emotions.iter().map(|(_, w)| w).sum();
    if (weight_sum - 1.0).abs() > 0.01 {
        return Err(EmotionError::InvalidBlendWeights { sum: weight_sum });
    }

    // Determine output length from first vector
    let out_len = target_emotions[0].0.len();

    let mut result = vec![0.0_f32; out_len];
    for (params, weight) in target_emotions {
        // If a shorter vector is provided, blend only its elements; the rest remain 0.
        let effective_len = params.len().min(out_len);
        for i in 0..effective_len {
            result[i] += params[i] * weight;
        }
    }

    Ok(result)
}

/// Run emotion recognition on each frame in a sequence of expression parameter vectors.
///
/// # Errors
///
/// Propagates any [`EmotionError`] from frame-level recognition.
pub fn emotion_trajectory(
    frames: &[Vec<f32>],
    config: &EmotionConfig,
) -> Result<EmotionTrajectory, EmotionError> {
    if frames.is_empty() {
        return Ok(EmotionTrajectory {
            emotions: Vec::new(),
            confidences: Vec::new(),
            av_history: Vec::new(),
        });
    }

    let mut emotions = Vec::with_capacity(frames.len());
    let mut confidences = Vec::with_capacity(frames.len());
    let mut av_history = Vec::with_capacity(frames.len());

    for frame in frames {
        let result = recognize_emotion(frame, config)?;
        emotions.push(result.dominant);
        confidences.push(result.dominant_confidence);
        av_history.push(result.arousal_valence);
    }

    Ok(EmotionTrajectory {
        emotions,
        confidences,
        av_history,
    })
}

/// Smooth an emotion trajectory:
/// - Dominant emotions are majority-voted over a sliding window of up to 5 frames.
/// - Confidences are smoothed with exponential moving average (EMA, decay = 0.7).
///
/// The returned `emotions`/`confidences` share a length equal to
/// `min(trajectory.emotions.len(), trajectory.confidences.len())`.
/// `EmotionTrajectory`'s fields are all public and independently writable,
/// so the two are not guaranteed to already have matching lengths; only
/// their common prefix can be smoothed safely. `av_history` is always
/// copied through unchanged.
#[must_use]
pub fn smooth_emotion_trajectory(trajectory: &EmotionTrajectory) -> EmotionTrajectory {
    const WINDOW: usize = 5;
    const HALF: usize = WINDOW / 2;
    const EMA_DECAY: f32 = 0.7;

    let n = trajectory.emotions.len().min(trajectory.confidences.len());
    if n == 0 {
        return EmotionTrajectory {
            emotions: Vec::new(),
            confidences: Vec::new(),
            av_history: trajectory.av_history.clone(),
        };
    }

    // Majority-vote dominant emotion
    let mut smoothed_emotions = Vec::with_capacity(n);
    for i in 0..n {
        let start = i.saturating_sub(HALF);
        let end = (i + HALF + 1).min(n);
        let window = &trajectory.emotions[start..end];

        // Count occurrences of each emotion in the window
        let mut counts = [0usize; 8];
        for &e in window {
            counts[emotion_index(e)] += 1;
        }

        // Find the most frequent; ties keep the original emotion
        let mut best_idx = emotion_index(trajectory.emotions[i]);
        let mut best_count = counts[best_idx];
        for (idx, &count) in counts.iter().enumerate() {
            if count > best_count {
                best_count = count;
                best_idx = idx;
            }
        }
        smoothed_emotions.push(emotion_from_index(best_idx));
    }

    // EMA smooth confidences (forward pass)
    let mut smoothed_confidences = Vec::with_capacity(n);
    let mut ema = trajectory.confidences[0];
    smoothed_confidences.push(ema);
    for &c in &trajectory.confidences[1..n] {
        ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * c;
        smoothed_confidences.push(ema);
    }

    EmotionTrajectory {
        emotions: smoothed_emotions,
        confidences: smoothed_confidences,
        av_history: trajectory.av_history.clone(),
    }
}

/// Compute the rate at which the dominant emotion changes across frames.
///
/// Returns transitions / (`n_frames` − 1), or 0.0 for trajectories with fewer than 2 frames.
#[must_use]
pub fn compute_emotion_transition_rate(trajectory: &EmotionTrajectory) -> f32 {
    let n = trajectory.emotions.len();
    if n < 2 {
        return 0.0;
    }
    let transitions = trajectory
        .emotions
        .windows(2)
        .filter(|w| w[0] != w[1])
        .count();
    transitions as f32 / (n - 1) as f32
}

/// Format an [`EmotionResult`] into a concise human-readable string.
#[must_use]
pub fn format_emotion_result(result: &EmotionResult) -> String {
    format!(
        "Dominant: {} ({:.2}), A/V: ({:.2}, {:.2}), Intensity: {:.2}",
        result.dominant.name(),
        result.dominant_confidence,
        result.arousal_valence.arousal,
        result.arousal_valence.valence,
        result.intensity
    )
}

/// Find the most frequently occurring dominant emotion in a frame window [start, end).
///
/// # Errors
///
/// Returns [`EmotionError::EmotionIndexOutOfRange`] when `start >= end` or `end > trajectory.len()`.
pub fn dominant_in_window(
    trajectory: &EmotionTrajectory,
    start: usize,
    end: usize,
) -> Result<BasicEmotion, EmotionError> {
    let n = trajectory.emotions.len();
    if start >= end || end > n {
        return Err(EmotionError::EmotionIndexOutOfRange { idx: end, max: n });
    }

    let mut counts = [0usize; 8];
    for &e in &trajectory.emotions[start..end] {
        counts[emotion_index(e)] += 1;
    }

    let (best_idx, _) = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .ok_or(EmotionError::EmptyParams)?;

    Ok(emotion_from_index(best_idx))
}

// ─────────────────────────────────────────────────────────────────
// Private index helpers
// ─────────────────────────────────────────────────────────────────

/// Map an emotion to its stable index in `BasicEmotion::all()`.
#[inline]
fn emotion_index(e: BasicEmotion) -> usize {
    match e {
        BasicEmotion::Neutral => 0,
        BasicEmotion::Happy => 1,
        BasicEmotion::Sad => 2,
        BasicEmotion::Angry => 3,
        BasicEmotion::Surprised => 4,
        BasicEmotion::Fearful => 5,
        BasicEmotion::Disgusted => 6,
        BasicEmotion::Contempt => 7,
    }
}

/// Map an index back to a `BasicEmotion` (saturating to `Neutral` if out of range).
#[inline]
fn emotion_from_index(idx: usize) -> BasicEmotion {
    match idx {
        1 => BasicEmotion::Happy,
        2 => BasicEmotion::Sad,
        3 => BasicEmotion::Angry,
        4 => BasicEmotion::Surprised,
        5 => BasicEmotion::Fearful,
        6 => BasicEmotion::Disgusted,
        7 => BasicEmotion::Contempt,
        _ => BasicEmotion::Neutral,
    }
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    // ── BasicEmotion ────────────────────────────────────────────

    #[test]
    fn all_returns_eight_emotions() {
        assert_eq!(BasicEmotion::all().len(), 8);
    }

    #[test]
    fn all_emotions_are_unique() {
        let all = BasicEmotion::all();
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "Duplicate at indices {i} and {j}");
            }
        }
    }

    #[test]
    fn name_non_empty_for_all() {
        for e in BasicEmotion::all() {
            assert!(!e.name().is_empty(), "{e:?} name is empty");
        }
    }

    #[test]
    fn name_correct_for_known_emotions() {
        assert_eq!(BasicEmotion::Happy.name(), "Happy");
        assert_eq!(BasicEmotion::Sad.name(), "Sad");
        assert_eq!(BasicEmotion::Neutral.name(), "Neutral");
        assert_eq!(BasicEmotion::Contempt.name(), "Contempt");
    }

    #[test]
    fn happy_has_positive_valence() {
        let (_, valence) = BasicEmotion::Happy.arousal_valence_target();
        assert!(
            valence > 0.0,
            "Happy should have positive valence, got {valence}"
        );
    }

    #[test]
    fn sad_has_negative_valence() {
        let (_, valence) = BasicEmotion::Sad.arousal_valence_target();
        assert!(
            valence < 0.0,
            "Sad should have negative valence, got {valence}"
        );
    }

    #[test]
    fn neutral_is_near_origin() {
        let (arousal, valence) = BasicEmotion::Neutral.arousal_valence_target();
        assert_abs_diff_eq!(arousal, 0.0, epsilon = 0.01);
        assert_abs_diff_eq!(valence, 0.0, epsilon = 0.01);
    }

    #[test]
    fn angry_has_high_arousal_negative_valence() {
        let (arousal, valence) = BasicEmotion::Angry.arousal_valence_target();
        assert!(arousal > 0.0, "Angry should have positive arousal");
        assert!(valence < 0.0, "Angry should have negative valence");
    }

    // ── compute_expression_intensity ────────────────────────────

    #[test]
    fn intensity_zero_for_zero_params() {
        let params = vec![0.0_f32; 30];
        let intensity = compute_expression_intensity(&params, &EmotionConfig::default()).unwrap();
        assert_abs_diff_eq!(intensity, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn intensity_large_values_clamped_to_one() {
        let params = vec![100.0_f32; 20];
        let intensity = compute_expression_intensity(&params, &EmotionConfig::default()).unwrap();
        assert_abs_diff_eq!(intensity, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn intensity_empty_params_errors() {
        let result = compute_expression_intensity(&[], &EmotionConfig::default());
        assert!(matches!(result, Err(EmotionError::EmptyParams)));
    }

    #[test]
    fn intensity_single_param_succeeds() {
        let params = vec![1.5_f32];
        let intensity = compute_expression_intensity(&params, &EmotionConfig::default()).unwrap();
        // RMS = 1.5, normalised = 1.5/3.0 = 0.5
        assert_relative_eq!(intensity, 0.5, epsilon = 1e-5);
    }

    #[test]
    fn intensity_in_unit_interval() {
        for mag in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0] {
            let params = vec![mag; 20];
            let intensity =
                compute_expression_intensity(&params, &EmotionConfig::default()).unwrap();
            assert!(
                (0.0..=1.0).contains(&intensity),
                "intensity {intensity} out of [0,1] for mag={mag}"
            );
        }
    }

    #[test]
    fn intensity_respects_config_n_params() {
        // Regression test: `EmotionConfig::n_params` previously had zero
        // effect on `compute_expression_intensity`.
        let mut params = vec![0.0_f32; 20];
        params[0] = 3.0; // at the clamp boundary: |3.0|/3.0 = 1.0
        for v in params.iter_mut().skip(10) {
            *v = 3.0; // large values, but past a restrictive n_params
        }

        let restrictive = EmotionConfig {
            n_params: 1,
            ..EmotionConfig::default()
        };
        let intensity = compute_expression_intensity(&params, &restrictive).unwrap();
        // Only params[0] = 3.0 is considered: RMS = 3.0, normalised = 1.0.
        assert_abs_diff_eq!(intensity, 1.0, epsilon = 1e-6);

        let permissive = EmotionConfig {
            n_params: 20,
            ..EmotionConfig::default()
        };
        let intensity_full = compute_expression_intensity(&params, &permissive).unwrap();
        // Widening the window changes the answer, which is the whole point of
        // the regression: `n_params` is genuinely consulted. The exact value
        // is the closed form of the documented metric — 11 entries of 3.0
        // (index 0 plus indices 10..20) averaged over the full 20-wide
        // window: sqrt(11 * 9 / 20) / 3 = sqrt(0.55).
        let expected_full = 0.55_f32.sqrt();
        assert_abs_diff_eq!(intensity_full, expected_full, epsilon = 1e-6);
        assert!(
            (intensity_full - intensity).abs() > 1e-3,
            "n_params must change the result: {intensity_full} vs {intensity}"
        );
    }

    #[test]
    fn intensity_is_not_monotonic_in_n_params() {
        // `compute_expression_intensity` is a *root-mean-square* over the
        // first `n` coefficients, so it is deliberately NOT monotonic in
        // `n_params`: extending the window over near-zero coefficients
        // divides by a larger `n` without adding to the numerator, which
        // *lowers* the reported intensity.
        //
        // This is the documented behaviour of an RMS ("how strong is the
        // average leading coefficient"), not a bug — but it is unintuitive
        // enough that an earlier version of `intensity_respects_config_n_params`
        // asserted the opposite (`intensity_full >= intensity`) and failed.
        // Pin the real property down so that mistake cannot recur.
        let mut params = vec![0.0_f32; 8];
        params[0] = 3.0; // one strong coefficient, the rest silent

        let narrow = compute_expression_intensity(
            &params,
            &EmotionConfig {
                n_params: 1,
                ..EmotionConfig::default()
            },
        )
        .unwrap();
        let wide = compute_expression_intensity(
            &params,
            &EmotionConfig {
                n_params: 8,
                ..EmotionConfig::default()
            },
        )
        .unwrap();

        assert_abs_diff_eq!(narrow, 1.0, epsilon = 1e-6);
        // sqrt(9 / 8) / 3 = sqrt(1/8)
        assert_abs_diff_eq!(wide, 0.125_f32.sqrt(), epsilon = 1e-6);
        assert!(
            wide < narrow,
            "padding the window with zeros must dilute the RMS: {wide} vs {narrow}"
        );
    }

    #[test]
    fn intensity_zero_n_params_is_zero_not_nan() {
        let params = vec![5.0_f32; 10];
        let config = EmotionConfig {
            n_params: 0,
            ..EmotionConfig::default()
        };
        let intensity = compute_expression_intensity(&params, &config).unwrap();
        assert_abs_diff_eq!(intensity, 0.0, epsilon = 1e-6);
    }

    // ── compute_arousal_valence ──────────────────────────────────

    #[test]
    fn arousal_valence_zeros_gives_origin() {
        let params = vec![0.0_f32; 10];
        let av = compute_arousal_valence(&params, &EmotionConfig::default()).unwrap();
        assert_abs_diff_eq!(av.arousal, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(av.valence, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn arousal_valence_empty_errors() {
        let result = compute_arousal_valence(&[], &EmotionConfig::default());
        assert!(matches!(result, Err(EmotionError::EmptyParams)));
    }

    #[test]
    fn arousal_valence_single_param_errors() {
        let result = compute_arousal_valence(&[1.0], &EmotionConfig::default());
        assert!(matches!(result, Err(EmotionError::ParamsTooShort { .. })));
    }

    #[test]
    fn arousal_valence_clamped_to_minus_one_one() {
        let params = vec![10.0_f32; 10];
        let av = compute_arousal_valence(&params, &EmotionConfig::default()).unwrap();
        assert!(av.arousal >= -1.0 && av.arousal <= 1.0);
        assert!(av.valence >= -1.0 && av.valence <= 1.0);
    }

    #[test]
    fn arousal_valence_respects_config_n_params() {
        // Regression test: `EmotionConfig::n_params` previously had zero
        // effect on `compute_arousal_valence`. With n_params=1, only index 0
        // (arousal weight 0.5) may contribute; index 2 and 4 must be dropped.
        let mut params = vec![0.0_f32; 10];
        params[0] = 1.0;
        params[2] = 1.0;
        params[4] = 1.0;
        let restrictive = EmotionConfig {
            n_params: 1,
            ..EmotionConfig::default()
        };
        let av = compute_arousal_valence(&params, &restrictive).unwrap();
        assert_abs_diff_eq!(av.arousal, 0.5, epsilon = 1e-6);

        let permissive = EmotionConfig {
            n_params: 10,
            ..EmotionConfig::default()
        };
        let av_full = compute_arousal_valence(&params, &permissive).unwrap();
        // All three terms (0.5 + 0.3 + 0.2) contribute now.
        assert_abs_diff_eq!(av_full.arousal, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn arousal_valence_uses_first_params() {
        // params[0] = 2.0, all others 0 → arousal = 2.0*0.5 = 1.0 (clamped), valence = 0
        let mut params = vec![0.0_f32; 10];
        params[0] = 2.0;
        let av = compute_arousal_valence(&params, &EmotionConfig::default()).unwrap();
        assert!(
            av.arousal > 0.0,
            "Expected positive arousal from params[0]=2.0"
        );
        assert_abs_diff_eq!(av.valence, 0.0, epsilon = 1e-6);
    }

    // ── softmax (via compute_emotion_scores) ─────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let vals = vec![1.0_f32, 2.0, 0.5, 3.0];
        let out = softmax(&vals, 1.0);
        let sum: f32 = out.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn softmax_empty_returns_empty() {
        let out = softmax(&[], 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn softmax_low_temperature_peaks_at_max() {
        let vals = vec![0.0_f32, 10.0, 0.0];
        let out = softmax(&vals, 0.01);
        // With very low temperature, practically all probability at index 1
        assert!(out[1] > 0.99, "Expected near-1 at max, got {}", out[1]);
    }

    #[test]
    fn softmax_high_temperature_is_more_uniform() {
        let vals = vec![1.0_f32, 2.0, 3.0];
        let low_temp = softmax(&vals, 0.01);
        let high_temp = softmax(&vals, 10.0);
        // High temperature → min is larger than with low temperature
        let min_low: f32 = low_temp.iter().copied().fold(f32::INFINITY, f32::min);
        let min_high: f32 = high_temp.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(min_high > min_low, "High temp should be more uniform");
    }

    // ── compute_emotion_scores ───────────────────────────────────

    #[test]
    fn emotion_scores_returns_eight_entries() {
        let params = vec![0.5_f32; 20];
        let config = EmotionConfig::default();
        let scores = compute_emotion_scores(&params, &config).unwrap();
        assert_eq!(scores.len(), 8);
    }

    #[test]
    fn emotion_scores_sum_to_one() {
        let params = vec![0.3_f32; 20];
        let config = EmotionConfig::default();
        let scores = compute_emotion_scores(&params, &config).unwrap();
        let sum: f32 = scores.iter().map(|s| s.confidence).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn emotion_scores_sorted_descending() {
        let params = vec![0.3_f32; 20];
        let config = EmotionConfig::default();
        let scores = compute_emotion_scores(&params, &config).unwrap();
        for w in scores.windows(2) {
            assert!(
                w[0].confidence >= w[1].confidence,
                "Scores not sorted: {} < {}",
                w[0].confidence,
                w[1].confidence
            );
        }
    }

    #[test]
    fn emotion_scores_confidences_in_unit_interval() {
        let params = vec![1.0_f32; 20];
        let config = EmotionConfig::default();
        let scores = compute_emotion_scores(&params, &config).unwrap();
        for s in &scores {
            assert!(
                (0.0..=1.0).contains(&s.confidence),
                "Confidence {} out of [0,1]",
                s.confidence
            );
        }
    }

    // ── recognize_emotion ────────────────────────────────────────

    #[test]
    fn recognize_smoke_test() {
        let params = vec![0.5_f32; 20];
        let config = EmotionConfig::default();
        let result = recognize_emotion(&params, &config);
        assert!(result.is_ok(), "Unexpected error: {:?}", result.err());
    }

    #[test]
    fn recognize_empty_params_errors() {
        let config = EmotionConfig::default();
        let result = recognize_emotion(&[], &config);
        assert!(matches!(result, Err(EmotionError::EmptyParams)));
    }

    #[test]
    fn recognize_single_param_errors() {
        let config = EmotionConfig::default();
        let result = recognize_emotion(&[1.0], &config);
        assert!(matches!(result, Err(EmotionError::ParamsTooShort { .. })));
    }

    #[test]
    fn recognize_intensity_in_unit_interval() {
        let params = vec![1.0_f32; 20];
        let config = EmotionConfig::default();
        let result = recognize_emotion(&params, &config).unwrap();
        assert!(
            (0.0..=1.0).contains(&result.intensity),
            "Intensity {} out of [0,1]",
            result.intensity
        );
    }

    #[test]
    fn recognize_dominant_matches_first_score() {
        let params = vec![0.7_f32, -0.8, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = EmotionConfig::default();
        let result = recognize_emotion(&params, &config).unwrap();
        assert_eq!(result.dominant, result.scores[0].emotion);
        assert_relative_eq!(
            result.dominant_confidence,
            result.scores[0].confidence,
            epsilon = 1e-6
        );
    }

    // ── blend_emotion_params ─────────────────────────────────────

    #[test]
    fn blend_identity_single_vector() {
        let params = vec![1.0_f32, 2.0, 3.0];
        let targets = vec![(params.clone(), 1.0_f32)];
        let blended = blend_emotion_params(&targets).unwrap();
        assert_eq!(blended, params);
    }

    #[test]
    fn blend_invalid_weights_errors() {
        let targets = vec![(vec![1.0_f32, 2.0], 0.3), (vec![1.0_f32, 2.0], 0.3)];
        let result = blend_emotion_params(&targets);
        assert!(matches!(
            result,
            Err(EmotionError::InvalidBlendWeights { .. })
        ));
    }

    #[test]
    fn blend_empty_errors() {
        let result = blend_emotion_params(&[]);
        assert!(matches!(result, Err(EmotionError::EmptyParams)));
    }

    #[test]
    fn blend_weighted_average_correctness() {
        let p1 = vec![0.0_f32, 0.0];
        let p2 = vec![2.0_f32, 4.0];
        let targets = vec![(p1, 0.5_f32), (p2, 0.5_f32)];
        let blended = blend_emotion_params(&targets).unwrap();
        assert_relative_eq!(blended[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(blended[1], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn blend_weights_tolerance() {
        // Weights summing to 1.005 — within 0.01 tolerance → should succeed
        let p = vec![1.0_f32; 5];
        let targets = vec![(p.clone(), 0.5025_f32), (p, 0.5025)];
        let result = blend_emotion_params(&targets);
        assert!(
            result.is_ok(),
            "Should allow small tolerance, got {:?}",
            result.err()
        );
    }

    // ── emotion_trajectory ───────────────────────────────────────

    #[test]
    fn trajectory_single_frame() {
        let frames = vec![vec![0.5_f32; 10]];
        let config = EmotionConfig::default();
        let traj = emotion_trajectory(&frames, &config).unwrap();
        assert_eq!(traj.emotions.len(), 1);
        assert_eq!(traj.confidences.len(), 1);
        assert_eq!(traj.av_history.len(), 1);
    }

    #[test]
    fn trajectory_multiple_frames() {
        let frames: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.1; 10]).collect();
        let config = EmotionConfig::default();
        let traj = emotion_trajectory(&frames, &config).unwrap();
        assert_eq!(traj.emotions.len(), 5);
        assert_eq!(traj.confidences.len(), 5);
        assert_eq!(traj.av_history.len(), 5);
    }

    #[test]
    fn trajectory_empty_frames_returns_empty() {
        let config = EmotionConfig::default();
        let traj = emotion_trajectory(&[], &config).unwrap();
        assert!(traj.emotions.is_empty());
    }

    // ── smooth_emotion_trajectory ────────────────────────────────

    #[test]
    fn smooth_same_length_as_input() {
        let frames: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.2; 10]).collect();
        let config = EmotionConfig::default();
        let traj = emotion_trajectory(&frames, &config).unwrap();
        let smoothed = smooth_emotion_trajectory(&traj);
        assert_eq!(smoothed.emotions.len(), traj.emotions.len());
        assert_eq!(smoothed.confidences.len(), traj.confidences.len());
    }

    #[test]
    fn smooth_empty_trajectory() {
        let traj = EmotionTrajectory {
            emotions: Vec::new(),
            confidences: Vec::new(),
            av_history: Vec::new(),
        };
        let smoothed = smooth_emotion_trajectory(&traj);
        assert!(smoothed.emotions.is_empty());
    }

    #[test]
    fn smooth_empty_confidences_does_not_panic() {
        // Regression test: `EmotionTrajectory`'s fields are all public and
        // independently writable, so `emotions` non-empty with
        // `confidences` empty is a shape the type permits even though
        // `emotion_trajectory` itself never produces it. This must not
        // index `confidences[0]` unconditionally.
        let traj = EmotionTrajectory {
            emotions: vec![BasicEmotion::Happy],
            confidences: Vec::new(),
            av_history: Vec::new(),
        };
        let smoothed = smooth_emotion_trajectory(&traj);
        assert!(smoothed.emotions.is_empty());
        assert!(smoothed.confidences.is_empty());
    }

    #[test]
    fn smooth_mismatched_lengths_uses_common_prefix() {
        // `confidences` longer than `emotions`: the result must not exceed
        // the shorter (common) length, so `emotions`/`confidences` stay the
        // same length as each other in the output.
        let traj = EmotionTrajectory {
            emotions: vec![BasicEmotion::Happy, BasicEmotion::Sad],
            confidences: vec![0.9, 0.8, 0.7, 0.6],
            av_history: Vec::new(),
        };
        let smoothed = smooth_emotion_trajectory(&traj);
        assert_eq!(smoothed.emotions.len(), 2);
        assert_eq!(smoothed.confidences.len(), 2);

        // `emotions` longer than `confidences`.
        let traj2 = EmotionTrajectory {
            emotions: vec![BasicEmotion::Happy, BasicEmotion::Sad, BasicEmotion::Angry],
            confidences: vec![0.9],
            av_history: Vec::new(),
        };
        let smoothed2 = smooth_emotion_trajectory(&traj2);
        assert_eq!(smoothed2.emotions.len(), 1);
        assert_eq!(smoothed2.confidences.len(), 1);
    }

    #[test]
    fn smooth_constant_emotion_unchanged() {
        // All frames same params → same dominant emotion → majority vote unchanged
        let frames: Vec<Vec<f32>> = (0..10).map(|_| vec![0.5_f32; 10]).collect();
        let config = EmotionConfig::default();
        let traj = emotion_trajectory(&frames, &config).unwrap();
        let original_dominant = traj.emotions[0];
        let smoothed = smooth_emotion_trajectory(&traj);
        // Every frame should still be original_dominant
        for &e in &smoothed.emotions {
            assert_eq!(e, original_dominant);
        }
    }

    // ── compute_emotion_transition_rate ──────────────────────────

    #[test]
    fn transition_rate_zero_for_constant() {
        let frames: Vec<Vec<f32>> = (0..5).map(|_| vec![0.5_f32; 10]).collect();
        let config = EmotionConfig::default();
        let traj = emotion_trajectory(&frames, &config).unwrap();
        let rate = compute_emotion_transition_rate(&traj);
        assert_abs_diff_eq!(rate, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn transition_rate_single_frame_is_zero() {
        let traj = EmotionTrajectory {
            emotions: vec![BasicEmotion::Happy],
            confidences: vec![1.0],
            av_history: vec![ArousalValence {
                arousal: 0.0,
                valence: 0.0,
            }],
        };
        assert_abs_diff_eq!(compute_emotion_transition_rate(&traj), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn transition_rate_all_different_is_one() {
        // Build a trajectory where adjacent frames are always different emotions
        let all = BasicEmotion::all();
        let emotions: Vec<BasicEmotion> = (0..8).map(|i| all[i]).collect();
        let traj = EmotionTrajectory {
            emotions,
            confidences: vec![0.5; 8],
            av_history: vec![
                ArousalValence {
                    arousal: 0.0,
                    valence: 0.0
                };
                8
            ],
        };
        let rate = compute_emotion_transition_rate(&traj);
        assert_relative_eq!(rate, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn transition_rate_in_unit_interval() {
        let frames: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                let mut p = vec![0.0_f32; 10];
                p[0] = (i as f32 % 2.0) * 2.0 - 1.0;
                p
            })
            .collect();
        let config = EmotionConfig::default();
        let traj = emotion_trajectory(&frames, &config).unwrap();
        let rate = compute_emotion_transition_rate(&traj);
        assert!((0.0..=1.0).contains(&rate), "rate {rate} out of [0,1]");
    }

    // ── format_emotion_result ────────────────────────────────────

    #[test]
    fn format_result_non_empty() {
        let params = vec![0.5_f32; 10];
        let config = EmotionConfig::default();
        let result = recognize_emotion(&params, &config).unwrap();
        let s = format_emotion_result(&result);
        assert!(!s.is_empty());
    }

    #[test]
    fn format_result_contains_dominant_name() {
        let params = vec![0.5_f32; 10];
        let config = EmotionConfig::default();
        let result = recognize_emotion(&params, &config).unwrap();
        let name = result.dominant.name();
        let s = format_emotion_result(&result);
        assert!(
            s.contains(name),
            "Format string '{s}' missing dominant name '{name}'"
        );
    }

    // ── dominant_in_window ───────────────────────────────────────

    #[test]
    fn dominant_in_window_valid_range() {
        let traj = EmotionTrajectory {
            emotions: vec![BasicEmotion::Happy, BasicEmotion::Happy, BasicEmotion::Sad],
            confidences: vec![0.8, 0.9, 0.7],
            av_history: vec![
                ArousalValence {
                    arousal: 0.0,
                    valence: 0.0
                };
                3
            ],
        };
        let dominant = dominant_in_window(&traj, 0, 3).unwrap();
        assert_eq!(dominant, BasicEmotion::Happy);
    }

    #[test]
    fn dominant_in_window_out_of_range_errors() {
        let traj = EmotionTrajectory {
            emotions: vec![BasicEmotion::Happy, BasicEmotion::Sad],
            confidences: vec![0.8, 0.7],
            av_history: vec![
                ArousalValence {
                    arousal: 0.0,
                    valence: 0.0
                };
                2
            ],
        };
        let result = dominant_in_window(&traj, 0, 10);
        assert!(matches!(
            result,
            Err(EmotionError::EmotionIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn dominant_in_window_start_equals_end_errors() {
        let traj = EmotionTrajectory {
            emotions: vec![BasicEmotion::Happy],
            confidences: vec![0.8],
            av_history: vec![ArousalValence {
                arousal: 0.0,
                valence: 0.0,
            }],
        };
        let result = dominant_in_window(&traj, 1, 1);
        assert!(matches!(
            result,
            Err(EmotionError::EmotionIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn dominant_in_window_sub_window() {
        let traj = EmotionTrajectory {
            emotions: vec![
                BasicEmotion::Sad,
                BasicEmotion::Happy,
                BasicEmotion::Happy,
                BasicEmotion::Happy,
                BasicEmotion::Sad,
            ],
            confidences: vec![0.5; 5],
            av_history: vec![
                ArousalValence {
                    arousal: 0.0,
                    valence: 0.0
                };
                5
            ],
        };
        // Window [1..4] = Happy, Happy, Happy
        let dominant = dominant_in_window(&traj, 1, 4).unwrap();
        assert_eq!(dominant, BasicEmotion::Happy);
    }

    // ── EmotionScore ordering ────────────────────────────────────

    #[test]
    fn emotion_score_ordering_reflects_confidence() {
        let params = vec![0.5_f32, -0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = EmotionConfig::default();
        let scores = compute_emotion_scores(&params, &config).unwrap();
        assert!(scores[0].confidence >= scores[1].confidence);
        assert!(scores[1].confidence >= scores[scores.len() - 1].confidence);
    }

    // ── ArousalValence bounds ────────────────────────────────────

    #[test]
    fn arousal_valence_within_bounds_for_extreme_params() {
        let params_pos = vec![100.0_f32; 10];
        let av = compute_arousal_valence(&params_pos, &EmotionConfig::default()).unwrap();
        assert!(av.arousal >= -1.0 && av.arousal <= 1.0);
        assert!(av.valence >= -1.0 && av.valence <= 1.0);

        let params_neg = vec![-100.0_f32; 10];
        let av2 = compute_arousal_valence(&params_neg, &EmotionConfig::default()).unwrap();
        assert!(av2.arousal >= -1.0 && av2.arousal <= 1.0);
        assert!(av2.valence >= -1.0 && av2.valence <= 1.0);
    }

    // ── EmotionResult intensity ──────────────────────────────────

    #[test]
    fn emotion_result_intensity_in_unit_interval() {
        for mag in [0.0_f32, 0.1, 0.5, 1.5, 3.0, 10.0] {
            let params = vec![mag; 20];
            let config = EmotionConfig::default();
            if params.len() >= 2 {
                if let Ok(result) = recognize_emotion(&params, &config) {
                    assert!(
                        (0.0..=1.0).contains(&result.intensity),
                        "Intensity {} out of range for mag={}",
                        result.intensity,
                        mag
                    );
                }
            }
        }
    }
}
