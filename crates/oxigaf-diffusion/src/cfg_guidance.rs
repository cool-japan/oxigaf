//! Classifier-Free Guidance (CFG) utilities for diffusion model inference.
//!
//! CFG steers generation toward a conditioning signal by scaling the difference
//! between conditional and unconditional noise predictions:
//!
//! ```text
//! output = unconditional + scale * (conditional - unconditional)
//!        = (1 - scale) * unconditional + scale * conditional
//! ```
//!
//! ## Features
//!
//! - **Core CFG**: [`apply_cfg`] for single conditional guidance.
//! - **Multi-guidance**: [`apply_cfg_multi_guidance`] for weighted sum of multiple conditioning signals.
//! - **Time-varying scales**: [`CfgScaleSchedule`] trait with constant, linear, cosine,
//!   and threshold schedules.
//! - **Dynamic thresholding**: [`DynamicThresholdCfg`] implements Imagen-style thresholding
//!   to reduce oversaturation at high CFG scales.
//! - **Noise rescaling**: [`rescale_cfg_result`] prevents oversaturation by matching
//!   unconditional prediction standard deviation.
//! - **Multi-view CFG**: [`apply_cfg_multi_view`] applies guidance per-view in a stacked
//!   multi-view latent tensor.
//! - **Negative prompt blending**: [`blend_unconditional`] interpolates between null and
//!   negative unconditional embeddings.
//!
//! ## Example
//!
//! ```
//! use oxigaf_diffusion::cfg_guidance::{apply_cfg, CfgGuidance};
//!
//! let cond = vec![1.0f32, 2.0, 3.0];
//! let uncond = vec![0.0f32, 0.0, 0.0];
//! let out = apply_cfg(&cond, &uncond, 7.5).unwrap();
//! // out ≈ [7.5, 15.0, 22.5]
//! assert!((out[0] - 7.5).abs() < 1e-5);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during classifier-free guidance operations.
#[derive(Debug, Error)]
pub enum CfgError {
    /// Slice lengths do not match when they must be equal.
    #[error("Length mismatch: expected {expected}, actual {actual}")]
    LengthMismatch {
        /// The required length.
        expected: usize,
        /// The actual length received.
        actual: usize,
    },

    /// Guidance scale is negative (must be >= 0).
    #[error("Invalid scale: {0}")]
    InvalidScale(String),

    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Operation received empty input where non-empty is required.
    #[error("Empty input")]
    EmptyInput,
}

// ---------------------------------------------------------------------------
// Core CFG function
// ---------------------------------------------------------------------------

/// Apply classifier-free guidance to noise predictions.
///
/// Computes:
/// ```text
/// output[i] = unconditional[i] + scale * (conditional[i] - unconditional[i])
/// ```
/// which is equivalent to `(1 - scale) * unconditional + scale * conditional`.
///
/// # Errors
///
/// - [`CfgError::LengthMismatch`] if slice lengths differ.
/// - [`CfgError::InvalidScale`] if `scale < 0`.
pub fn apply_cfg(
    conditional: &[f32],
    unconditional: &[f32],
    scale: f32,
) -> Result<Vec<f32>, CfgError> {
    if scale < 0.0 {
        return Err(CfgError::InvalidScale(format!(
            "scale must be >= 0, got {scale}"
        )));
    }
    if conditional.len() != unconditional.len() {
        return Err(CfgError::LengthMismatch {
            expected: unconditional.len(),
            actual: conditional.len(),
        });
    }
    let output = conditional
        .iter()
        .zip(unconditional.iter())
        .map(|(&c, &u)| u + scale * (c - u))
        .collect();
    Ok(output)
}

// ---------------------------------------------------------------------------
// Multi-guidance CFG
// ---------------------------------------------------------------------------

/// Apply multi-guidance CFG: combine multiple conditional signals with different scales.
///
/// Computes:
/// ```text
/// output = unconditional + Σ_i  scale_i * (conditional_i - unconditional)
/// ```
///
/// # Errors
///
/// - [`CfgError::LengthMismatch`] if any conditional or scales length mismatches.
/// - [`CfgError::InvalidScale`] if any scale < 0.
/// - [`CfgError::EmptyInput`] if `conditionals` is empty.
pub fn apply_cfg_multi_guidance(
    conditionals: &[&[f32]],
    unconditional: &[f32],
    scales: &[f32],
) -> Result<Vec<f32>, CfgError> {
    if conditionals.len() != scales.len() {
        return Err(CfgError::LengthMismatch {
            expected: conditionals.len(),
            actual: scales.len(),
        });
    }

    // Validate all scales.
    for (i, &s) in scales.iter().enumerate() {
        if s < 0.0 {
            return Err(CfgError::InvalidScale(format!(
                "scale[{i}] must be >= 0, got {s}"
            )));
        }
    }

    // Validate all conditional lengths.
    for (i, cond) in conditionals.iter().enumerate() {
        if cond.len() != unconditional.len() {
            return Err(CfgError::LengthMismatch {
                expected: unconditional.len(),
                actual: cond.len(),
            });
        }
        let _ = i; // suppress unused warning
    }

    let mut output = unconditional.to_vec();

    for (cond, &scale) in conditionals.iter().zip(scales.iter()) {
        for i in 0..output.len() {
            // output[i] was initialised to unconditional[i]; add each guidance's delta
            output[i] += scale * (cond[i] - unconditional[i]);
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// CfgScaleSchedule trait
// ---------------------------------------------------------------------------

/// Provides a time-varying guidance scale over denoising timesteps.
pub trait CfgScaleSchedule {
    /// Returns the guidance scale at a given `timestep`.
    ///
    /// `timestep = 0` corresponds to the noisiest step; `timestep = total_timesteps - 1`
    /// corresponds to the cleanest step.
    fn scale_at(&self, timestep: usize, total_timesteps: usize) -> f32;

    /// Human-readable description of this schedule.
    fn description(&self) -> &str;
}

// ---------------------------------------------------------------------------
// ConstantCfgSchedule
// ---------------------------------------------------------------------------

/// A guidance schedule that returns the same scale at every timestep.
#[derive(Debug, Clone)]
pub struct ConstantCfgSchedule {
    /// The fixed guidance scale.
    pub scale: f32,
}

impl ConstantCfgSchedule {
    /// Create a new constant schedule with the given scale.
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}

impl CfgScaleSchedule for ConstantCfgSchedule {
    fn scale_at(&self, _timestep: usize, _total_timesteps: usize) -> f32 {
        self.scale
    }

    fn description(&self) -> &str {
        "constant"
    }
}

// ---------------------------------------------------------------------------
// LinearCfgSchedule
// ---------------------------------------------------------------------------

/// Linearly interpolate guidance scale from `start_scale` (at timestep 0)
/// to `end_scale` (at timestep `T-1`).
#[derive(Debug, Clone)]
pub struct LinearCfgSchedule {
    /// Scale at the noisiest step (timestep 0).
    pub start_scale: f32,
    /// Scale at the cleanest step (timestep T-1).
    pub end_scale: f32,
}

impl LinearCfgSchedule {
    /// Create a new linear schedule.
    pub fn new(start_scale: f32, end_scale: f32) -> Self {
        Self {
            start_scale,
            end_scale,
        }
    }
}

impl CfgScaleSchedule for LinearCfgSchedule {
    fn scale_at(&self, timestep: usize, total_timesteps: usize) -> f32 {
        if total_timesteps <= 1 {
            return self.start_scale;
        }
        let t = timestep as f32;
        let t_max = (total_timesteps - 1) as f32;
        self.start_scale + (self.end_scale - self.start_scale) * t / t_max
    }

    fn description(&self) -> &str {
        "linear"
    }
}

// ---------------------------------------------------------------------------
// CosineCfgSchedule
// ---------------------------------------------------------------------------

/// Cosine guidance schedule from high scale (noisy) to low scale (clean).
///
/// At `t=0` the schedule returns `max_scale`; at `t=T-1` it returns `min_scale`.
/// The intermediate values follow a half-cosine curve.
#[derive(Debug, Clone)]
pub struct CosineCfgSchedule {
    /// Scale at the noisiest step (timestep 0).
    pub max_scale: f32,
    /// Scale at the cleanest step (timestep T-1).
    pub min_scale: f32,
}

impl CosineCfgSchedule {
    /// Create a new cosine schedule.
    pub fn new(max_scale: f32, min_scale: f32) -> Self {
        Self {
            max_scale,
            min_scale,
        }
    }
}

impl CfgScaleSchedule for CosineCfgSchedule {
    fn scale_at(&self, timestep: usize, total_timesteps: usize) -> f32 {
        if total_timesteps <= 1 {
            return self.max_scale;
        }
        let t = timestep as f32;
        let t_max = (total_timesteps - 1) as f32;
        // scale = min + 0.5*(max-min)*(1 + cos(π * t / (T-1)))
        // t=0 → cos(0)=1 → max_scale; t=T-1 → cos(π)=-1 → min_scale
        self.min_scale
            + 0.5
                * (self.max_scale - self.min_scale)
                * (1.0 + (std::f32::consts::PI * t / t_max).cos())
    }

    fn description(&self) -> &str {
        "cosine"
    }
}

// ---------------------------------------------------------------------------
// ThresholdCfgSchedule
// ---------------------------------------------------------------------------

/// Full guidance until a threshold step, then reduced guidance.
///
/// Steps `t < threshold_step` use `full_scale`; all subsequent steps use
/// `reduced_scale`. This is useful for focusing strong conditioning during
/// the noisy phase while allowing the model more freedom during refinement.
#[derive(Debug, Clone)]
pub struct ThresholdCfgSchedule {
    /// Scale used before the threshold step.
    pub full_scale: f32,
    /// Scale used at and after the threshold step.
    pub reduced_scale: f32,
    /// The step index at which to switch from `full_scale` to `reduced_scale`.
    pub threshold_step: usize,
}

impl ThresholdCfgSchedule {
    /// Create a new threshold schedule.
    pub fn new(full_scale: f32, reduced_scale: f32, threshold_step: usize) -> Self {
        Self {
            full_scale,
            reduced_scale,
            threshold_step,
        }
    }
}

impl CfgScaleSchedule for ThresholdCfgSchedule {
    fn scale_at(&self, timestep: usize, _total_timesteps: usize) -> f32 {
        if timestep < self.threshold_step {
            self.full_scale
        } else {
            self.reduced_scale
        }
    }

    fn description(&self) -> &str {
        "threshold"
    }
}

// ---------------------------------------------------------------------------
// DynamicThresholdCfg
// ---------------------------------------------------------------------------

/// Dynamic thresholding for diffusion model predictions.
///
/// Described in the Imagen paper (Saharia et al., 2022). At high CFG scales,
/// predictions can saturate (exceed the valid range) leading to artifacts.
/// Dynamic thresholding clamps each prediction to `[-s, s]` and normalises,
/// where `s` is the specified percentile of the absolute values.
///
/// The procedure:
/// 1. Compute `s = percentile(|x|)`.
/// 2. `s = max(s, 1.0)` (never shrink values below their original magnitude).
/// 3. `output = clamp(x, -s, s) / s`.
#[derive(Debug, Clone)]
pub struct DynamicThresholdCfg {
    /// Percentile value in `(0, 1]` (e.g. `0.995` = 99.5th percentile).
    pub percentile: f32,
}

impl DynamicThresholdCfg {
    /// Create a new `DynamicThresholdCfg`.
    ///
    /// # Errors
    ///
    /// Returns [`CfgError::InvalidConfig`] if `percentile` is not in `(0, 1]`.
    pub fn new(percentile: f32) -> Result<Self, CfgError> {
        if percentile <= 0.0 || percentile > 1.0 {
            return Err(CfgError::InvalidConfig(format!(
                "percentile must be in (0, 1], got {percentile}"
            )));
        }
        Ok(Self { percentile })
    }

    /// Compute the threshold value `s` for a given slice.
    ///
    /// `s = percentile(|x|)`, clamped to a minimum of `1.0`.
    pub fn threshold_value(&self, predictions: &[f32]) -> f32 {
        if predictions.is_empty() {
            return 1.0;
        }
        let mut abs_vals: Vec<f32> = predictions.iter().map(|&x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = abs_vals.len();
        // Index = floor(percentile * n), clamped to [0, n-1]
        let idx = ((self.percentile * n as f32).floor() as usize).min(n.saturating_sub(1));
        let s = abs_vals.get(idx).copied().unwrap_or(1.0);
        s.max(1.0)
    }

    /// Apply dynamic thresholding to a prediction tensor.
    ///
    /// Returns an empty `Vec` if `predictions` is empty.
    pub fn apply(&self, predictions: &[f32]) -> Vec<f32> {
        if predictions.is_empty() {
            return Vec::new();
        }
        let s = self.threshold_value(predictions);
        predictions.iter().map(|&x| x.clamp(-s, s) / s).collect()
    }
}

// ---------------------------------------------------------------------------
// CfgScheduleKind enum (for CfgGuidance)
// ---------------------------------------------------------------------------

/// A closed enumeration of supported CFG schedule kinds.
///
/// This mirrors the trait-based schedule hierarchy but allows `CfgGuidance`
/// to own its schedule without a `Box<dyn ...>`.
#[derive(Debug, Clone)]
pub enum CfgScheduleKind {
    /// Constant guidance scale.
    Constant(f32),
    /// Linearly interpolated from `start` to `end`.
    Linear {
        /// Scale at timestep 0.
        start: f32,
        /// Scale at timestep T-1.
        end: f32,
    },
    /// Cosine schedule from high (noisy) to low (clean).
    Cosine {
        /// Scale at timestep 0 (noisiest).
        max: f32,
        /// Scale at timestep T-1 (cleanest).
        min: f32,
    },
    /// Full scale until `step`, then reduced scale.
    Threshold {
        /// Scale before `step`.
        full: f32,
        /// Scale at and after `step`.
        reduced: f32,
        /// Switchover step index.
        step: usize,
    },
}

impl CfgScheduleKind {
    /// Evaluate the guidance scale at the given timestep.
    pub fn scale_at(&self, timestep: usize, total_timesteps: usize) -> f32 {
        match self {
            Self::Constant(s) => *s,
            Self::Linear { start, end } => {
                if total_timesteps <= 1 {
                    return *start;
                }
                let t = timestep as f32;
                let t_max = (total_timesteps - 1) as f32;
                start + (end - start) * t / t_max
            }
            Self::Cosine { max, min } => {
                if total_timesteps <= 1 {
                    return *max;
                }
                let t = timestep as f32;
                let t_max = (total_timesteps - 1) as f32;
                min + 0.5 * (max - min) * (1.0 + (std::f32::consts::PI * t / t_max).cos())
            }
            Self::Threshold {
                full,
                reduced,
                step,
            } => {
                if timestep < *step {
                    *full
                } else {
                    *reduced
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CfgGuidance (main orchestrator)
// ---------------------------------------------------------------------------

/// Main orchestrator for classifier-free guidance.
///
/// Combines a schedule, optional dynamic thresholding, and optional noise
/// rescaling into a single `apply` call.
#[derive(Debug, Clone)]
pub struct CfgGuidance {
    /// The guidance scale schedule.
    pub schedule: CfgScheduleKind,
    /// Optional dynamic thresholding (Imagen-style).
    pub dynamic_threshold: Option<DynamicThresholdCfg>,
    /// If `true`, rescale the CFG output to match the unconditional prediction's
    /// standard deviation before returning.
    pub rescale_noise: bool,
    /// Blend factor for noise rescaling: `rescaled * factor + output * (1 - factor)`.
    pub rescale_factor: f32,
}

impl CfgGuidance {
    /// Create a `CfgGuidance` with a constant scale.
    ///
    /// # Errors
    ///
    /// Returns [`CfgError::InvalidScale`] if `scale < 0`.
    pub fn constant(scale: f32) -> Result<Self, CfgError> {
        if scale < 0.0 {
            return Err(CfgError::InvalidScale(format!(
                "scale must be >= 0, got {scale}"
            )));
        }
        Ok(Self {
            schedule: CfgScheduleKind::Constant(scale),
            dynamic_threshold: None,
            rescale_noise: false,
            rescale_factor: 0.7,
        })
    }

    /// Create a `CfgGuidance` with a linear schedule.
    pub fn linear(start_scale: f32, end_scale: f32) -> Self {
        Self {
            schedule: CfgScheduleKind::Linear {
                start: start_scale,
                end: end_scale,
            },
            dynamic_threshold: None,
            rescale_noise: false,
            rescale_factor: 0.7,
        }
    }

    /// Create a `CfgGuidance` with a cosine schedule.
    pub fn cosine(max_scale: f32, min_scale: f32) -> Self {
        Self {
            schedule: CfgScheduleKind::Cosine {
                max: max_scale,
                min: min_scale,
            },
            dynamic_threshold: None,
            rescale_noise: false,
            rescale_factor: 0.7,
        }
    }

    /// Get the guidance scale at a given timestep.
    pub fn scale_at(&self, timestep: usize, total_timesteps: usize) -> f32 {
        self.schedule.scale_at(timestep, total_timesteps)
    }

    /// Apply the full CFG pipeline.
    ///
    /// Steps:
    /// 1. Compute scale from schedule.
    /// 2. Apply core CFG formula.
    /// 3. If `dynamic_threshold` is set, apply dynamic thresholding.
    /// 4. If `rescale_noise` is set, rescale to match unconditional std.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`apply_cfg`].
    pub fn apply(
        &self,
        conditional: &[f32],
        unconditional: &[f32],
        timestep: usize,
        total_timesteps: usize,
    ) -> Result<Vec<f32>, CfgError> {
        let scale = self.scale_at(timestep, total_timesteps);
        let mut output = apply_cfg(conditional, unconditional, scale)?;

        if let Some(ref dyn_thresh) = self.dynamic_threshold {
            output = dyn_thresh.apply(&output);
        }

        if self.rescale_noise {
            output = rescale_cfg_result(&output, unconditional, self.rescale_factor);
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Noise rescaling
// ---------------------------------------------------------------------------

/// Rescale CFG noise prediction to match unconditional standard deviation.
///
/// Prevents oversaturation at high guidance scales by anchoring the output
/// magnitude to the unconditional prediction scale.
///
/// ```text
/// std_uncond = std(unconditional)
/// std_cfg    = std(cfg_output)
/// rescaled   = cfg_output * (std_uncond / max(std_cfg, 1e-8))
/// final      = rescaled * factor + cfg_output * (1 - factor)
/// ```
pub fn rescale_cfg_result(cfg_output: &[f32], unconditional: &[f32], factor: f32) -> Vec<f32> {
    let std_uncond = compute_std(unconditional);
    let std_cfg = compute_std(cfg_output);
    let ratio = std_uncond / std_cfg.max(1e-8);
    cfg_output
        .iter()
        .map(|&x| {
            let rescaled = x * ratio;
            rescaled * factor + x * (1.0 - factor)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-view CFG
// ---------------------------------------------------------------------------

/// Apply CFG separately to each view in a multi-view latent stack.
///
/// Both `conditional` and `unconditional` are flat `[N_views * latent_size]`
/// row-major tensors. Guidance is applied independently per view.
///
/// # Errors
///
/// - [`CfgError::LengthMismatch`] if `conditional.len() != unconditional.len()`.
/// - [`CfgError::InvalidConfig`] if the length is not divisible by `num_views`.
/// - [`CfgError::InvalidScale`] if `scale < 0`.
pub fn apply_cfg_multi_view(
    conditional: &[f32],
    unconditional: &[f32],
    scale: f32,
    num_views: usize,
) -> Result<Vec<f32>, CfgError> {
    if scale < 0.0 {
        return Err(CfgError::InvalidScale(format!(
            "scale must be >= 0, got {scale}"
        )));
    }
    if conditional.len() != unconditional.len() {
        return Err(CfgError::LengthMismatch {
            expected: unconditional.len(),
            actual: conditional.len(),
        });
    }
    let total = conditional.len();
    if num_views == 0 || !total.is_multiple_of(num_views) {
        return Err(CfgError::InvalidConfig(format!(
            "total length {total} is not divisible by num_views {num_views}"
        )));
    }
    let latent_size = total / num_views;
    let mut output = Vec::with_capacity(total);
    for v in 0..num_views {
        let start = v * latent_size;
        let end = start + latent_size;
        let cond_view = &conditional[start..end];
        let uncond_view = &unconditional[start..end];
        // apply_cfg validates scale >= 0, which we already checked above
        let view_out = apply_cfg(cond_view, uncond_view, scale)?;
        output.extend_from_slice(&view_out);
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Negative prompt / unconditional blending
// ---------------------------------------------------------------------------

/// Interpolate between two unconditional embeddings.
///
/// Used to blend a null (empty prompt) embedding with a negative prompt
/// embedding. A `strength` of `0.0` returns `null_uncond` unchanged;
/// `1.0` returns `negative_uncond` unchanged.
///
/// `strength` is clamped to `[0, 1]`.
///
/// # Errors
///
/// - [`CfgError::LengthMismatch`] if the two embeddings have different lengths.
pub fn blend_unconditional(
    null_uncond: &[f32],
    negative_uncond: &[f32],
    strength: f32,
) -> Result<Vec<f32>, CfgError> {
    if null_uncond.len() != negative_uncond.len() {
        return Err(CfgError::LengthMismatch {
            expected: null_uncond.len(),
            actual: negative_uncond.len(),
        });
    }
    let s = strength.clamp(0.0, 1.0);
    let output = null_uncond
        .iter()
        .zip(negative_uncond.iter())
        .map(|(&n, &neg)| n + s * (neg - n))
        .collect();
    Ok(output)
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Compute the population mean of a slice.
///
/// Returns `0.0` for an empty slice.
pub fn compute_mean(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f32>() / vals.len() as f32
}

/// Compute the population standard deviation of a slice.
///
/// Returns `0.0` for slices with fewer than 2 elements.
pub fn compute_std(vals: &[f32]) -> f32 {
    if vals.len() < 2 {
        return 0.0;
    }
    let mean = compute_mean(vals);
    let variance = vals.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- apply_cfg -----------------------------------------------------------

    #[test]
    fn test_apply_cfg_scale_one_returns_conditional() {
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![4.0f32, 5.0, 6.0];
        let out = apply_cfg(&cond, &uncond, 1.0).unwrap();
        for (o, c) in out.iter().zip(cond.iter()) {
            assert!((o - c).abs() < 1e-6, "expected {c}, got {o}");
        }
    }

    #[test]
    fn test_apply_cfg_scale_zero_returns_unconditional() {
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![4.0f32, 5.0, 6.0];
        let out = apply_cfg(&cond, &uncond, 0.0).unwrap();
        for (o, u) in out.iter().zip(uncond.iter()) {
            assert!((o - u).abs() < 1e-6, "expected {u}, got {o}");
        }
    }

    #[test]
    fn test_apply_cfg_typical_scale() {
        let cond = vec![1.0f32];
        let uncond = vec![0.0f32];
        // output = 0 + 7.5 * (1 - 0) = 7.5
        let out = apply_cfg(&cond, &uncond, 7.5).unwrap();
        assert!((out[0] - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_apply_cfg_length_mismatch() {
        let cond = vec![1.0f32, 2.0];
        let uncond = vec![1.0f32];
        let err = apply_cfg(&cond, &uncond, 1.0).unwrap_err();
        assert!(matches!(err, CfgError::LengthMismatch { .. }));
    }

    #[test]
    fn test_apply_cfg_negative_scale_err() {
        let cond = vec![1.0f32];
        let uncond = vec![0.0f32];
        let err = apply_cfg(&cond, &uncond, -1.0).unwrap_err();
        assert!(matches!(err, CfgError::InvalidScale(_)));
    }

    // ---- apply_cfg_multi_guidance --------------------------------------------

    #[test]
    fn test_apply_cfg_multi_guidance_single_equals_apply_cfg() {
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![0.5f32, 0.5, 0.5];
        let expected = apply_cfg(&cond, &uncond, 5.0).unwrap();
        let multi_out = apply_cfg_multi_guidance(&[cond.as_slice()], &uncond, &[5.0]).unwrap();
        for (e, m) in expected.iter().zip(multi_out.iter()) {
            assert!((e - m).abs() < 1e-5, "expected {e}, got {m}");
        }
    }

    #[test]
    fn test_apply_cfg_multi_guidance_two_guidances() {
        // output = uncond + s1*(cond1 - uncond) + s2*(cond2 - uncond)
        let uncond = vec![0.0f32, 0.0];
        let cond1 = vec![1.0f32, 1.0];
        let cond2 = vec![2.0f32, 2.0];
        // output = 0 + 1*(1-0) + 1*(2-0) = 3
        let out =
            apply_cfg_multi_guidance(&[cond1.as_slice(), cond2.as_slice()], &uncond, &[1.0, 1.0])
                .unwrap();
        for &o in &out {
            assert!((o - 3.0).abs() < 1e-5, "expected 3.0, got {o}");
        }
    }

    #[test]
    fn test_apply_cfg_multi_guidance_scale_mismatch_err() {
        let uncond = vec![0.0f32];
        let cond1 = vec![1.0f32];
        // 1 conditional, 2 scales → mismatch
        let err = apply_cfg_multi_guidance(&[cond1.as_slice()], &uncond, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, CfgError::LengthMismatch { .. }));
    }

    // ---- ConstantCfgSchedule -------------------------------------------------

    #[test]
    fn test_constant_cfg_schedule_always_same() {
        let sched = ConstantCfgSchedule::new(7.5);
        for t in 0..20 {
            assert!((sched.scale_at(t, 20) - 7.5).abs() < 1e-6);
        }
    }

    // ---- LinearCfgSchedule ---------------------------------------------------

    #[test]
    fn test_linear_cfg_schedule_endpoints() {
        let sched = LinearCfgSchedule::new(1.0, 10.0);
        assert!((sched.scale_at(0, 10) - 1.0).abs() < 1e-5, "start");
        assert!((sched.scale_at(9, 10) - 10.0).abs() < 1e-5, "end");
    }

    // ---- CosineCfgSchedule ---------------------------------------------------

    #[test]
    fn test_cosine_cfg_schedule_endpoints() {
        let sched = CosineCfgSchedule::new(10.0, 2.0);
        // t=0 → max_scale = 10
        assert!(
            (sched.scale_at(0, 100) - 10.0).abs() < 1e-4,
            "at t=0 expected 10, got {}",
            sched.scale_at(0, 100)
        );
        // t=T-1 → min_scale = 2
        assert!(
            (sched.scale_at(99, 100) - 2.0).abs() < 1e-4,
            "at t=99 expected 2, got {}",
            sched.scale_at(99, 100)
        );
    }

    // ---- ThresholdCfgSchedule ------------------------------------------------

    #[test]
    fn test_threshold_cfg_schedule_below_and_above() {
        let sched = ThresholdCfgSchedule::new(10.0, 3.0, 5);
        // Below threshold
        assert!((sched.scale_at(3, 20) - 10.0).abs() < 1e-6);
        // At threshold
        assert!((sched.scale_at(5, 20) - 3.0).abs() < 1e-6);
        // Above threshold
        assert!((sched.scale_at(10, 20) - 3.0).abs() < 1e-6);
    }

    // ---- DynamicThresholdCfg -------------------------------------------------

    #[test]
    fn test_dynamic_threshold_clamps_high_values() {
        let dt = DynamicThresholdCfg::new(0.995).unwrap();
        // Large outliers should be clamped toward [-1, 1]
        let preds = vec![100.0f32, -100.0, 0.5, -0.5];
        let out = dt.apply(&preds);
        for &o in &out {
            assert!(o.abs() <= 1.0 + 1e-5, "expected clamped output, got {o}");
        }
    }

    #[test]
    fn test_dynamic_threshold_small_values_unchanged_shape() {
        // When all |x| < 1 the threshold s is clamped to 1, so output = x/1 = x.
        let dt = DynamicThresholdCfg::new(0.995).unwrap();
        let preds = vec![0.1f32, -0.2, 0.3];
        let out = dt.apply(&preds);
        assert_eq!(out.len(), preds.len());
        for (&o, &p) in out.iter().zip(preds.iter()) {
            assert!((o - p).abs() < 1e-5, "expected {p}, got {o}");
        }
    }

    #[test]
    fn test_dynamic_threshold_invalid_percentile() {
        let err = DynamicThresholdCfg::new(0.0).unwrap_err();
        assert!(matches!(err, CfgError::InvalidConfig(_)));
        let err2 = DynamicThresholdCfg::new(-0.1).unwrap_err();
        assert!(matches!(err2, CfgError::InvalidConfig(_)));
    }

    // ---- CfgGuidance ---------------------------------------------------------

    #[test]
    fn test_cfg_guidance_constant_apply() {
        let guidance = CfgGuidance::constant(7.5).unwrap();
        let cond = vec![1.0f32, 0.0, 0.0];
        let uncond = vec![0.0f32, 0.0, 0.0];
        let out = guidance.apply(&cond, &uncond, 0, 10).unwrap();
        // output = 0 + 7.5*(1-0) = 7.5
        assert!((out[0] - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_cfg_guidance_constant_negative_err() {
        let err = CfgGuidance::constant(-1.0).unwrap_err();
        assert!(matches!(err, CfgError::InvalidScale(_)));
    }

    // ---- rescale_cfg_result --------------------------------------------------

    #[test]
    fn test_rescale_cfg_result_std_closer_to_uncond() {
        // unconditional with std ~1; cfg output with std ~10
        let uncond: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let cfg_output: Vec<f32> = uncond.iter().map(|&x| x * 10.0).collect();
        let rescaled = rescale_cfg_result(&cfg_output, &uncond, 0.7);
        let std_orig = compute_std(&cfg_output);
        let std_rescaled = compute_std(&rescaled);
        let std_uncond = compute_std(&uncond);
        // The rescaled std should be closer to std_uncond than the original was.
        let dist_before = (std_orig - std_uncond).abs();
        let dist_after = (std_rescaled - std_uncond).abs();
        assert!(dist_after < dist_before, "rescaled std {std_rescaled} should be closer to uncond std {std_uncond} than original {std_orig}");
    }

    // ---- apply_cfg_multi_view ------------------------------------------------

    #[test]
    fn test_apply_cfg_multi_view_independent() {
        // 2 views, latent_size=2
        // View 0: cond=[1,1], uncond=[0,0]
        // View 1: cond=[2,2], uncond=[0,0]
        // scale=1 → output = conditional
        let cond = vec![1.0f32, 1.0, 2.0, 2.0];
        let uncond = vec![0.0f32, 0.0, 0.0, 0.0];
        let out = apply_cfg_multi_view(&cond, &uncond, 1.0, 2).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
        assert!((out[2] - 2.0).abs() < 1e-5);
        assert!((out[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_cfg_multi_view_non_divisible_err() {
        let cond = vec![1.0f32, 2.0, 3.0]; // 3 elements, not divisible by 2
        let uncond = vec![0.0f32, 0.0, 0.0];
        let err = apply_cfg_multi_view(&cond, &uncond, 1.0, 2).unwrap_err();
        assert!(matches!(err, CfgError::InvalidConfig(_)));
    }

    // ---- blend_unconditional -------------------------------------------------

    #[test]
    fn test_blend_unconditional_strength_zero() {
        let null = vec![1.0f32, 2.0, 3.0];
        let neg = vec![4.0f32, 5.0, 6.0];
        let out = blend_unconditional(&null, &neg, 0.0).unwrap();
        for (o, n) in out.iter().zip(null.iter()) {
            assert!((o - n).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blend_unconditional_strength_one() {
        let null = vec![1.0f32, 2.0, 3.0];
        let neg = vec![4.0f32, 5.0, 6.0];
        let out = blend_unconditional(&null, &neg, 1.0).unwrap();
        for (o, n) in out.iter().zip(neg.iter()) {
            assert!((o - n).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blend_unconditional_length_mismatch_err() {
        let null = vec![1.0f32, 2.0];
        let neg = vec![1.0f32];
        let err = blend_unconditional(&null, &neg, 0.5).unwrap_err();
        assert!(matches!(err, CfgError::LengthMismatch { .. }));
    }

    // ---- compute_std / compute_mean ------------------------------------------

    #[test]
    fn test_compute_std_known_values() {
        // std([2, 4, 4, 4, 5, 5, 7, 9]) = 2.0
        let vals = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = compute_std(&vals);
        assert!((s - 2.0).abs() < 1e-4, "expected ~2.0, got {s}");
    }

    #[test]
    fn test_compute_mean_correct() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let m = compute_mean(&vals);
        assert!((m - 3.0).abs() < 1e-6, "expected 3.0, got {m}");
    }
}
