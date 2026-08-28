//! # Guidance Rescaling for Classifier-Free Guidance
//!
//! Implements advanced guidance rescaling strategies for CFG (Classifier-Free Guidance)
//! to prevent oversaturation at high guidance scales (7.5+).
//!
//! ## Strategies
//!
//! - **Phi-rescaling** — from "Common Diffusion Noise Schedules are Flawed"
//!   (Lin et al. 2024). Normalizes the CFG output std to match the conditional
//!   prediction std, interpolated by a factor φ ∈ [0, 1].
//!
//! - **Dynamic thresholding** — from Imagen (Saharia et al. 2022). Clips latent
//!   values at a high percentile of their absolute magnitude, then rescales to
//!   [-1, 1]. Prevents individual dimensions from saturating the output.
//!
//! - **Adaptive guidance scale** — adjusts the effective CFG scale so the
//!   output standard deviation matches a desired target.
//!
//! - **Self-Attention Guidance (SAG)** — uses a blurred version of the
//!   prediction as the "negative" guidance signal.
//!
//! - **Annealed phi schedule** — decays the rescaling strength linearly over
//!   the denoising trajectory so early (noisy) steps use full rescaling while
//!   later (detail) steps reduce it.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during guidance rescaling operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GuidanceRescalingError {
    /// Input predictions array is empty.
    #[error("Empty predictions: cannot rescale empty array")]
    EmptyPredictions,

    /// Conditional and unconditional prediction arrays have different lengths.
    #[error("Array length mismatch: cond has {cond}, uncond has {uncond}")]
    LengthMismatch { cond: usize, uncond: usize },

    /// Guidance scale is negative, which is undefined.
    #[error("Invalid guidance scale {scale}: must be >= 0")]
    InvalidGuidanceScale { scale: f32 },

    /// Percentile value is outside the valid open interval (0, 100).
    #[error("Invalid percentile {p}: must be in (0, 100)")]
    InvalidPercentile { p: f32 },

    /// Phi rescaling factor is outside [0, 1].
    #[error("Invalid rescale factor {factor}: must be in [0, 1]")]
    InvalidRescaleFactor { factor: f32 },

    /// `clamp_range` bounds are reversed or non-finite.
    #[error("Invalid clamp range ({lo}, {hi}): both bounds must be finite with lo <= hi")]
    InvalidClampRange { lo: f32, hi: f32 },
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the full guidance-rescaling pipeline.
#[derive(Debug, Clone)]
pub struct RescalingConfig {
    /// CFG guidance scale (default 7.5).
    pub guidance_scale: f32,
    /// Phi factor for phi-rescaling; 0 = no rescaling, 1 = full (default 0.7).
    pub rescale_factor: f32,
    /// Percentile for dynamic thresholding, e.g. 99.5 (default 99.5).
    pub dynamic_threshold_pct: f32,
    /// Whether to apply phi-rescaling (default true).
    pub use_phi_rescaling: bool,
    /// Whether to apply dynamic thresholding (default false).
    pub use_dynamic_thresholding: bool,
    /// If `Some`, clamp the output to this `(min, max)` range (default None).
    pub clamp_range: Option<(f32, f32)>,
}

impl Default for RescalingConfig {
    fn default() -> Self {
        Self {
            guidance_scale: 7.5,
            rescale_factor: 0.7,
            dynamic_threshold_pct: 99.5,
            use_phi_rescaling: true,
            use_dynamic_thresholding: false,
            clamp_range: None,
        }
    }
}

impl RescalingConfig {
    /// Validate all configuration fields, returning an error on the first
    /// invalid value found.
    pub fn validate(&self) -> Result<(), GuidanceRescalingError> {
        if self.guidance_scale < 0.0 {
            return Err(GuidanceRescalingError::InvalidGuidanceScale {
                scale: self.guidance_scale,
            });
        }
        if !(0.0..=1.0).contains(&self.rescale_factor) {
            return Err(GuidanceRescalingError::InvalidRescaleFactor {
                factor: self.rescale_factor,
            });
        }
        if self.dynamic_threshold_pct <= 0.0 || self.dynamic_threshold_pct >= 100.0 {
            return Err(GuidanceRescalingError::InvalidPercentile {
                p: self.dynamic_threshold_pct,
            });
        }
        if let Some((lo, hi)) = self.clamp_range {
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(GuidanceRescalingError::InvalidClampRange { lo, hi });
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostic statistics describing a single guidance step.
#[derive(Debug, Clone)]
pub struct GuidanceStats {
    /// Mean of the CFG signal `(cond - uncond) * scale`.
    pub cfg_mean: f32,
    /// Standard deviation of the CFG signal.
    pub cfg_std: f32,
    /// L2 norm of the CFG signal.
    pub cfg_norm: f32,
    /// Standard deviation of the conditional prediction.
    pub cond_std: f32,
    /// Standard deviation of the unconditional prediction.
    pub uncond_std: f32,
    /// Effective rescale factor applied (mirrors `RescalingConfig::rescale_factor`).
    pub scale_factor: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the population standard deviation of `v`.
///
/// Returns `0.0` for empty or single-element slices.
pub fn std_dev(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let mean = v.iter().copied().sum::<f32>() / n as f32;
    let variance = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    variance.sqrt()
}

/// L2 (Euclidean) norm of `v`.
pub fn l2_norm_vec(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Normalize `v` to unit L2 norm.
///
/// Returns a zero vector when the norm is smaller than `1e-8` to avoid
/// division by near-zero.
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm_vec(v);
    if norm < 1e-8 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|&x| x / norm).collect()
}

/// Compute the `percentile`-th value of the absolute values of `values`.
///
/// Uses linear interpolation between the two surrounding sorted samples.
/// `percentile` must be in the open interval (0, 100).
pub fn abs_percentile(values: &[f32], percentile: f32) -> Result<f32, GuidanceRescalingError> {
    if values.is_empty() {
        return Err(GuidanceRescalingError::EmptyPredictions);
    }
    if percentile <= 0.0 || percentile >= 100.0 {
        return Err(GuidanceRescalingError::InvalidPercentile { p: percentile });
    }

    let mut abs_vals: Vec<f32> = values.iter().map(|&x| x.abs()).collect();
    // `total_cmp` is a genuine total order (unlike `partial_cmp` collapsed
    // to `Equal` on NaN), so this can never panic and never silently
    // reorders finite values around a NaN entry.
    abs_vals.sort_by(f32::total_cmp);

    let n = abs_vals.len();
    // Exact float index in [0, n-1]
    let float_idx = (percentile / 100.0) * (n as f32 - 1.0);
    let lo = float_idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = float_idx - lo as f32;

    let result = abs_vals[lo] * (1.0 - frac) + abs_vals[hi] * frac;
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Core guidance functions
// ─────────────────────────────────────────────────────────────────────────────

/// Standard Classifier-Free Guidance.
///
/// `result[i] = uncond[i] + scale * (cond[i] - uncond[i])`
pub fn apply_cfg_guidance(
    cond: &[f32],
    uncond: &[f32],
    scale: f32,
) -> Result<Vec<f32>, GuidanceRescalingError> {
    if cond.is_empty() {
        return Err(GuidanceRescalingError::EmptyPredictions);
    }
    if cond.len() != uncond.len() {
        return Err(GuidanceRescalingError::LengthMismatch {
            cond: cond.len(),
            uncond: uncond.len(),
        });
    }
    if scale < 0.0 {
        return Err(GuidanceRescalingError::InvalidGuidanceScale { scale });
    }
    let result = cond
        .iter()
        .zip(uncond.iter())
        .map(|(&c, &u)| u + scale * (c - u))
        .collect();
    Ok(result)
}

/// Phi-rescaling (Lin et al. 2024, "Common Diffusion Noise Schedules are Flawed").
///
/// Rescales the CFG output so its standard deviation matches that of the
/// conditional prediction, blended by `phi`.
///
/// ```text
/// cfg        = uncond + scale * (cond - uncond)
/// factor     = std(cond) / (std(cfg) + 1e-8)
/// rescaled   = cfg * factor
/// result     = phi * rescaled + (1 - phi) * cfg
/// ```
pub fn phi_rescale(
    cond: &[f32],
    uncond: &[f32],
    scale: f32,
    phi: f32,
) -> Result<Vec<f32>, GuidanceRescalingError> {
    if !(0.0..=1.0).contains(&phi) {
        return Err(GuidanceRescalingError::InvalidRescaleFactor { factor: phi });
    }
    let cfg = apply_cfg_guidance(cond, uncond, scale)?;

    let std_cond = std_dev(cond);
    let std_cfg = std_dev(&cfg);
    let factor = std_cond / (std_cfg + 1e-8);

    let result = cfg
        .iter()
        .map(|&v| {
            let rescaled = v * factor;
            phi * rescaled + (1.0 - phi) * v
        })
        .collect();
    Ok(result)
}

/// Dynamic thresholding (Saharia et al. 2022, Imagen).
///
/// Clips each latent element to `[-s, s]` where `s` is the `percentile`-th
/// percentile of absolute values (but at least 1.0), then divides by `s` to
/// keep the output in `[-1, 1]`.
pub fn dynamic_threshold(
    latents: &[f32],
    percentile: f32,
) -> Result<Vec<f32>, GuidanceRescalingError> {
    if latents.is_empty() {
        return Err(GuidanceRescalingError::EmptyPredictions);
    }
    let s_raw = abs_percentile(latents, percentile)?;
    // Never shrink the range below ±1
    let s = s_raw.max(1.0);
    let result = latents.iter().map(|&x| x.clamp(-s, s) / s).collect();
    Ok(result)
}

/// Adaptive guidance: rescale CFG output so its std matches `target_std`.
///
/// Returns `(rescaled_output, effective_scale_used)`.
///
/// When the output std is near zero, the input is returned unchanged.
pub fn adaptive_guidance_scale(
    cond: &[f32],
    uncond: &[f32],
    target_std: f32,
    scale: f32,
) -> Result<(Vec<f32>, f32), GuidanceRescalingError> {
    let cfg = apply_cfg_guidance(cond, uncond, scale)?;
    let current_std = std_dev(&cfg);

    if current_std < 1e-8 {
        return Ok((cfg, scale));
    }

    let adjust_factor = target_std / current_std;
    let effective_scale = scale * adjust_factor;
    let rescaled = cfg.iter().map(|&v| v * adjust_factor).collect();
    Ok((rescaled, effective_scale))
}

/// Compute diagnostic statistics for one guidance step.
pub fn compute_guidance_stats(
    cond: &[f32],
    uncond: &[f32],
    scale: f32,
    rescale_factor: f32,
) -> Result<GuidanceStats, GuidanceRescalingError> {
    if cond.is_empty() {
        return Err(GuidanceRescalingError::EmptyPredictions);
    }
    if cond.len() != uncond.len() {
        return Err(GuidanceRescalingError::LengthMismatch {
            cond: cond.len(),
            uncond: uncond.len(),
        });
    }
    if scale < 0.0 {
        return Err(GuidanceRescalingError::InvalidGuidanceScale { scale });
    }

    let cfg_signal: Vec<f32> = cond
        .iter()
        .zip(uncond.iter())
        .map(|(&c, &u)| (c - u) * scale)
        .collect();

    let cfg_mean = cfg_signal.iter().copied().sum::<f32>() / cfg_signal.len() as f32;
    let cfg_std = std_dev(&cfg_signal);
    let cfg_norm = l2_norm_vec(&cfg_signal);
    let cond_std = std_dev(cond);
    let uncond_std = std_dev(uncond);

    Ok(GuidanceStats {
        cfg_mean,
        cfg_std,
        cfg_norm,
        cond_std,
        uncond_std,
        scale_factor: rescale_factor,
    })
}

/// Full guidance-rescaling pipeline driven by a [`RescalingConfig`].
///
/// Steps:
/// 1. Apply standard CFG.
/// 2. If `use_phi_rescaling`: apply phi-rescaling.
/// 3. If `use_dynamic_thresholding`: apply dynamic thresholding.
/// 4. If `clamp_range` is set: clamp the output.
pub fn apply_rescaled_guidance(
    cond: &[f32],
    uncond: &[f32],
    config: &RescalingConfig,
) -> Result<Vec<f32>, GuidanceRescalingError> {
    config.validate()?;

    // Step 1 — standard CFG (computed once; step 2 reuses this buffer
    // in place instead of recomputing CFG a second time via `phi_rescale`).
    let mut out = apply_cfg_guidance(cond, uncond, config.guidance_scale)?;

    // Step 2 — phi-rescaling
    if config.use_phi_rescaling {
        phi_rescale_in_place(&mut out, std_dev(cond), config.rescale_factor);
    }

    // Step 3 — dynamic thresholding
    if config.use_dynamic_thresholding {
        out = dynamic_threshold(&out, config.dynamic_threshold_pct)?;
    }

    // Step 4 — optional hard clamp
    if let Some((lo, hi)) = config.clamp_range {
        out.iter_mut().for_each(|v| *v = v.clamp(lo, hi));
    }

    Ok(out)
}

/// In-place equivalent of [`phi_rescale`] given an already-computed CFG
/// vector and the conditional prediction's standard deviation, avoiding a
/// second full CFG pass. See [`phi_rescale`] for the formula.
fn phi_rescale_in_place(cfg: &mut [f32], std_cond: f32, phi: f32) {
    let std_cfg = std_dev(cfg);
    let factor = std_cond / (std_cfg + 1e-8);
    for v in cfg.iter_mut() {
        let rescaled = *v * factor;
        *v = phi * rescaled + (1.0 - phi) * *v;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Box blur and Self-Attention Guidance
// ─────────────────────────────────────────────────────────────────────────────

/// 1-D box filter with clamped (edge-replicated) boundaries.
///
/// `get(i)` supplies the source sample at logical position `i ∈ [0, len)`.
/// `out[i]` receives the mean of the `2*radius+1` samples centred at `i`,
/// with out-of-range offsets clamped to the nearest edge — matching the
/// boundary convention of a naive clamped 2-D box window exactly (the sum
/// of `image[clamp(y+dy), clamp(x+dx)]` over a 2-D window separates into a
/// horizontal pass followed by a vertical pass because each axis is
/// clamped independently). Runs in `O(len)` time via a prefix sum,
/// independent of `radius`.
fn box_filter_1d(len: usize, radius: i64, get: impl Fn(usize) -> f32, out: &mut [f32]) {
    if len == 0 {
        return;
    }
    let mut prefix = vec![0.0f32; len + 1];
    for i in 0..len {
        prefix[i + 1] = prefix[i] + get(i);
    }
    let len_i = len as i64;
    let total = (2 * radius + 1) as f32;
    let first = get(0);
    let last = get(len - 1);
    for x in 0..len_i {
        let lo = x - radius;
        let hi = x + radius;
        let interior_lo = lo.max(0);
        let interior_hi = hi.min(len_i - 1);
        let interior_sum = prefix[(interior_hi + 1) as usize] - prefix[interior_lo as usize];
        let left_count = (interior_lo - lo).max(0) as f32;
        let right_count = (hi - interior_hi).max(0) as f32;
        let sum = left_count * first + interior_sum + right_count * last;
        out[x as usize] = sum / total;
    }
}

/// Box blur on a flat interleaved `f32` image buffer.
///
/// Each output pixel is the arithmetic mean of all pixels within `radius` in
/// each spatial dimension (inclusive). Boundary pixels are clamped rather than
/// zero-padded to avoid artefacts at the edges.
///
/// Implemented as a separable filter (horizontal pass then vertical pass),
/// which is mathematically identical to the naive `O(w*h*c*radius²)` 2-D
/// window for this clamped-boundary convention but runs in `O(w*h*c)` time.
/// `radius` is capped at `max(width, height)` — both because a larger
/// radius cannot change the result (every sample is already in range) and
/// to keep the internal `i64` arithmetic from overflowing for pathological
/// inputs such as `radius = usize::MAX`.
pub fn box_blur_guidance(
    image: &[f32],
    width: u32,
    height: u32,
    channels: u32,
    radius: usize,
) -> Result<Vec<f32>, GuidanceRescalingError> {
    let w = width as usize;
    let h = height as usize;
    let c = channels as usize;
    let expected = w * h * c;

    if expected == 0 {
        return Ok(Vec::new());
    }
    if image.len() != expected {
        return Err(GuidanceRescalingError::LengthMismatch {
            cond: image.len(),
            uncond: expected,
        });
    }

    let radius_i64 = radius.min(w.max(h)) as i64;
    if radius_i64 == 0 {
        return Ok(image.to_vec());
    }

    let mut scratch = vec![0.0f32; w.max(h)];

    // Horizontal pass: image -> temp.
    let mut temp = vec![0.0f32; expected];
    for y in 0..h {
        for ch in 0..c {
            box_filter_1d(
                w,
                radius_i64,
                |x| image[(y * w + x) * c + ch],
                &mut scratch[..w],
            );
            for x in 0..w {
                temp[(y * w + x) * c + ch] = scratch[x];
            }
        }
    }

    // Vertical pass: temp -> output.
    let mut output = vec![0.0f32; expected];
    for x in 0..w {
        for ch in 0..c {
            box_filter_1d(
                h,
                radius_i64,
                |y| temp[(y * w + x) * c + ch],
                &mut scratch[..h],
            );
            for y in 0..h {
                output[(y * w + x) * c + ch] = scratch[y];
            }
        }
    }

    Ok(output)
}

/// Self-Attention Guidance (simplified spatial variant).
///
/// Uses a box-blurred version of `pred` as the "negative" prediction:
///
/// ```text
/// result = pred + sag_scale * (pred - blur(pred))
/// ```
///
/// When `sag_scale = 0.0` the result equals `pred` exactly.
pub fn self_attention_guidance(
    pred: &[f32],
    sag_scale: f32,
    blur_radius: usize,
    width: u32,
    height: u32,
    channels: u32,
) -> Result<Vec<f32>, GuidanceRescalingError> {
    if pred.is_empty() {
        return Err(GuidanceRescalingError::EmptyPredictions);
    }

    let blurred = box_blur_guidance(pred, width, height, channels, blur_radius)?;

    let result = pred
        .iter()
        .zip(blurred.iter())
        .map(|(&p, &b)| p + sag_scale * (p - b))
        .collect();
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Schedule helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Linearly decay phi from `phi_start` (at step 0) to `phi_end` (at `total_steps`).
///
/// When `total_steps == 0`, returns `phi_start`.
pub fn annealed_phi(phi_start: f32, phi_end: f32, current_step: usize, total_steps: usize) -> f32 {
    if total_steps == 0 {
        return phi_start;
    }
    let t = (current_step as f32) / (total_steps as f32);
    phi_start + (phi_end - phi_start) * t
}

/// Compute the effective "direct" guidance scale equivalent when phi-rescaling
/// is active.
///
/// ```text
/// effective = nominal * (1 - phi) + nominal * (cond_std / cfg_std) * phi
/// ```
///
/// Returns `nominal` when `cfg_std` is near zero (avoids division by zero).
pub fn effective_scale_with_phi(nominal_scale: f32, cond_std: f32, cfg_std: f32, phi: f32) -> f32 {
    if cfg_std < 1e-8 {
        return nominal_scale;
    }
    let ratio = cond_std / cfg_std;
    nominal_scale * (1.0 - phi) + nominal_scale * ratio * phi
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tolerance helper ──────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ═════════════════════════════════════════════════════════════════════════
    // apply_cfg_guidance
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cfg_guidance_zero_scale_returns_uncond() {
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![0.5f32, 0.5, 0.5];
        let result = apply_cfg_guidance(&cond, &uncond, 0.0).unwrap();
        assert_eq!(result, uncond);
    }

    #[test]
    fn test_cfg_guidance_scale_one_returns_cond() {
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![0.5f32, 0.5, 0.5];
        let result = apply_cfg_guidance(&cond, &uncond, 1.0).unwrap();
        for (r, &c) in result.iter().zip(cond.iter()) {
            assert!(approx_eq(*r, c, 1e-6));
        }
    }

    #[test]
    fn test_cfg_guidance_known_values() {
        // scale=2: result = uncond + 2*(cond-uncond) = 2*cond - uncond
        let cond = vec![3.0f32, 5.0];
        let uncond = vec![1.0f32, 1.0];
        let result = apply_cfg_guidance(&cond, &uncond, 2.0).unwrap();
        assert!(approx_eq(result[0], 5.0, 1e-5)); // 1 + 2*(3-1) = 5
        assert!(approx_eq(result[1], 9.0, 1e-5)); // 1 + 2*(5-1) = 9
    }

    #[test]
    fn test_cfg_guidance_length_mismatch() {
        let cond = vec![1.0f32, 2.0];
        let uncond = vec![0.5f32];
        let err = apply_cfg_guidance(&cond, &uncond, 7.5).unwrap_err();
        assert!(matches!(err, GuidanceRescalingError::LengthMismatch { .. }));
    }

    #[test]
    fn test_cfg_guidance_empty_cond() {
        let err = apply_cfg_guidance(&[], &[], 7.5).unwrap_err();
        assert!(matches!(err, GuidanceRescalingError::EmptyPredictions));
    }

    #[test]
    fn test_cfg_guidance_negative_scale_error() {
        let cond = vec![1.0f32, 2.0];
        let uncond = vec![0.5f32, 0.5];
        let err = apply_cfg_guidance(&cond, &uncond, -1.0).unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidGuidanceScale { .. }
        ));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // phi_rescale
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_phi_rescale_phi_zero_equals_cfg() {
        let cond = vec![2.0f32, 4.0, 6.0];
        let uncond = vec![1.0f32, 2.0, 3.0];
        let cfg = apply_cfg_guidance(&cond, &uncond, 7.5).unwrap();
        let rescaled = phi_rescale(&cond, &uncond, 7.5, 0.0).unwrap();
        for (r, c) in rescaled.iter().zip(cfg.iter()) {
            assert!(approx_eq(*r, *c, 1e-5));
        }
    }

    #[test]
    fn test_phi_rescale_phi_one_matches_std() {
        let cond: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let uncond: Vec<f32> = vec![0.0; 20];
        let result = phi_rescale(&cond, &uncond, 7.5, 1.0).unwrap();
        // With phi=1, result std should be close to std(cond)
        let std_cond = std_dev(&cond);
        let std_result = std_dev(&result);
        assert!(approx_eq(std_cond, std_result, 0.05));
    }

    #[test]
    fn test_phi_rescale_invalid_phi() {
        let cond = vec![1.0f32, 2.0];
        let uncond = vec![0.5f32, 0.5];
        let err = phi_rescale(&cond, &uncond, 7.5, 1.5).unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidRescaleFactor { .. }
        ));
    }

    #[test]
    fn test_phi_rescale_phi_negative_invalid() {
        let cond = vec![1.0f32, 2.0];
        let uncond = vec![0.5f32, 0.5];
        let err = phi_rescale(&cond, &uncond, 7.5, -0.1).unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidRescaleFactor { .. }
        ));
    }

    #[test]
    fn test_phi_rescale_midpoint_phi() {
        // phi=0.5 should produce a result between standard CFG and full rescaling
        let cond: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let uncond: Vec<f32> = vec![0.0; 16];
        let result = phi_rescale(&cond, &uncond, 5.0, 0.5).unwrap();
        assert_eq!(result.len(), 16);
        // All values finite
        assert!(result.iter().all(|v| v.is_finite()));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // dynamic_threshold
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dynamic_threshold_within_range_unchanged() {
        // All values in [-0.5, 0.5]; percentile s > 1, so clamp is at s not 0.5
        let latents: Vec<f32> = (0..10).map(|i| (i as f32 - 5.0) * 0.1).collect();
        let result = dynamic_threshold(&latents, 99.0).unwrap();
        // With s >= 1, values in [-0.5, 0.5] are untouched, then divided by s
        // so they should be smaller in magnitude
        assert!(result.iter().all(|v| v.abs() <= 1.0));
    }

    #[test]
    fn test_dynamic_threshold_large_values_clamped() {
        // Values far exceeding 1 should be clamped to ±1 after thresholding
        let latents: Vec<f32> = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        let result = dynamic_threshold(&latents, 50.0).unwrap();
        assert!(result.iter().all(|v| v.abs() <= 1.0 + 1e-6));
    }

    #[test]
    fn test_dynamic_threshold_result_in_minus_one_plus_one() {
        let latents: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 2.0).collect();
        let result = dynamic_threshold(&latents, 95.0).unwrap();
        assert!(result
            .iter()
            .all(|v| (-1.0 - 1e-6..=1.0 + 1e-6).contains(v)));
    }

    #[test]
    fn test_dynamic_threshold_invalid_percentile_zero() {
        let latents = vec![1.0f32, 2.0];
        let err = dynamic_threshold(&latents, 0.0).unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidPercentile { .. }
        ));
    }

    #[test]
    fn test_dynamic_threshold_invalid_percentile_hundred() {
        let latents = vec![1.0f32, 2.0];
        let err = dynamic_threshold(&latents, 100.0).unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidPercentile { .. }
        ));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // abs_percentile
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_abs_percentile_near_zero_returns_min() {
        let values = vec![-3.0f32, -1.0, 0.0, 1.0, 3.0];
        // p very small → returns something near the minimum absolute value
        let v = abs_percentile(&values, 1.0).unwrap();
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    fn test_abs_percentile_near_hundred_returns_max() {
        let values = vec![-3.0f32, -1.0, 0.0, 1.0, 3.0];
        let v = abs_percentile(&values, 99.0).unwrap();
        // Should be close to 3.0 (the maximum absolute value)
        assert!(v >= 2.5);
    }

    #[test]
    fn test_abs_percentile_sorted_correctly() {
        // Sorted abs: [0, 1, 1, 2, 2, 3, 3] — median (50th) should be 2
        let values = vec![3.0f32, -3.0, 2.0, -2.0, 1.0, -1.0, 0.0];
        let v = abs_percentile(&values, 50.0).unwrap();
        assert!(approx_eq(v, 2.0, 0.1));
    }

    #[test]
    fn test_abs_percentile_invalid_percentile() {
        let values = vec![1.0f32, 2.0];
        let err = abs_percentile(&values, 0.0).unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidPercentile { .. }
        ));
    }

    #[test]
    fn test_abs_percentile_nan_input_does_not_panic() {
        // Regression test: `partial_cmp(...).unwrap_or(Equal)` is not a
        // total order when NaN is present (Rust >= 1.81's sort can panic
        // on that); `total_cmp` must never panic here regardless of the
        // percentile requested.
        let values = vec![1.0f32, f32::NAN, -2.0, 3.0, f32::NAN];
        let result = abs_percentile(&values, 50.0);
        assert!(result.is_ok());
        // The result may legitimately be NaN if the percentile lands on a
        // NaN entry, but the call itself must not panic and must not error.
    }

    // ═════════════════════════════════════════════════════════════════════════
    // l2_normalize
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_l2_normalize_unit_vector_unchanged() {
        let v = vec![1.0f32, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!(approx_eq(n[0], 1.0, 1e-6));
        assert!(approx_eq(n[1], 0.0, 1e-6));
        assert!(approx_eq(n[2], 0.0, 1e-6));
    }

    #[test]
    fn test_l2_normalize_zero_vector_stays_zero() {
        let v = vec![0.0f32, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_l2_normalize_arbitrary_vector() {
        let v = vec![3.0f32, 4.0];
        let n = l2_normalize(&v);
        let norm = l2_norm_vec(&n);
        assert!(approx_eq(norm, 1.0, 1e-6));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // std_dev
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_std_dev_constant_is_zero() {
        let v = vec![5.0f32; 10];
        assert!(approx_eq(std_dev(&v), 0.0, 1e-6));
    }

    #[test]
    fn test_std_dev_known_distribution() {
        // Values [0, 1, 2, 3, 4] → mean=2, variance=2, std=√2≈1.4142
        let v = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let s = std_dev(&v);
        assert!(approx_eq(s, 2.0f32.sqrt(), 1e-5));
    }

    #[test]
    fn test_std_dev_empty_returns_zero() {
        assert_eq!(std_dev(&[]), 0.0);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // adaptive_guidance_scale
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_guidance_target_equals_current() {
        let cond: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let uncond: Vec<f32> = vec![0.0; 10];
        // Compute the current std of CFG output at scale=1.0
        let cfg_plain = apply_cfg_guidance(&cond, &uncond, 1.0).unwrap();
        let current_std = std_dev(&cfg_plain);
        // Ask adaptive to target exactly that std — should leave scale unchanged
        let (result, eff_scale) =
            adaptive_guidance_scale(&cond, &uncond, current_std, 1.0).unwrap();
        assert!(approx_eq(eff_scale, 1.0, 0.01));
        let result_std = std_dev(&result);
        assert!(approx_eq(result_std, current_std, 0.01));
    }

    #[test]
    fn test_adaptive_guidance_target_larger_than_current() {
        let cond: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let uncond: Vec<f32> = vec![0.0; 20];
        let cfg_plain = apply_cfg_guidance(&cond, &uncond, 2.0).unwrap();
        let current_std = std_dev(&cfg_plain);
        let target_std = current_std * 2.0;
        let (result, eff) = adaptive_guidance_scale(&cond, &uncond, target_std, 2.0).unwrap();
        let result_std = std_dev(&result);
        assert!(approx_eq(result_std, target_std, 0.01));
        assert!(eff > 2.0);
    }

    #[test]
    fn test_adaptive_guidance_zero_std_returns_unchanged() {
        // All identical → std=0 → should return cfg unchanged
        let cond = vec![1.0f32; 5];
        let uncond = vec![1.0f32; 5];
        let (result, eff_scale) = adaptive_guidance_scale(&cond, &uncond, 2.0, 7.5).unwrap();
        assert!(approx_eq(eff_scale, 7.5, 1e-5));
        // result should be the standard CFG
        let expected = apply_cfg_guidance(&cond, &uncond, 7.5).unwrap();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(approx_eq(*r, *e, 1e-5));
        }
    }

    #[test]
    fn test_adaptive_guidance_output_length() {
        let cond: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let uncond: Vec<f32> = vec![0.0; 4];
        let (result, _) = adaptive_guidance_scale(&cond, &uncond, 1.0, 5.0).unwrap();
        assert_eq!(result.len(), 4);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // compute_guidance_stats
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_compute_guidance_stats_basic() {
        let cond = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let uncond = vec![0.0f32; 5];
        let stats = compute_guidance_stats(&cond, &uncond, 7.5, 0.7).unwrap();
        assert!(stats.cfg_norm > 0.0);
        assert!(stats.cond_std > 0.0);
        assert_eq!(stats.uncond_std, 0.0);
    }

    #[test]
    fn test_compute_guidance_stats_scale_factor_preserved() {
        let cond = vec![1.0f32, 2.0];
        let uncond = vec![0.0f32; 2];
        let stats = compute_guidance_stats(&cond, &uncond, 5.0, 0.3).unwrap();
        assert!(approx_eq(stats.scale_factor, 0.3, 1e-6));
    }

    #[test]
    fn test_compute_guidance_stats_empty_error() {
        let err = compute_guidance_stats(&[], &[], 5.0, 0.7).unwrap_err();
        assert!(matches!(err, GuidanceRescalingError::EmptyPredictions));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // apply_rescaled_guidance
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_apply_rescaled_guidance_default_config() {
        let cond: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let uncond: Vec<f32> = vec![0.0; 16];
        let config = RescalingConfig::default();
        let result = apply_rescaled_guidance(&cond, &uncond, &config).unwrap();
        assert_eq!(result.len(), 16);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_apply_rescaled_guidance_no_rescaling() {
        let cond = vec![2.0f32, 4.0, 6.0];
        let uncond = vec![1.0f32, 2.0, 3.0];
        let config = RescalingConfig {
            guidance_scale: 2.0,
            use_phi_rescaling: false,
            use_dynamic_thresholding: false,
            clamp_range: None,
            ..Default::default()
        };
        let result = apply_rescaled_guidance(&cond, &uncond, &config).unwrap();
        let expected = apply_cfg_guidance(&cond, &uncond, 2.0).unwrap();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(approx_eq(*r, *e, 1e-5));
        }
    }

    #[test]
    fn test_apply_rescaled_guidance_phi_rescaling_matches_standalone_phi_rescale() {
        // Regression test for the CFG-computed-twice-then-discarded bug: the
        // in-place fast path used inside apply_rescaled_guidance must still
        // produce the exact same result as calling phi_rescale directly.
        let cond: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.3).collect();
        let uncond: Vec<f32> = vec![0.0; 16];
        let config = RescalingConfig {
            guidance_scale: 6.0,
            rescale_factor: 0.6,
            use_phi_rescaling: true,
            use_dynamic_thresholding: false,
            clamp_range: None,
            ..Default::default()
        };
        let result = apply_rescaled_guidance(&cond, &uncond, &config).unwrap();
        let expected = phi_rescale(&cond, &uncond, config.guidance_scale, config.rescale_factor)
            .expect("phi_rescale failed");
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(approx_eq(*r, *e, 1e-4), "{} vs {}", r, e);
        }
    }

    #[test]
    fn test_apply_rescaled_guidance_with_dynamic_threshold() {
        let cond: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 5.0).collect();
        let uncond: Vec<f32> = vec![0.0; 20];
        let config = RescalingConfig {
            guidance_scale: 7.5,
            use_phi_rescaling: false,
            use_dynamic_thresholding: true,
            ..Default::default()
        };
        let result = apply_rescaled_guidance(&cond, &uncond, &config).unwrap();
        assert!(result.iter().all(|v| v.abs() <= 1.0 + 1e-5));
    }

    #[test]
    fn test_apply_rescaled_guidance_with_clamp() {
        let cond: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let uncond: Vec<f32> = vec![0.0; 10];
        let config = RescalingConfig {
            guidance_scale: 3.0,
            use_phi_rescaling: false,
            use_dynamic_thresholding: false,
            clamp_range: Some((-1.0, 1.0)),
            ..Default::default()
        };
        let result = apply_rescaled_guidance(&cond, &uncond, &config).unwrap();
        assert!(result
            .iter()
            .all(|v| (-1.0 - 1e-6..=1.0 + 1e-6).contains(v)));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // box_blur_guidance
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_box_blur_uniform_image_unchanged() {
        // A uniform image blurred should remain uniform
        let w = 4u32;
        let h = 4u32;
        let c = 3u32;
        let image = vec![0.5f32; (w * h * c) as usize];
        let blurred = box_blur_guidance(&image, w, h, c, 1).unwrap();
        for v in &blurred {
            assert!(approx_eq(*v, 0.5, 1e-5));
        }
    }

    #[test]
    fn test_box_blur_zero_radius_identity() {
        let w = 3u32;
        let h = 3u32;
        let c = 1u32;
        let image: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let blurred = box_blur_guidance(&image, w, h, c, 0).unwrap();
        for (b, o) in blurred.iter().zip(image.iter()) {
            assert!(approx_eq(*b, *o, 1e-5));
        }
    }

    #[test]
    fn test_box_blur_length_preserved() {
        let w = 5u32;
        let h = 5u32;
        let c = 2u32;
        let image = vec![1.0f32; (w * h * c) as usize];
        let blurred = box_blur_guidance(&image, w, h, c, 2).unwrap();
        assert_eq!(blurred.len(), image.len());
    }

    #[test]
    fn test_box_blur_matches_naive_definition() {
        // Cross-check the separable rewrite against the original
        // O(w*h*c*radius^2) nested-loop definition for a small non-uniform
        // image, so the separable implementation is verified correct (not
        // just "runs without panicking").
        let w = 5usize;
        let h = 4usize;
        let c = 2usize;
        let image: Vec<f32> = (0..(w * h * c)).map(|i| (i as f32 * 0.37).sin()).collect();
        let radius = 2usize;

        let mut naive = vec![0.0f32; w * h * c];
        let r = radius as i64;
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for dy in -r..=r {
                        let ny = (y + dy).clamp(0, h as i64 - 1) as usize;
                        for dx in -r..=r {
                            let nx = (x + dx).clamp(0, w as i64 - 1) as usize;
                            sum += image[(ny * w + nx) * c + ch];
                            count += 1;
                        }
                    }
                    naive[(y as usize * w + x as usize) * c + ch] = sum / count as f32;
                }
            }
        }

        let blurred =
            box_blur_guidance(&image, w as u32, h as u32, c as u32, radius).expect("blur failed");
        for (a, b) in blurred.iter().zip(naive.iter()) {
            assert!((a - b).abs() < 1e-4, "{} vs {} (naive)", a, b);
        }
    }

    #[test]
    fn test_box_blur_huge_radius_does_not_overflow_or_hang() {
        // Regression test: `radius = usize::MAX` used to cast to `r = -1`,
        // making `-r..=r` the empty range `1..=-1`, so `count` stayed 0 and
        // every output became `0.0 / 0.0 = NaN`. The radius must be capped
        // to the image extent instead of overflowing.
        let w = 4u32;
        let h = 4u32;
        let c = 1u32;
        let image: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let blurred =
            box_blur_guidance(&image, w, h, c, usize::MAX).expect("blur with huge radius failed");
        assert_eq!(blurred.len(), 16);
        assert!(
            blurred
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.6),
            "got NaN/Inf/out-of-range: {:?}",
            blurred
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // self_attention_guidance
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_sag_zero_scale_is_identity() {
        let pred: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect();
        let result = self_attention_guidance(&pred, 0.0, 1, 3, 3, 1).unwrap();
        for (r, p) in result.iter().zip(pred.iter()) {
            assert!(approx_eq(*r, *p, 1e-5));
        }
    }

    #[test]
    fn test_sag_nonzero_scale_modifies_output() {
        let pred = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = self_attention_guidance(&pred, 1.0, 1, 3, 3, 1).unwrap();
        // At least one element should differ from pred
        let any_different = result
            .iter()
            .zip(pred.iter())
            .any(|(r, p)| (r - p).abs() > 1e-6);
        assert!(any_different);
    }

    #[test]
    fn test_sag_empty_input_error() {
        let err = self_attention_guidance(&[], 1.0, 1, 3, 3, 1).unwrap_err();
        assert!(matches!(err, GuidanceRescalingError::EmptyPredictions));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // annealed_phi
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_annealed_phi_step_zero_returns_start() {
        let v = annealed_phi(1.0, 0.0, 0, 100);
        assert!(approx_eq(v, 1.0, 1e-6));
    }

    #[test]
    fn test_annealed_phi_step_total_returns_end() {
        let v = annealed_phi(1.0, 0.0, 100, 100);
        assert!(approx_eq(v, 0.0, 1e-6));
    }

    #[test]
    fn test_annealed_phi_midpoint() {
        let v = annealed_phi(1.0, 0.0, 50, 100);
        assert!(approx_eq(v, 0.5, 1e-6));
    }

    #[test]
    fn test_annealed_phi_zero_total_steps() {
        let v = annealed_phi(0.8, 0.2, 0, 0);
        assert!(approx_eq(v, 0.8, 1e-6));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // effective_scale_with_phi
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_effective_scale_phi_zero_returns_nominal() {
        let eff = effective_scale_with_phi(7.5, 1.0, 2.0, 0.0);
        assert!(approx_eq(eff, 7.5, 1e-5));
    }

    #[test]
    fn test_effective_scale_phi_one_adjusts_by_ratio() {
        // phi=1 → effective = nominal * (cond_std / cfg_std)
        let nominal = 7.5f32;
        let cond_std = 1.0f32;
        let cfg_std = 2.0f32;
        let eff = effective_scale_with_phi(nominal, cond_std, cfg_std, 1.0);
        let expected = nominal * (cond_std / cfg_std);
        assert!(approx_eq(eff, expected, 1e-5));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // RescalingConfig::validate
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_rescaling_config_valid_default() {
        assert!(RescalingConfig::default().validate().is_ok());
    }

    #[test]
    fn test_rescaling_config_invalid_scale() {
        let cfg = RescalingConfig {
            guidance_scale: -1.0,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidGuidanceScale { .. }
        ));
    }

    #[test]
    fn test_rescaling_config_invalid_rescale_factor() {
        let cfg = RescalingConfig {
            rescale_factor: 1.5,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidRescaleFactor { .. }
        ));
    }

    #[test]
    fn test_rescaling_config_invalid_percentile() {
        let cfg = RescalingConfig {
            dynamic_threshold_pct: 0.0,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidPercentile { .. }
        ));
    }

    #[test]
    fn test_rescaling_config_invalid_clamp_range_reversed() {
        let cfg = RescalingConfig {
            clamp_range: Some((1.0, -1.0)),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidClampRange { .. }
        ));
    }

    #[test]
    fn test_rescaling_config_invalid_clamp_range_nan() {
        let cfg = RescalingConfig {
            clamp_range: Some((f32::NAN, 1.0)),
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidClampRange { .. }
        ));
    }

    #[test]
    fn test_apply_rescaled_guidance_reversed_clamp_range_errors_not_panics() {
        // Regression test: an invalid clamp_range must be rejected by
        // config validation instead of reaching `f32::clamp` and panicking.
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![0.0f32; 3];
        let config = RescalingConfig {
            clamp_range: Some((1.0, -1.0)),
            ..Default::default()
        };
        let err = apply_rescaled_guidance(&cond, &uncond, &config).unwrap_err();
        assert!(matches!(
            err,
            GuidanceRescalingError::InvalidClampRange { .. }
        ));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // l2_norm_vec
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_l2_norm_vec_known_value() {
        // 3-4-5 right triangle
        let v = vec![3.0f32, 4.0];
        assert!(approx_eq(l2_norm_vec(&v), 5.0, 1e-5));
    }

    #[test]
    fn test_l2_norm_vec_empty() {
        assert_eq!(l2_norm_vec(&[]), 0.0);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Edge cases / integration
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_full_pipeline_all_options_enabled() {
        let cond: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
        let uncond: Vec<f32> = vec![0.0; 32];
        let config = RescalingConfig {
            guidance_scale: 7.5,
            rescale_factor: 0.7,
            dynamic_threshold_pct: 95.0,
            use_phi_rescaling: true,
            use_dynamic_thresholding: true,
            clamp_range: Some((-1.0, 1.0)),
        };
        let result = apply_rescaled_guidance(&cond, &uncond, &config).unwrap();
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|v| v.is_finite()));
        assert!(result
            .iter()
            .all(|v| (-1.0 - 1e-6..=1.0 + 1e-6).contains(v)));
    }

    #[test]
    fn test_annealed_phi_monotone() {
        let total = 50usize;
        let phi_start = 1.0f32;
        let phi_end = 0.0f32;
        let values: Vec<f32> = (0..=total)
            .map(|s| annealed_phi(phi_start, phi_end, s, total))
            .collect();
        // Should be strictly non-increasing
        for w in values.windows(2) {
            assert!(w[0] >= w[1] - 1e-6);
        }
    }
}
