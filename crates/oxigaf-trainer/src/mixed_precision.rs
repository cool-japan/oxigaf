//! Mixed precision training support for the OxiGAF optimization loop.
//!
//! Provides:
//! - [`TrainingPrecision`] — precision mode selector (FP32 / BF16 / FP16).
//! - [`LossScaler`] — dynamic loss scaler with overflow detection and
//!   scale-window–based automatic scale adjustment.
//! - [`LossScalerStats`] — snapshot of scaler statistics.
//! - [`MixedPrecisionTrainer`] — orchestrates precision selection, loss
//!   scaling, gradient unscaling, and overflow detection in one place.
//!
//! The implementation follows the approach used in PyTorch's `GradScaler`:
//! on each step the loss is scaled up before the backward pass (handled
//! externally), the gradients are then divided by the scale before the
//! optimizer step, any NaN/Inf in the gradients signals an overflow, and the
//! scale is adapted accordingly.

use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TrainingPrecision
// ---------------------------------------------------------------------------

/// Precision mode for training.
///
/// `BFloat16` and `Float16` both use dynamic loss scaling; `Float32` uses a
/// fixed scale of `1.0` (i.e., no scaling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingPrecision {
    /// Full 32-bit floating point — no loss scaling required.
    Float32,
    /// Brain float 16 — more numerically stable than FP16 due to wider
    /// exponent range; lower initial scale recommended.
    BFloat16,
    /// IEEE 754 half precision — narrower dynamic range; higher initial scale
    /// is required to avoid underflow.
    Float16,
}

impl TrainingPrecision {
    /// Human-readable label for display output.
    pub fn label(self) -> &'static str {
        match self {
            TrainingPrecision::Float32 => "FP32",
            TrainingPrecision::BFloat16 => "BF16",
            TrainingPrecision::Float16 => "FP16",
        }
    }

    /// Whether this precision mode requires loss scaling.
    pub fn requires_scaling(self) -> bool {
        match self {
            TrainingPrecision::Float32 => false,
            TrainingPrecision::BFloat16 | TrainingPrecision::Float16 => true,
        }
    }
}

// ---------------------------------------------------------------------------
// LossScalerStats
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of [`LossScaler`] state, suitable for logging.
#[derive(Debug, Clone)]
pub struct LossScalerStats {
    /// Current loss scale value.
    pub current_scale: f32,
    /// Total number of optimizer steps attempted (overflowed + succeeded).
    pub total_steps: u64,
    /// Number of steps that were skipped due to gradient overflow.
    pub overflow_count: u64,
    /// Fraction of steps that overflowed: `overflow_count / total_steps`.
    ///
    /// `0.0` when no steps have been taken.
    pub overflow_rate: f64,
}

// ---------------------------------------------------------------------------
// LossScaler
// ---------------------------------------------------------------------------

/// Dynamic loss scaler for FP16/BF16 training stability.
///
/// ### Algorithm
///
/// 1. Before the backward pass, multiply the scalar loss by [`scale`](LossScaler::scale).
/// 2. After the backward pass, call [`unscale_gradients`](LossScaler::unscale_gradients).
/// 3. Check for overflow with [`has_overflow`](LossScaler::has_overflow).
/// 4. Call [`update`](LossScaler::update) with the overflow flag.
///    - On overflow: scale is halved (down to `min_scale`), success counter resets.
///    - On success: success counter increments; when it reaches `scale_window`
///      the scale is doubled (up to `max_scale`) and the counter resets.
/// 5. If there was no overflow, apply the optimizer step.
#[derive(Debug, Clone)]
pub struct LossScaler {
    /// Current loss scale value.
    scale: f32,
    /// Multiplicative factor applied when increasing or decreasing the scale.
    scale_factor: f32,
    /// Number of consecutive successful steps before the scale is increased.
    scale_window: u32,
    /// Minimum allowed scale value.
    min_scale: f32,
    /// Maximum allowed scale value.
    max_scale: f32,
    /// Number of consecutive steps without overflow.
    consecutive_successes: u32,
    /// Total optimizer steps attempted (both overflowed and successful).
    total_steps: u64,
    /// Total steps that experienced gradient overflow.
    overflow_count: u64,
}

impl LossScaler {
    /// Create a new scaler with the specified initial scale.
    ///
    /// Defaults:
    /// - `scale_factor = 2.0`
    /// - `scale_window = 2000`
    /// - `min_scale = 1.0`
    /// - `max_scale = 65536.0`
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            scale_factor: 2.0,
            scale_window: 2000,
            min_scale: 1.0,
            max_scale: 65536.0,
            consecutive_successes: 0,
            total_steps: 0,
            overflow_count: 0,
        }
    }

    /// Return the current loss scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Multiply every element of `grads` by the current scale.
    ///
    /// Call this on the scalar loss *before* the backward pass, or equivalently
    /// on the gradients immediately after accumulation.
    pub fn scale_gradients(&self, grads: &mut [f32]) {
        let s = self.scale;
        for g in grads.iter_mut() {
            *g *= s;
        }
    }

    /// Divide every element of `grads` by the current scale.
    ///
    /// Call this *after* the backward pass to recover the true gradients.
    pub fn unscale_gradients(&self, grads: &mut [f32]) {
        let inv = 1.0 / self.scale;
        for g in grads.iter_mut() {
            *g *= inv;
        }
    }

    /// Return `true` if any gradient value is NaN or infinite.
    ///
    /// This is a static method that takes a gradient slice by reference.
    pub fn has_overflow(grads: &[f32]) -> bool {
        grads.iter().any(|&g| !g.is_finite())
    }

    /// Update the scale after an optimizer step.
    ///
    /// - `had_overflow = true`: scale is divided by `scale_factor` (clamped to
    ///   `min_scale`), and the consecutive-success counter resets.
    /// - `had_overflow = false`: consecutive-success counter increments; once
    ///   it reaches `scale_window`, scale is multiplied by `scale_factor`
    ///   (clamped to `max_scale`) and the counter resets.
    pub fn update(&mut self, had_overflow: bool) {
        self.total_steps += 1;
        if had_overflow {
            self.overflow_count += 1;
            self.scale = (self.scale / self.scale_factor).max(self.min_scale);
            self.consecutive_successes = 0;
        } else {
            self.consecutive_successes += 1;
            if self.consecutive_successes >= self.scale_window {
                self.scale = (self.scale * self.scale_factor).min(self.max_scale);
                self.consecutive_successes = 0;
            }
        }
    }

    /// Return a snapshot of the current scaler statistics.
    pub fn stats(&self) -> LossScalerStats {
        let overflow_rate = if self.total_steps == 0 {
            0.0
        } else {
            self.overflow_count as f64 / self.total_steps as f64
        };
        LossScalerStats {
            current_scale: self.scale,
            total_steps: self.total_steps,
            overflow_count: self.overflow_count,
            overflow_rate,
        }
    }

    /// Override the scale factor (multiplicative).  Default is `2.0`.
    pub fn with_scale_factor(mut self, factor: f32) -> Self {
        self.scale_factor = factor;
        self
    }

    /// Override the scale window (consecutive successes before scaling up).
    /// Default is `2000`.
    pub fn with_scale_window(mut self, window: u32) -> Self {
        self.scale_window = window;
        self
    }

    /// Override the minimum allowed scale.  Default is `1.0`.
    pub fn with_min_scale(mut self, min: f32) -> Self {
        self.min_scale = min;
        self
    }

    /// Override the maximum allowed scale.  Default is `65536.0`.
    pub fn with_max_scale(mut self, max: f32) -> Self {
        self.max_scale = max;
        self
    }
}

impl Default for LossScaler {
    /// Default scaler uses an initial scale of `65536.0`, matching PyTorch's
    /// `GradScaler` default.
    fn default() -> Self {
        Self::new(65536.0)
    }
}

// ---------------------------------------------------------------------------
// MixedPrecisionTrainer
// ---------------------------------------------------------------------------

/// Orchestrates mixed-precision training: precision selection, loss scaling,
/// gradient unscaling, and overflow detection.
///
/// ### Typical Usage
///
/// ```rust,no_run
/// # use oxigaf_trainer::mixed_precision::{MixedPrecisionTrainer, TrainingPrecision};
/// let mut trainer = MixedPrecisionTrainer::float16();
/// let mut grads = vec![0.5_f32; 1024];
///
/// // After backward pass:
/// let should_step = trainer.step(&mut grads);
/// if should_step {
///     // Apply optimizer step with unscaled, valid gradients.
/// }
/// println!("{}", trainer.format_stats());
/// ```
pub struct MixedPrecisionTrainer {
    /// Active precision mode.
    pub precision: TrainingPrecision,
    /// Dynamic loss scaler.
    pub scaler: LossScaler,
    /// Layer name patterns that must remain in FP32 (e.g. layer norm, softmax).
    pub fp32_layers: Vec<String>,
}

impl MixedPrecisionTrainer {
    /// Create a new trainer for the given precision.
    ///
    /// The initial scale is chosen automatically:
    /// - `Float32`  → `1.0`   (no scaling)
    /// - `BFloat16` → `1024.0` (stable)
    /// - `Float16`  → `65536.0` (full range)
    pub fn new(precision: TrainingPrecision) -> Self {
        let initial_scale = match precision {
            TrainingPrecision::Float32 => 1.0,
            TrainingPrecision::BFloat16 => 1024.0,
            TrainingPrecision::Float16 => 65536.0,
        };
        Self {
            precision,
            scaler: LossScaler::new(initial_scale),
            fp32_layers: Vec::new(),
        }
    }

    /// Create a full-precision (FP32) trainer with a no-op scaler (scale = 1.0).
    pub fn float32() -> Self {
        Self::new(TrainingPrecision::Float32)
    }

    /// Create a BFloat16 trainer with an initial scale of `1024.0`.
    pub fn bfloat16() -> Self {
        Self::new(TrainingPrecision::BFloat16)
    }

    /// Create an FP16 trainer with an initial scale of `65536.0`.
    pub fn float16() -> Self {
        Self::new(TrainingPrecision::Float16)
    }

    /// Perform a mixed-precision gradient processing step.
    ///
    /// Steps performed:
    /// 1. Unscale gradients (divide by current scale).
    /// 2. Check for NaN / Inf overflow.
    /// 3. Update the loss scaler.
    /// 4. Return `true` if the optimizer step should proceed (no overflow).
    ///
    /// The caller should only apply the optimizer update when this returns
    /// `true`.
    pub fn step(&mut self, grads: &mut [f32]) -> bool {
        self.scaler.unscale_gradients(grads);
        let overflow = LossScaler::has_overflow(grads);
        self.scaler.update(overflow);
        !overflow
    }

    /// Add a layer name pattern to the FP32-pinned list.
    pub fn pin_fp32_layer(&mut self, layer: impl Into<String>) {
        self.fp32_layers.push(layer.into());
    }

    /// Return `true` if the given layer name matches any pinned FP32 pattern.
    ///
    /// Matching is substring-based: if any entry in `fp32_layers` is contained
    /// in `layer_name` the layer is considered pinned.
    pub fn is_fp32_pinned(&self, layer_name: &str) -> bool {
        self.fp32_layers
            .iter()
            .any(|pattern| layer_name.contains(pattern.as_str()))
    }

    /// Format a one-line statistics summary.
    ///
    /// Format: `"precision=BF16 scale=1024 overflows=0 (0.00%)"`
    pub fn format_stats(&self) -> String {
        let stats = self.scaler.stats();
        let mut out = String::new();
        let _ = write!(
            out,
            "precision={} scale={} overflows={} ({:.2}%)",
            self.precision.label(),
            stats.current_scale as u64,
            stats.overflow_count,
            stats.overflow_rate * 100.0,
        );
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ LossScaler defaults

    #[test]
    fn test_loss_scaler_default() {
        let s = LossScaler::default();
        assert_eq!(s.scale(), 65536.0, "default scale must be 65536");
        let stats = s.stats();
        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.overflow_count, 0);
        assert_eq!(stats.overflow_rate, 0.0);
    }

    // ------------------------------------------------------------------ scale_gradients

    #[test]
    fn test_scale_gradients() {
        let s = LossScaler::new(4.0);
        let mut grads = vec![1.0_f32, 2.0, 3.0];
        s.scale_gradients(&mut grads);
        assert!((grads[0] - 4.0).abs() < 1e-6, "1.0 * 4 = 4.0");
        assert!((grads[1] - 8.0).abs() < 1e-6, "2.0 * 4 = 8.0");
        assert!((grads[2] - 12.0).abs() < 1e-6, "3.0 * 4 = 12.0");
    }

    // ------------------------------------------------------------------ unscale_gradients

    #[test]
    fn test_unscale_gradients() {
        let s = LossScaler::new(4.0);
        let mut grads = vec![4.0_f32, 8.0, 12.0];
        s.unscale_gradients(&mut grads);
        assert!((grads[0] - 1.0).abs() < 1e-6, "4.0 / 4 = 1.0");
        assert!((grads[1] - 2.0).abs() < 1e-6, "8.0 / 4 = 2.0");
        assert!((grads[2] - 3.0).abs() < 1e-6, "12.0 / 4 = 3.0");
    }

    // ------------------------------------------------------------------ has_overflow

    #[test]
    fn test_has_overflow_with_nan() {
        let grads = vec![1.0_f32, f32::NAN, 3.0];
        assert!(
            LossScaler::has_overflow(&grads),
            "NaN must be detected as overflow"
        );
    }

    #[test]
    fn test_has_overflow_with_inf() {
        let grads = vec![1.0_f32, f32::INFINITY, 3.0];
        assert!(
            LossScaler::has_overflow(&grads),
            "+Inf must be detected as overflow"
        );
        let grads_neg = vec![1.0_f32, f32::NEG_INFINITY, 3.0];
        assert!(
            LossScaler::has_overflow(&grads_neg),
            "-Inf must be detected as overflow"
        );
    }

    #[test]
    fn test_has_overflow_clean() {
        let grads = vec![0.0_f32, 1.0, -1.0, 1e-6, 1e6];
        assert!(
            !LossScaler::has_overflow(&grads),
            "clean gradients must not be detected as overflow"
        );
    }

    // ------------------------------------------------------------------ update — overflow path

    #[test]
    fn test_update_on_overflow_halves_scale() {
        let mut s = LossScaler::new(1024.0);
        s.update(true); // overflow
        assert_eq!(s.scale(), 512.0, "scale must halve on overflow");
        assert_eq!(s.stats().overflow_count, 1);
        assert_eq!(s.stats().total_steps, 1);
    }

    // ------------------------------------------------------------------ update — success path

    #[test]
    fn test_update_consecutive_success_doubles_scale() {
        // Use a tiny scale_window so the test runs fast.
        let mut s = LossScaler::new(128.0).with_scale_window(3);
        // Two successes — not yet at window.
        s.update(false);
        s.update(false);
        assert_eq!(
            s.scale(),
            128.0,
            "scale must not change before window reached"
        );
        // Third success — triggers scale-up.
        s.update(false);
        assert_eq!(
            s.scale(),
            256.0,
            "scale must double after scale_window successes"
        );
        assert_eq!(s.stats().overflow_count, 0);
    }

    // ------------------------------------------------------------------ clamping

    #[test]
    fn test_scale_clamped_to_max() {
        let mut s = LossScaler::new(32768.0).with_scale_window(1);
        // One success triggers scale-up: 32768 * 2 = 65536 (at max).
        s.update(false);
        assert_eq!(s.scale(), 65536.0, "scale must not exceed max_scale");
        // Another success: already at max, should stay.
        s.update(false);
        assert_eq!(s.scale(), 65536.0, "scale must stay at max_scale");
    }

    #[test]
    fn test_scale_clamped_to_min() {
        let mut s = LossScaler::new(2.0);
        // Overflow: 2.0 / 2.0 = 1.0 (at min).
        s.update(true);
        assert_eq!(s.scale(), 1.0, "scale must not drop below min_scale");
        // Another overflow: already at min.
        s.update(true);
        assert_eq!(s.scale(), 1.0, "scale must stay at min_scale");
    }

    // ------------------------------------------------------------------ MixedPrecisionTrainer

    #[test]
    fn test_mixed_precision_trainer_float16() {
        let t = MixedPrecisionTrainer::float16();
        assert_eq!(t.precision, TrainingPrecision::Float16);
        assert_eq!(t.scaler.scale(), 65536.0);
    }

    #[test]
    fn test_step_with_overflow_returns_false() {
        let mut t = MixedPrecisionTrainer::float16();
        // Gradients containing NaN will trigger overflow detection.
        let mut grads = vec![1.0_f32, f32::NAN, 3.0];
        // Unscale is applied first, then NaN is checked.
        let should_step = t.step(&mut grads);
        assert!(
            !should_step,
            "step must return false when gradients contain NaN"
        );
        assert_eq!(
            t.scaler.stats().overflow_count,
            1,
            "overflow_count must be incremented"
        );
    }

    #[test]
    fn test_step_without_overflow_returns_true() {
        let mut t = MixedPrecisionTrainer::float16();
        // All finite gradients → no overflow.
        let mut grads = vec![0.001_f32; 64];
        let should_step = t.step(&mut grads);
        assert!(
            should_step,
            "step must return true when gradients are clean"
        );
        assert_eq!(
            t.scaler.stats().overflow_count,
            0,
            "overflow_count must remain 0"
        );
    }

    #[test]
    fn test_format_stats() {
        let t = MixedPrecisionTrainer::bfloat16();
        let stats = t.format_stats();
        assert!(
            stats.contains("BF16"),
            "stats must contain precision label BF16"
        );
        assert!(
            stats.contains("1024"),
            "stats must contain the initial scale"
        );
        assert!(
            stats.contains("overflows=0"),
            "stats must contain overflow count"
        );
        assert!(
            stats.contains("0.00%"),
            "stats must contain overflow rate percentage"
        );
    }

    // ------------------------------------------------------------------ Additional edge cases

    #[test]
    fn test_float32_trainer_no_scaling() {
        let t = MixedPrecisionTrainer::float32();
        assert_eq!(t.precision, TrainingPrecision::Float32);
        assert_eq!(t.scaler.scale(), 1.0, "FP32 trainer must use scale = 1.0");
        assert!(!t.precision.requires_scaling());
    }

    #[test]
    fn test_bfloat16_initial_scale() {
        let t = MixedPrecisionTrainer::bfloat16();
        assert_eq!(
            t.scaler.scale(),
            1024.0,
            "BF16 must use initial scale 1024.0"
        );
        assert!(t.precision.requires_scaling());
    }

    #[test]
    fn test_overflow_rate_calculation() {
        let mut s = LossScaler::new(1024.0);
        s.update(true); // overflow
        s.update(false); // success
        s.update(false); // success
        s.update(true); // overflow
        let stats = s.stats();
        assert_eq!(stats.total_steps, 4);
        assert_eq!(stats.overflow_count, 2);
        let expected_rate = 2.0 / 4.0;
        assert!(
            (stats.overflow_rate - expected_rate).abs() < 1e-9,
            "overflow rate must be 0.5, got {}",
            stats.overflow_rate
        );
    }

    #[test]
    fn test_fp32_pinned_layers() {
        let mut t = MixedPrecisionTrainer::float16();
        t.pin_fp32_layer("layer_norm");
        t.pin_fp32_layer("softmax");
        assert!(t.is_fp32_pinned("model.encoder.layer_norm.weight"));
        assert!(t.is_fp32_pinned("output.softmax.bias"));
        assert!(!t.is_fp32_pinned("model.attention.query.weight"));
    }

    #[test]
    fn test_scale_and_unscale_roundtrip() {
        let s = LossScaler::new(1024.0);
        let original = vec![0.1_f32, -0.5, 3.1, 0.0];
        let mut grads = original.clone();
        s.scale_gradients(&mut grads);
        s.unscale_gradients(&mut grads);
        for (a, b) in grads.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "round-trip must recover original gradient: {} != {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_consecutive_success_resets_after_overflow() {
        let mut s = LossScaler::new(1024.0).with_scale_window(5);
        // Three successes, then overflow.
        s.update(false);
        s.update(false);
        s.update(false);
        s.update(true); // overflow resets counter
                        // Three more successes — counter restarted, not at window yet.
        s.update(false);
        s.update(false);
        s.update(false);
        // Scale should NOT have been increased (window = 5, only 3 successes).
        // After overflow it dropped from 1024 to 512.
        assert_eq!(
            s.scale(),
            512.0,
            "scale must remain at post-overflow value after fewer successes than window"
        );
    }
}
