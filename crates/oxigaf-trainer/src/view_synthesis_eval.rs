//! Novel view synthesis evaluation module.
//!
//! Provides comprehensive quality metrics for comparing rendered images against
//! held-out ground truth views: PSNR, SSIM, MAE, MSE, and an approximate LPIPS
//! computed from Sobel gradient features.
//!
//! # Example
//! ```rust,ignore
//! use oxigaf_trainer::view_synthesis_eval::{EvalConfig, ViewSynthesisEvaluator};
//!
//! let config = EvalConfig::default();
//! let mut evaluator = ViewSynthesisEvaluator::new(config);
//! // predicted / ground_truth: Vec<Vec<f32>> each of length H*W*3
//! let metrics = evaluator.evaluate(1000, &predicted, &ground_truth)?;
//! println!("{}", format_eval_metrics(&metrics));
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the view synthesis evaluation subsystem.
#[derive(Debug, Error)]
pub enum EvalError {
    #[error("Image size mismatch: predicted {pred_w}×{pred_h} vs ground truth {gt_w}×{gt_h}")]
    SizeMismatch {
        pred_w: usize,
        pred_h: usize,
        gt_w: usize,
        gt_h: usize,
    },

    #[error("Empty view set")]
    EmptyViews,

    #[error("View index {0} out of range")]
    ViewIndexOutOfRange(usize),

    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    #[error("Evaluation not started")]
    NotStarted,
}

// ---------------------------------------------------------------------------
// Metric structs
// ---------------------------------------------------------------------------

/// Quality metrics for a single rendered view.
#[derive(Debug, Clone)]
pub struct ViewMetrics {
    /// Index of the view in the evaluation set.
    pub view_id: usize,
    /// Peak Signal-to-Noise Ratio in dB.
    pub psnr: f32,
    /// Structural Similarity Index ∈ [-1, 1].
    pub ssim: f32,
    /// Mean Absolute Error.
    pub mae: f32,
    /// Mean Squared Error.
    pub mse: f32,
    /// Simplified LPIPS approximation (gradient-based).
    pub lpips_approx: f32,
    /// Render time in milliseconds (0.0 if not timed by the caller).
    pub render_time_ms: f32,
}

/// Aggregated evaluation metrics over multiple views.
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    pub mean_psnr: f32,
    pub median_psnr: f32,
    pub min_psnr: f32,
    pub max_psnr: f32,
    pub mean_ssim: f32,
    pub mean_mae: f32,
    pub mean_lpips_approx: f32,
    pub total_views: usize,
    pub completed_views: usize,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the view synthesis evaluator.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// Evaluate every N training steps.
    pub eval_interval: usize,
    /// Number of views to evaluate per evaluation call.
    pub n_eval_views: usize,
    /// Image width in pixels.
    pub image_width: usize,
    /// Image height in pixels.
    pub image_height: usize,
    /// Whether rendered images should be saved (handled externally).
    pub save_renders: bool,
    /// Patch size for the LPIPS approximation.
    pub lpips_patch_size: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            eval_interval: 1000,
            n_eval_views: 10,
            image_width: 512,
            image_height: 512,
            save_renders: false,
            lpips_patch_size: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// A single historical evaluation record.
#[derive(Debug, Clone)]
pub struct EvalRecord {
    /// Training step at which the evaluation was run.
    pub step: usize,
    /// Aggregated metrics for this evaluation.
    pub metrics: EvalMetrics,
    /// Per-view breakdown.
    pub per_view: Vec<ViewMetrics>,
}

/// Stateful evaluator that tracks evaluation history and the best result.
#[derive(Debug)]
pub struct ViewSynthesisEvaluator {
    config: EvalConfig,
    eval_history: Vec<EvalRecord>,
    best_psnr: f32,
    best_step: usize,
}

impl ViewSynthesisEvaluator {
    /// Create a new evaluator with the given configuration.
    pub fn new(config: EvalConfig) -> Self {
        Self {
            config,
            eval_history: Vec::new(),
            best_psnr: f32::NEG_INFINITY,
            best_step: 0,
        }
    }

    /// Returns `true` if an evaluation should be run at `step`.
    ///
    /// Evaluates when `step % eval_interval == 0` (and `step > 0`).
    pub fn should_eval(&self, step: usize) -> bool {
        step > 0 && step.is_multiple_of(self.config.eval_interval)
    }

    /// Evaluate novel view quality at a given training step.
    ///
    /// # Parameters
    /// - `step` — current training step (used for history).
    /// - `predicted` — slice of `n_views` flat RGB images (`H*W*3` f32 values each).
    /// - `ground_truth` — slice of `n_views` flat RGB images, same layout.
    pub fn evaluate(
        &mut self,
        step: usize,
        predicted: &[Vec<f32>],
        ground_truth: &[Vec<f32>],
    ) -> Result<EvalMetrics, EvalError> {
        if predicted.is_empty() {
            return Err(EvalError::EmptyViews);
        }
        if predicted.len() != ground_truth.len() {
            return Err(EvalError::InvalidParam(format!(
                "predicted has {} views but ground_truth has {}",
                predicted.len(),
                ground_truth.len()
            )));
        }

        let w = self.config.image_width;
        let h = self.config.image_height;
        let patch = self.config.lpips_patch_size;

        let mut per_view = Vec::with_capacity(predicted.len());
        for (view_id, (pred, gt)) in predicted.iter().zip(ground_truth.iter()).enumerate() {
            let vm = eval_view_metrics(view_id, pred, gt, w, h, patch)?;
            per_view.push(vm);
        }

        let metrics = eval_aggregate_metrics(&per_view);

        // Update best-PSNR tracking.
        if metrics.mean_psnr > self.best_psnr {
            self.best_psnr = metrics.mean_psnr;
            self.best_step = step;
        }

        let record = EvalRecord {
            step,
            metrics: metrics.clone(),
            per_view,
        };
        self.eval_history.push(record);

        Ok(metrics)
    }

    /// Return the record with the highest mean PSNR so far.
    pub fn best_metrics(&self) -> Option<&EvalRecord> {
        self.eval_history.iter().max_by(|a, b| {
            a.metrics
                .mean_psnr
                .partial_cmp(&b.metrics.mean_psnr)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the most recent evaluation record.
    pub fn latest_metrics(&self) -> Option<&EvalRecord> {
        self.eval_history.last()
    }

    /// Full evaluation history.
    pub fn history(&self) -> &[EvalRecord] {
        &self.eval_history
    }

    /// Returns `(step, mean_psnr)` pairs for all evaluations, in order.
    pub fn psnr_trend(&self) -> Vec<(usize, f32)> {
        self.eval_history
            .iter()
            .map(|r| (r.step, r.metrics.mean_psnr))
            .collect()
    }

    /// Returns `true` if the latest evaluation improved on the previous one.
    ///
    /// Requires at least two evaluations in the history.
    pub fn has_improved(&self) -> bool {
        if self.eval_history.len() < 2 {
            return false;
        }
        let n = self.eval_history.len();
        let latest = &self.eval_history[n - 1];
        let prev = &self.eval_history[n - 2];
        latest.metrics.mean_psnr > prev.metrics.mean_psnr
    }
}

// ---------------------------------------------------------------------------
// Pixel indexing helper
// ---------------------------------------------------------------------------

/// Index into a flat H×W×3 buffer: `(row, col, channel)`.
#[inline(always)]
fn pixel_idx(row: usize, col: usize, width: usize, channel: usize) -> usize {
    (row * width + col) * 3 + channel
}

// ---------------------------------------------------------------------------
// Core metric functions
// ---------------------------------------------------------------------------

/// Mean Squared Error between two equal-length buffers.
///
/// Returns `Err(EvalError::SizeMismatch)` if lengths differ.
pub fn eval_mse(predicted: &[f32], ground_truth: &[f32]) -> Result<f32, EvalError> {
    if predicted.len() != ground_truth.len() {
        let n = predicted.len();
        let m = ground_truth.len();
        // Surface as a generic SizeMismatch with dummy dimensions for flat arrays.
        return Err(EvalError::SizeMismatch {
            pred_w: n,
            pred_h: 1,
            gt_w: m,
            gt_h: 1,
        });
    }
    if predicted.is_empty() {
        return Err(EvalError::EmptyViews);
    }
    let sum: f32 = predicted
        .iter()
        .zip(ground_truth.iter())
        .map(|(p, g)| {
            let d = p - g;
            d * d
        })
        .sum();
    Ok(sum / predicted.len() as f32)
}

/// Mean Absolute Error between two equal-length buffers.
pub fn eval_mae(predicted: &[f32], ground_truth: &[f32]) -> Result<f32, EvalError> {
    if predicted.len() != ground_truth.len() {
        let n = predicted.len();
        let m = ground_truth.len();
        return Err(EvalError::SizeMismatch {
            pred_w: n,
            pred_h: 1,
            gt_w: m,
            gt_h: 1,
        });
    }
    if predicted.is_empty() {
        return Err(EvalError::EmptyViews);
    }
    let sum: f32 = predicted
        .iter()
        .zip(ground_truth.iter())
        .map(|(p, g)| (p - g).abs())
        .sum();
    Ok(sum / predicted.len() as f32)
}

/// PSNR (Peak Signal-to-Noise Ratio) in dB.
///
/// Assumes pixel values in `[0, 1]`. Returns `100.0` for effectively identical
/// images (MSE < 1e-10).
pub fn eval_psnr(predicted: &[f32], ground_truth: &[f32]) -> Result<f32, EvalError> {
    let mse = eval_mse(predicted, ground_truth)?;
    if mse < 1e-10 {
        return Ok(100.0);
    }
    Ok(10.0 * (1.0_f32 / mse).log10())
}

/// Simplified SSIM (Structural Similarity Index) using an 11×11 sliding window.
///
/// Uses Wang et al. (2004) constants adapted for `[0, 1]` pixel range:
/// `C1 = 0.0001`, `C2 = 0.0009`. Strides by 4 pixels for efficiency.
pub fn eval_ssim(
    predicted: &[f32],
    ground_truth: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, EvalError> {
    let expected = width * height * 3;
    if predicted.len() != expected {
        return Err(EvalError::SizeMismatch {
            pred_w: width,
            pred_h: height,
            gt_w: predicted.len() / 3,
            gt_h: 1,
        });
    }
    if ground_truth.len() != expected {
        return Err(EvalError::SizeMismatch {
            pred_w: width,
            pred_h: height,
            gt_w: ground_truth.len() / 3,
            gt_h: 1,
        });
    }
    if width == 0 || height == 0 {
        return Err(EvalError::InvalidParam(
            "image dimensions must be non-zero".into(),
        ));
    }

    // Constants for images in [0, 1].
    const C1: f32 = 0.0001; // (0.01)^2
    const C2: f32 = 0.0009; // (0.03)^2
    const WIN: usize = 11;
    const HALF: usize = WIN / 2;
    const STRIDE: usize = 4;

    let mut total_ssim = 0.0_f32;
    let mut n_windows = 0_usize;

    for ch in 0..3_usize {
        let mut row = HALF;
        while row + HALF < height {
            let mut col = HALF;
            while col + HALF < width {
                // Collect window pixels.
                let mut sum_x = 0.0_f32;
                let mut sum_y = 0.0_f32;
                let mut sum_xx = 0.0_f32;
                let mut sum_yy = 0.0_f32;
                let mut sum_xy = 0.0_f32;
                let n = (WIN * WIN) as f32;

                for wr in 0..WIN {
                    let r = row + wr - HALF;
                    for wc in 0..WIN {
                        let c = col + wc - HALF;
                        let x = predicted[pixel_idx(r, c, width, ch)];
                        let y = ground_truth[pixel_idx(r, c, width, ch)];
                        sum_x += x;
                        sum_y += y;
                        sum_xx += x * x;
                        sum_yy += y * y;
                        sum_xy += x * y;
                    }
                }

                let mu_x = sum_x / n;
                let mu_y = sum_y / n;
                let sigma_x2 = (sum_xx / n) - mu_x * mu_x;
                let sigma_y2 = (sum_yy / n) - mu_y * mu_y;
                let sigma_xy = (sum_xy / n) - mu_x * mu_y;

                // Wang et al. SSIM formula.
                let numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
                let denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2);

                total_ssim += numerator / denominator;
                n_windows += 1;

                col += STRIDE;
            }
            row += STRIDE;
        }
    }

    if n_windows == 0 {
        // Image too small for the 11×11 window — return 1.0 for identical, else compare directly.
        let mse = eval_mse(predicted, ground_truth)?;
        return Ok(if mse < 1e-10 { 1.0 } else { 0.0 });
    }

    Ok(total_ssim / n_windows as f32)
}

/// Compute Sobel gradient magnitude for a single-channel H×W image.
///
/// Returns a flat `H*W` buffer of gradient magnitudes.
fn sobel_magnitude(channel: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut mag = vec![0.0_f32; width * height];
    // Sobel kernels (3×3):
    //  Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    //  Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
    for row in 1..height.saturating_sub(1) {
        for col in 1..width.saturating_sub(1) {
            let p = |dr: isize, dc: isize| -> f32 {
                let r = (row as isize + dr) as usize;
                let c = (col as isize + dc) as usize;
                channel[r * width + c]
            };

            let gx = -p(-1, -1) + p(-1, 1) - 2.0 * p(0, -1) + 2.0 * p(0, 1) - p(1, -1) + p(1, 1);

            let gy = -p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2.0 * p(1, 0) + p(1, 1);

            mag[row * width + col] = (gx * gx + gy * gy).sqrt();
        }
    }
    mag
}

/// Extract a single channel from an H×W×3 interleaved buffer.
fn extract_channel(image: &[f32], width: usize, height: usize, ch: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(width * height);
    for r in 0..height {
        for c in 0..width {
            out.push(image[pixel_idx(r, c, width, ch)]);
        }
    }
    out
}

/// Simplified LPIPS approximation using per-channel Sobel gradient magnitudes.
///
/// `lpips_approx = mean(|grad_mag_pred - grad_mag_gt|)` averaged over all channels.
/// The `patch_size` parameter is accepted for API compatibility but not used in the
/// current gradient-based implementation (gradient computation naturally captures
/// multi-scale spatial structure).
pub fn eval_lpips_approx(
    predicted: &[f32],
    ground_truth: &[f32],
    width: usize,
    height: usize,
    _patch_size: usize,
) -> Result<f32, EvalError> {
    let expected = width * height * 3;
    if predicted.len() != expected || ground_truth.len() != expected {
        return Err(EvalError::SizeMismatch {
            pred_w: width,
            pred_h: height,
            gt_w: ground_truth.len() / 3,
            gt_h: 1,
        });
    }
    if width == 0 || height == 0 {
        return Err(EvalError::InvalidParam(
            "image dimensions must be non-zero".into(),
        ));
    }

    let n_pixels = width * height;
    let mut total = 0.0_f32;

    for ch in 0..3_usize {
        let pred_ch = extract_channel(predicted, width, height, ch);
        let gt_ch = extract_channel(ground_truth, width, height, ch);

        let pred_mag = sobel_magnitude(&pred_ch, width, height);
        let gt_mag = sobel_magnitude(&gt_ch, width, height);

        let channel_sum: f32 = pred_mag
            .iter()
            .zip(gt_mag.iter())
            .map(|(pm, gm)| (pm - gm).abs())
            .sum();

        total += channel_sum / n_pixels as f32;
    }

    Ok(total / 3.0)
}

/// Compute all quality metrics for a single view pair.
///
/// `render_time_ms` is initialised to `0.0`; callers that time their render
/// pipeline should fill it in after calling this function.
pub fn eval_view_metrics(
    view_id: usize,
    predicted: &[f32],
    ground_truth: &[f32],
    width: usize,
    height: usize,
    patch_size: usize,
) -> Result<ViewMetrics, EvalError> {
    let mse = eval_mse(predicted, ground_truth)?;
    let mae = eval_mae(predicted, ground_truth)?;
    let psnr = {
        if mse < 1e-10 {
            100.0
        } else {
            10.0 * (1.0_f32 / mse).log10()
        }
    };
    let ssim = eval_ssim(predicted, ground_truth, width, height)?;
    let lpips_approx = eval_lpips_approx(predicted, ground_truth, width, height, patch_size)?;

    Ok(ViewMetrics {
        view_id,
        psnr,
        ssim,
        mae,
        mse,
        lpips_approx,
        render_time_ms: 0.0,
    })
}

/// Aggregate per-view metrics into overall [`EvalMetrics`].
///
/// Returns zeroed metrics for an empty slice (callers should guard against this).
pub fn eval_aggregate_metrics(per_view: &[ViewMetrics]) -> EvalMetrics {
    if per_view.is_empty() {
        return EvalMetrics {
            mean_psnr: 0.0,
            median_psnr: 0.0,
            min_psnr: 0.0,
            max_psnr: 0.0,
            mean_ssim: 0.0,
            mean_mae: 0.0,
            mean_lpips_approx: 0.0,
            total_views: 0,
            completed_views: 0,
        };
    }

    let n = per_view.len();
    let nf = n as f32;

    let mean_psnr = per_view.iter().map(|v| v.psnr).sum::<f32>() / nf;
    let mean_ssim = per_view.iter().map(|v| v.ssim).sum::<f32>() / nf;
    let mean_mae = per_view.iter().map(|v| v.mae).sum::<f32>() / nf;
    let mean_lpips_approx = per_view.iter().map(|v| v.lpips_approx).sum::<f32>() / nf;

    let min_psnr = per_view
        .iter()
        .map(|v| v.psnr)
        .fold(f32::INFINITY, f32::min);
    let max_psnr = per_view
        .iter()
        .map(|v| v.psnr)
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute median without nan (sort a copy).
    let mut psnrs: Vec<f32> = per_view.iter().map(|v| v.psnr).collect();
    psnrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_psnr = if n.is_multiple_of(2) {
        (psnrs[n / 2 - 1] + psnrs[n / 2]) / 2.0
    } else {
        psnrs[n / 2]
    };

    EvalMetrics {
        mean_psnr,
        median_psnr,
        min_psnr,
        max_psnr,
        mean_ssim,
        mean_mae,
        mean_lpips_approx,
        total_views: n,
        completed_views: n,
    }
}

/// Format [`EvalMetrics`] as a human-readable string.
pub fn format_eval_metrics(metrics: &EvalMetrics) -> String {
    format!(
        "Views: {}/{} | PSNR: mean={:.2} median={:.2} min={:.2} max={:.2} dB | \
         SSIM: {:.4} | MAE: {:.5} | LPIPS≈: {:.5}",
        metrics.completed_views,
        metrics.total_views,
        metrics.mean_psnr,
        metrics.median_psnr,
        metrics.min_psnr,
        metrics.max_psnr,
        metrics.mean_ssim,
        metrics.mean_mae,
        metrics.mean_lpips_approx,
    )
}

/// Format [`ViewMetrics`] as a human-readable string.
pub fn format_view_metrics(metrics: &ViewMetrics) -> String {
    format!(
        "View {:>4} | PSNR={:.2} dB | SSIM={:.4} | MAE={:.5} | MSE={:.6} | \
         LPIPS≈={:.5} | time={:.2} ms",
        metrics.view_id,
        metrics.psnr,
        metrics.ssim,
        metrics.mae,
        metrics.mse,
        metrics.lpips_approx,
        metrics.render_time_ms,
    )
}

/// Compute a per-pixel absolute-difference error map.
///
/// The returned buffer has the same length as the inputs (H×W×3).
pub fn eval_error_map(predicted: &[f32], ground_truth: &[f32]) -> Result<Vec<f32>, EvalError> {
    if predicted.len() != ground_truth.len() {
        let n = predicted.len();
        let m = ground_truth.len();
        return Err(EvalError::SizeMismatch {
            pred_w: n,
            pred_h: 1,
            gt_w: m,
            gt_h: 1,
        });
    }
    if predicted.is_empty() {
        return Err(EvalError::EmptyViews);
    }
    let map = predicted
        .iter()
        .zip(ground_truth.iter())
        .map(|(p, g)| (p - g).abs())
        .collect();
    Ok(map)
}

/// Find the views with the worst and best PSNR in a per-view collection.
///
/// Returns `(worst, best)`. Both are `None` for an empty slice.
pub fn find_extreme_views(
    per_view: &[ViewMetrics],
) -> (Option<&ViewMetrics>, Option<&ViewMetrics>) {
    if per_view.is_empty() {
        return (None, None);
    }
    let mut worst = &per_view[0];
    let mut best = &per_view[0];
    for v in per_view.iter().skip(1) {
        if v.psnr < worst.psnr {
            worst = v;
        }
        if v.psnr > best.psnr {
            best = v;
        }
    }
    (Some(worst), Some(best))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_image(width: usize, height: usize, value: f32) -> Vec<f32> {
        vec![value; width * height * 3]
    }

    fn make_gradient_image(width: usize, height: usize) -> Vec<f32> {
        let mut img = Vec::with_capacity(width * height * 3);
        for r in 0..height {
            for c in 0..width {
                let v = (r * width + c) as f32 / (width * height) as f32;
                img.push(v);
                img.push(v * 0.5);
                img.push(1.0 - v);
            }
        }
        img
    }

    // ------------------------------------------------------------------
    // eval_mse
    // ------------------------------------------------------------------

    #[test]
    fn eval_mse_identical_is_zero() {
        let a = vec![0.3_f32; 300];
        assert!((eval_mse(&a, &a).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn eval_mse_known_value() {
        let a = vec![0.0_f32; 4];
        let b = vec![1.0_f32; 4];
        let mse = eval_mse(&a, &b).unwrap();
        assert!((mse - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eval_mse_partial_error() {
        // [0.0, 0.0] vs [1.0, 0.0] → MSE = 0.5
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        let mse = eval_mse(&a, &b).unwrap();
        assert!((mse - 0.5).abs() < 1e-6);
    }

    #[test]
    fn eval_mse_size_mismatch_errors() {
        let a = vec![0.0_f32; 3];
        let b = vec![0.0_f32; 4];
        assert!(matches!(
            eval_mse(&a, &b),
            Err(EvalError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn eval_mse_empty_errors() {
        assert!(matches!(eval_mse(&[], &[]), Err(EvalError::EmptyViews)));
    }

    // ------------------------------------------------------------------
    // eval_mae
    // ------------------------------------------------------------------

    #[test]
    fn eval_mae_identical_is_zero() {
        let a = vec![0.7_f32; 100];
        assert!((eval_mae(&a, &a).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn eval_mae_known_value() {
        let a = vec![0.0_f32; 4];
        let b = vec![0.5_f32; 4];
        let mae = eval_mae(&a, &b).unwrap();
        assert!((mae - 0.5).abs() < 1e-6);
    }

    #[test]
    fn eval_mae_mixed_signs() {
        let a = vec![-0.5_f32, 0.5];
        let b = vec![0.5_f32, -0.5];
        let mae = eval_mae(&a, &b).unwrap();
        assert!((mae - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eval_mae_size_mismatch() {
        let a = vec![0.0_f32; 5];
        let b = vec![0.0_f32; 6];
        assert!(matches!(
            eval_mae(&a, &b),
            Err(EvalError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn eval_mae_empty_errors() {
        assert!(matches!(eval_mae(&[], &[]), Err(EvalError::EmptyViews)));
    }

    // ------------------------------------------------------------------
    // eval_psnr
    // ------------------------------------------------------------------

    #[test]
    fn eval_psnr_identical_returns_100() {
        let a = vec![0.5_f32; 300];
        let psnr = eval_psnr(&a, &a).unwrap();
        assert!((psnr - 100.0).abs() < 1e-5, "expected 100, got {}", psnr);
    }

    #[test]
    fn eval_psnr_all_zeros_vs_all_ones() {
        let a = vec![0.0_f32; 300];
        let b = vec![1.0_f32; 300];
        let psnr = eval_psnr(&a, &b).unwrap();
        // MSE=1.0 → PSNR = 10*log10(1/1) = 0 dB
        assert!(psnr.is_finite(), "PSNR should be finite");
        assert!((psnr - 0.0).abs() < 1.0, "expected ~0 dB, got {}", psnr);
    }

    #[test]
    fn eval_psnr_known_mse() {
        // MSE = 0.01 → PSNR = 10*log10(100) = 20 dB
        let n = 100;
        let a = vec![0.0_f32; n];
        let b = vec![0.1_f32; n]; // each element differs by 0.1 → d^2=0.01 → MSE=0.01
        let psnr = eval_psnr(&a, &b).unwrap();
        let expected = 10.0 * (100.0_f32).log10(); // = 20 dB
        assert!(
            (psnr - expected).abs() < 0.01,
            "expected ~{} dB, got {}",
            expected,
            psnr
        );
    }

    #[test]
    fn eval_psnr_size_mismatch() {
        let a = vec![0.0_f32; 10];
        let b = vec![0.0_f32; 11];
        assert!(matches!(
            eval_psnr(&a, &b),
            Err(EvalError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn eval_psnr_increases_as_error_decreases() {
        let gt = vec![0.5_f32; 300];
        let bad: Vec<f32> = gt.iter().map(|v| v + 0.2).collect();
        let good: Vec<f32> = gt.iter().map(|v| v + 0.01).collect();
        let psnr_bad = eval_psnr(&bad, &gt).unwrap();
        let psnr_good = eval_psnr(&good, &gt).unwrap();
        assert!(
            psnr_good > psnr_bad,
            "better predictions should yield higher PSNR"
        );
    }

    // ------------------------------------------------------------------
    // eval_ssim
    // ------------------------------------------------------------------

    #[test]
    fn eval_ssim_identical_is_one() {
        let w = 64;
        let h = 64;
        let a = make_gradient_image(w, h);
        let ssim = eval_ssim(&a, &a, w, h).unwrap();
        assert!(
            (ssim - 1.0).abs() < 1e-3,
            "identical images should have SSIM≈1, got {}",
            ssim
        );
    }

    #[test]
    fn eval_ssim_inverted_less_than_identical() {
        let w = 32;
        let h = 32;
        let a = make_gradient_image(w, h);
        let b: Vec<f32> = a.iter().map(|v| 1.0 - v).collect();
        let ssim_identical = eval_ssim(&a, &a, w, h).unwrap();
        let ssim_inverted = eval_ssim(&a, &b, w, h).unwrap();
        assert!(
            ssim_inverted < ssim_identical,
            "inverted ({}) should be less than identical ({})",
            ssim_inverted,
            ssim_identical
        );
    }

    #[test]
    fn eval_ssim_different_images_in_range() {
        let w = 32;
        let h = 32;
        let a = make_image(w, h, 0.0);
        let b = make_image(w, h, 1.0);
        let ssim = eval_ssim(&a, &b, w, h).unwrap();
        assert!(
            (-1.0..=1.0).contains(&ssim),
            "SSIM must be in [-1, 1], got {}",
            ssim
        );
    }

    #[test]
    fn eval_ssim_size_mismatch_predicted() {
        // pass a buffer that is too short for the claimed dimensions
        let a = vec![0.5_f32; 10]; // too short for 32×32×3
        let b = make_image(32, 32, 0.5);
        assert!(matches!(
            eval_ssim(&a, &b, 32, 32),
            Err(EvalError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn eval_ssim_size_mismatch_gt() {
        let a = make_image(32, 32, 0.5);
        let b = vec![0.5_f32; 10]; // too short
        assert!(matches!(
            eval_ssim(&a, &b, 32, 32),
            Err(EvalError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn eval_ssim_small_image_still_works() {
        // 4×4 image — smaller than the 11×11 window, falls back to MSE-based result.
        let w = 4;
        let h = 4;
        let a = make_image(w, h, 0.5);
        let ssim = eval_ssim(&a, &a, w, h).unwrap();
        // Identical images → MSE=0 → returns 1.0
        assert!(
            (ssim - 1.0).abs() < 1e-5,
            "small identical → SSIM=1, got {}",
            ssim
        );
    }

    #[test]
    fn eval_ssim_medium_image() {
        let w = 64;
        let h = 48;
        let a = make_gradient_image(w, h);
        let b = make_gradient_image(w, h);
        let ssim = eval_ssim(&a, &b, w, h).unwrap();
        assert!((ssim - 1.0).abs() < 1e-3);
    }

    // ------------------------------------------------------------------
    // eval_lpips_approx
    // ------------------------------------------------------------------

    #[test]
    fn eval_lpips_approx_identical_is_zero() {
        let w = 32;
        let h = 32;
        let a = make_gradient_image(w, h);
        let val = eval_lpips_approx(&a, &a, w, h, 16).unwrap();
        assert!(val.abs() < 1e-6, "identical → LPIPS≈0, got {}", val);
    }

    #[test]
    fn eval_lpips_approx_different_is_positive() {
        let w = 32;
        let h = 32;
        let a = make_image(w, h, 0.0);
        let b = make_gradient_image(w, h);
        let val = eval_lpips_approx(&a, &b, w, h, 16).unwrap();
        assert!(val > 0.0, "different images → LPIPS>0, got {}", val);
    }

    #[test]
    fn eval_lpips_approx_larger_error_means_higher() {
        // Uniform-shift images have the same Sobel gradients, so use images with
        // genuinely different gradient structure: constant vs. gradient.
        let w = 32;
        let h = 32;
        // Identical → LPIPS ≈ 0
        let grad = make_gradient_image(w, h);
        let identical_lpips = eval_lpips_approx(&grad, &grad, w, h, 16).unwrap();
        // White vs gradient → Sobel differs strongly
        let white = make_image(w, h, 1.0);
        let diff_lpips = eval_lpips_approx(&white, &grad, w, h, 16).unwrap();
        assert!(
            diff_lpips > identical_lpips,
            "different gradient structures should give higher LPIPS ({} vs {})",
            diff_lpips,
            identical_lpips,
        );
    }

    #[test]
    fn eval_lpips_approx_size_mismatch() {
        let a = vec![0.0_f32; 10];
        let b = make_image(32, 32, 0.0);
        assert!(matches!(
            eval_lpips_approx(&a, &b, 32, 32, 16),
            Err(EvalError::SizeMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // eval_error_map
    // ------------------------------------------------------------------

    #[test]
    fn eval_error_map_known_values() {
        let a = vec![0.0_f32, 0.5, 1.0];
        let b = vec![1.0_f32, 0.5, 0.0];
        let map = eval_error_map(&a, &b).unwrap();
        assert_eq!(map.len(), 3);
        assert!((map[0] - 1.0).abs() < 1e-6);
        assert!((map[1] - 0.0).abs() < 1e-6);
        assert!((map[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eval_error_map_identical_is_zero() {
        let a = vec![0.3_f32; 100];
        let map = eval_error_map(&a, &a).unwrap();
        assert!(map.iter().all(|&v| v.abs() < 1e-7));
    }

    #[test]
    fn eval_error_map_size_mismatch() {
        let a = vec![0.0_f32; 5];
        let b = vec![0.0_f32; 6];
        assert!(matches!(
            eval_error_map(&a, &b),
            Err(EvalError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn eval_error_map_empty_errors() {
        assert!(matches!(
            eval_error_map(&[], &[]),
            Err(EvalError::EmptyViews)
        ));
    }

    #[test]
    fn eval_error_map_length_preserved() {
        let a: Vec<f32> = (0..60).map(|i| i as f32 / 60.0).collect();
        let b: Vec<f32> = (0..60).map(|i| (60 - i) as f32 / 60.0).collect();
        let map = eval_error_map(&a, &b).unwrap();
        assert_eq!(map.len(), 60);
    }

    // ------------------------------------------------------------------
    // eval_view_metrics
    // ------------------------------------------------------------------

    #[test]
    fn eval_view_metrics_all_fields_populated() {
        let w = 32;
        let h = 32;
        let pred = make_gradient_image(w, h);
        let gt = make_image(w, h, 0.5);
        let vm = eval_view_metrics(7, &pred, &gt, w, h, 16).unwrap();
        assert_eq!(vm.view_id, 7);
        assert!(vm.psnr.is_finite());
        assert!(vm.ssim.is_finite());
        assert!(vm.mae >= 0.0);
        assert!(vm.mse >= 0.0);
        assert!(vm.lpips_approx >= 0.0);
        assert_eq!(vm.render_time_ms, 0.0);
    }

    #[test]
    fn eval_view_metrics_identical_images() {
        let w = 32;
        let h = 32;
        let a = make_gradient_image(w, h);
        let vm = eval_view_metrics(0, &a, &a, w, h, 16).unwrap();
        assert!((vm.psnr - 100.0).abs() < 1e-4, "identical → PSNR=100");
        assert!((vm.mae - 0.0).abs() < 1e-7, "identical → MAE=0");
        assert!((vm.mse - 0.0).abs() < 1e-7, "identical → MSE=0");
    }

    // ------------------------------------------------------------------
    // eval_aggregate_metrics
    // ------------------------------------------------------------------

    #[test]
    fn eval_aggregate_single_view_matches() {
        let vm = ViewMetrics {
            view_id: 0,
            psnr: 25.0,
            ssim: 0.85,
            mae: 0.05,
            mse: 0.003,
            lpips_approx: 0.2,
            render_time_ms: 0.0,
        };
        let agg = eval_aggregate_metrics(&[vm]);
        assert!((agg.mean_psnr - 25.0).abs() < 1e-5);
        assert!((agg.median_psnr - 25.0).abs() < 1e-5);
        assert!((agg.min_psnr - 25.0).abs() < 1e-5);
        assert!((agg.max_psnr - 25.0).abs() < 1e-5);
        assert!((agg.mean_ssim - 0.85).abs() < 1e-5);
        assert!((agg.mean_mae - 0.05).abs() < 1e-5);
        assert!((agg.mean_lpips_approx - 0.2).abs() < 1e-5);
        assert_eq!(agg.total_views, 1);
        assert_eq!(agg.completed_views, 1);
    }

    #[test]
    fn eval_aggregate_multiple_views_mean_psnr() {
        let views: Vec<ViewMetrics> = (0..4)
            .map(|i| ViewMetrics {
                view_id: i,
                psnr: (i as f32 + 1.0) * 10.0, // 10, 20, 30, 40
                ssim: 0.9,
                mae: 0.1,
                mse: 0.01,
                lpips_approx: 0.1,
                render_time_ms: 0.0,
            })
            .collect();
        let agg = eval_aggregate_metrics(&views);
        // mean of [10, 20, 30, 40] = 25
        assert!((agg.mean_psnr - 25.0).abs() < 1e-4);
        assert!((agg.min_psnr - 10.0).abs() < 1e-4);
        assert!((agg.max_psnr - 40.0).abs() < 1e-4);
        // median of sorted [10, 20, 30, 40] = (20+30)/2 = 25
        assert!((agg.median_psnr - 25.0).abs() < 1e-4);
    }

    #[test]
    fn eval_aggregate_odd_count_median() {
        let psnrs = [10.0_f32, 20.0, 30.0];
        let views: Vec<ViewMetrics> = psnrs
            .iter()
            .enumerate()
            .map(|(i, &p)| ViewMetrics {
                view_id: i,
                psnr: p,
                ssim: 0.9,
                mae: 0.0,
                mse: 0.0,
                lpips_approx: 0.0,
                render_time_ms: 0.0,
            })
            .collect();
        let agg = eval_aggregate_metrics(&views);
        // median of [10, 20, 30] = 20
        assert!((agg.median_psnr - 20.0).abs() < 1e-4);
    }

    #[test]
    fn eval_aggregate_empty_slice() {
        let agg = eval_aggregate_metrics(&[]);
        assert_eq!(agg.total_views, 0);
        assert_eq!(agg.completed_views, 0);
        assert_eq!(agg.mean_psnr, 0.0);
    }

    // ------------------------------------------------------------------
    // find_extreme_views
    // ------------------------------------------------------------------

    #[test]
    fn find_extreme_views_correct_worst_best() {
        let views: Vec<ViewMetrics> = vec![25.0, 10.0, 40.0, 30.0]
            .into_iter()
            .enumerate()
            .map(|(i, p)| ViewMetrics {
                view_id: i,
                psnr: p,
                ssim: 0.9,
                mae: 0.0,
                mse: 0.0,
                lpips_approx: 0.0,
                render_time_ms: 0.0,
            })
            .collect();
        let (worst, best) = find_extreme_views(&views);
        let worst = worst.unwrap();
        let best = best.unwrap();
        assert_eq!(worst.view_id, 1); // PSNR=10
        assert_eq!(best.view_id, 2); // PSNR=40
    }

    #[test]
    fn find_extreme_views_single() {
        let views = vec![ViewMetrics {
            view_id: 0,
            psnr: 22.0,
            ssim: 0.8,
            mae: 0.1,
            mse: 0.01,
            lpips_approx: 0.1,
            render_time_ms: 0.0,
        }];
        let (worst, best) = find_extreme_views(&views);
        assert!(worst.is_some());
        assert!(best.is_some());
        assert_eq!(worst.unwrap().view_id, 0);
        assert_eq!(best.unwrap().view_id, 0);
    }

    #[test]
    fn find_extreme_views_empty() {
        let (worst, best) = find_extreme_views(&[]);
        assert!(worst.is_none());
        assert!(best.is_none());
    }

    // ------------------------------------------------------------------
    // ViewSynthesisEvaluator::should_eval
    // ------------------------------------------------------------------

    #[test]
    fn should_eval_true_at_multiples() {
        let ev = ViewSynthesisEvaluator::new(EvalConfig {
            eval_interval: 100,
            ..EvalConfig::default()
        });
        assert!(ev.should_eval(100));
        assert!(ev.should_eval(200));
        assert!(ev.should_eval(1000));
    }

    #[test]
    fn should_eval_false_at_non_multiples() {
        let ev = ViewSynthesisEvaluator::new(EvalConfig {
            eval_interval: 100,
            ..EvalConfig::default()
        });
        assert!(!ev.should_eval(0));
        assert!(!ev.should_eval(1));
        assert!(!ev.should_eval(50));
        assert!(!ev.should_eval(99));
        assert!(!ev.should_eval(101));
    }

    // ------------------------------------------------------------------
    // ViewSynthesisEvaluator::evaluate
    // ------------------------------------------------------------------

    fn make_eval_views(n: usize, w: usize, h: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let pred: Vec<Vec<f32>> = (0..n).map(|_| make_gradient_image(w, h)).collect();
        let gt: Vec<Vec<f32>> = (0..n).map(|_| make_image(w, h, 0.5)).collect();
        (pred, gt)
    }

    #[test]
    fn evaluate_single_updates_history() {
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            n_eval_views: 3,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);
        let (pred, gt) = make_eval_views(3, 16, 16);
        let metrics = ev.evaluate(1000, &pred, &gt).unwrap();
        assert!(metrics.mean_psnr.is_finite());
        assert_eq!(ev.history().len(), 1);
        assert!(ev.latest_metrics().is_some());
    }

    #[test]
    fn evaluate_two_evals_has_improved_works() {
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            n_eval_views: 2,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);

        // First eval: images with larger error.
        let bad_pred: Vec<Vec<f32>> = (0..2).map(|_| make_image(16, 16, 0.0)).collect();
        let gt: Vec<Vec<f32>> = (0..2).map(|_| make_image(16, 16, 1.0)).collect();
        ev.evaluate(1000, &bad_pred, &gt).unwrap();

        // Second eval: images closer to gt.
        let good_pred: Vec<Vec<f32>> = (0..2).map(|_| make_image(16, 16, 0.9)).collect();
        ev.evaluate(2000, &good_pred, &gt).unwrap();

        assert!(ev.has_improved(), "second eval should show improvement");
    }

    #[test]
    fn evaluate_empty_predicted_errors() {
        let mut ev = ViewSynthesisEvaluator::new(EvalConfig::default());
        let result = ev.evaluate(100, &[], &[]);
        assert!(matches!(result, Err(EvalError::EmptyViews)));
    }

    #[test]
    fn evaluate_mismatched_view_count_errors() {
        let mut ev = ViewSynthesisEvaluator::new(EvalConfig {
            image_width: 16,
            image_height: 16,
            ..EvalConfig::default()
        });
        let pred: Vec<Vec<f32>> = vec![make_image(16, 16, 0.5)];
        let gt: Vec<Vec<f32>> = vec![make_image(16, 16, 0.5), make_image(16, 16, 0.5)];
        assert!(matches!(
            ev.evaluate(100, &pred, &gt),
            Err(EvalError::InvalidParam(_))
        ));
    }

    // ------------------------------------------------------------------
    // psnr_trend
    // ------------------------------------------------------------------

    #[test]
    fn psnr_trend_correct_pairs() {
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);

        let (pred, gt) = make_eval_views(2, 16, 16);
        ev.evaluate(500, &pred, &gt).unwrap();
        ev.evaluate(1000, &pred, &gt).unwrap();

        let trend = ev.psnr_trend();
        assert_eq!(trend.len(), 2);
        assert_eq!(trend[0].0, 500);
        assert_eq!(trend[1].0, 1000);
        // Both steps had the same pred/gt, so PSNR should be equal.
        assert!((trend[0].1 - trend[1].1).abs() < 1e-3);
    }

    #[test]
    fn psnr_trend_empty_before_eval() {
        let ev = ViewSynthesisEvaluator::new(EvalConfig::default());
        assert!(ev.psnr_trend().is_empty());
    }

    // ------------------------------------------------------------------
    // best_metrics
    // ------------------------------------------------------------------

    #[test]
    fn best_metrics_returns_highest_psnr_record() {
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);

        // First eval: bad predictions (low PSNR).
        let bad: Vec<Vec<f32>> = vec![make_image(16, 16, 0.0)];
        let gt: Vec<Vec<f32>> = vec![make_image(16, 16, 1.0)];
        ev.evaluate(100, &bad, &gt).unwrap();

        // Second eval: perfect predictions (high PSNR).
        let perfect: Vec<Vec<f32>> = vec![make_image(16, 16, 1.0)];
        ev.evaluate(200, &perfect, &gt).unwrap();

        let best = ev.best_metrics().unwrap();
        // The best should be the second eval (higher PSNR).
        assert_eq!(best.step, 200, "best should be step 200");
    }

    #[test]
    fn best_metrics_none_before_eval() {
        let ev = ViewSynthesisEvaluator::new(EvalConfig::default());
        assert!(ev.best_metrics().is_none());
    }

    #[test]
    fn latest_metrics_returns_last_record() {
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);
        let (pred, gt) = make_eval_views(2, 16, 16);
        ev.evaluate(100, &pred, &gt).unwrap();
        ev.evaluate(200, &pred, &gt).unwrap();
        let latest = ev.latest_metrics().unwrap();
        assert_eq!(latest.step, 200);
    }

    // ------------------------------------------------------------------
    // format functions
    // ------------------------------------------------------------------

    #[test]
    fn format_eval_metrics_non_empty() {
        let m = EvalMetrics {
            mean_psnr: 28.5,
            median_psnr: 28.0,
            min_psnr: 20.0,
            max_psnr: 35.0,
            mean_ssim: 0.92,
            mean_mae: 0.03,
            mean_lpips_approx: 0.15,
            total_views: 10,
            completed_views: 10,
        };
        let s = format_eval_metrics(&m);
        assert!(!s.is_empty());
        assert!(
            s.contains("28.5") || s.contains("28.50"),
            "should contain mean PSNR"
        );
        assert!(s.contains("10"), "should contain view count");
    }

    #[test]
    fn format_view_metrics_non_empty() {
        let vm = ViewMetrics {
            view_id: 3,
            psnr: 25.0,
            ssim: 0.88,
            mae: 0.04,
            mse: 0.002,
            lpips_approx: 0.12,
            render_time_ms: 5.5,
        };
        let s = format_view_metrics(&vm);
        assert!(!s.is_empty());
        assert!(s.contains('3'), "should contain view_id");
        assert!(
            s.contains("25") || s.contains("25.00"),
            "should contain PSNR"
        );
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn single_pixel_image_metrics() {
        let w = 1;
        let h = 1;
        let a = vec![0.5_f32, 0.5, 0.5]; // H*W*3 = 3 elements
        let b = vec![0.7_f32, 0.3, 0.5];
        let mse = eval_mse(&a, &b).unwrap();
        let mae = eval_mae(&a, &b).unwrap();
        assert!(mse >= 0.0);
        assert!(mae >= 0.0);
        // SSIM on a 1×1 image should fall back and succeed.
        let ssim = eval_ssim(&a, &b, w, h).unwrap();
        assert!(ssim.is_finite());
    }

    #[test]
    fn all_white_vs_all_black() {
        let w = 32;
        let h = 32;
        let white = make_image(w, h, 1.0);
        let black = make_image(w, h, 0.0);
        let psnr = eval_psnr(&white, &black).unwrap();
        let ssim = eval_ssim(&white, &black, w, h).unwrap();
        let mae = eval_mae(&white, &black).unwrap();
        assert!(
            psnr < 1.0,
            "all-white vs all-black → very low PSNR, got {}",
            psnr
        );
        assert!((-1.0..=1.0).contains(&ssim));
        assert!((mae - 1.0).abs() < 1e-5, "all-white vs all-black → MAE=1");
    }

    #[test]
    fn error_type_display() {
        let e = EvalError::SizeMismatch {
            pred_w: 512,
            pred_h: 512,
            gt_w: 256,
            gt_h: 256,
        };
        let s = format!("{}", e);
        assert!(
            s.contains("512") && s.contains("256"),
            "display should include dimensions"
        );

        let e2 = EvalError::EmptyViews;
        assert!(!format!("{}", e2).is_empty());

        let e3 = EvalError::ViewIndexOutOfRange(42);
        assert!(format!("{}", e3).contains("42"));

        let e4 = EvalError::InvalidParam("bad input".into());
        assert!(format!("{}", e4).contains("bad input"));

        let e5 = EvalError::NotStarted;
        assert!(!format!("{}", e5).is_empty());
    }

    #[test]
    fn has_improved_false_with_one_eval() {
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);
        let (pred, gt) = make_eval_views(1, 16, 16);
        ev.evaluate(100, &pred, &gt).unwrap();
        assert!(!ev.has_improved(), "only one eval → no comparison possible");
    }

    #[test]
    fn has_improved_false_when_psnr_drops() {
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);

        // First eval: near-perfect.
        let perfect: Vec<Vec<f32>> = vec![make_image(16, 16, 1.0)];
        let gt: Vec<Vec<f32>> = vec![make_image(16, 16, 1.0)];
        ev.evaluate(100, &perfect, &gt).unwrap();

        // Second eval: bad predictions.
        let bad: Vec<Vec<f32>> = vec![make_image(16, 16, 0.0)];
        ev.evaluate(200, &bad, &gt).unwrap();

        assert!(
            !ev.has_improved(),
            "PSNR dropped → has_improved should be false"
        );
    }

    #[test]
    fn eval_aggregate_view_counts() {
        let views: Vec<ViewMetrics> = (0..7)
            .map(|i| ViewMetrics {
                view_id: i,
                psnr: 20.0,
                ssim: 0.8,
                mae: 0.1,
                mse: 0.01,
                lpips_approx: 0.1,
                render_time_ms: 0.0,
            })
            .collect();
        let agg = eval_aggregate_metrics(&views);
        assert_eq!(agg.total_views, 7);
        assert_eq!(agg.completed_views, 7);
    }

    #[test]
    fn eval_view_metrics_view_id_preserved() {
        let w = 16;
        let h = 16;
        let a = make_image(w, h, 0.5);
        let vm = eval_view_metrics(42, &a, &a, w, h, 8).unwrap();
        assert_eq!(vm.view_id, 42);
    }

    #[test]
    fn psnr_trend_grows_with_each_eval() {
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);
        let (pred, gt) = make_eval_views(1, 16, 16);
        for step in [100, 200, 300] {
            ev.evaluate(step, &pred, &gt).unwrap();
        }
        let trend = ev.psnr_trend();
        assert_eq!(trend.len(), 3);
        assert!(trend.iter().map(|(s, _)| s).eq([&100, &200, &300]));
    }
}
