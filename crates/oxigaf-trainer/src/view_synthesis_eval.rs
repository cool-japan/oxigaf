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

use std::path::{Path, PathBuf};

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
    /// Number of views to evaluate per evaluation call. `evaluate` uses at
    /// most this many of the views it's handed, in the order given, so a
    /// caller can pass a large held-out set and let this field cap the
    /// per-call cost (`0` evaluates zero views).
    pub n_eval_views: usize,
    /// Image width in pixels.
    pub image_width: usize,
    /// Image height in pixels.
    pub image_height: usize,
    /// Whether rendered images should be saved as PNGs into
    /// `render_output_dir` during `evaluate`.
    pub save_renders: bool,
    /// Directory predicted/ground-truth PNGs are written into when
    /// `save_renders` is `true`. If `save_renders` is `true` but this is
    /// `None`, `evaluate` logs a `tracing::warn!` and skips saving (rather
    /// than silently doing nothing with no signal either way).
    pub render_output_dir: Option<PathBuf>,
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
            render_output_dir: None,
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

        // Honour `n_eval_views`: evaluate at most this many of the views
        // given, in order (previously accepted but never read — `evaluate`
        // always processed every view it was handed regardless).
        let n_views = self.config.n_eval_views.min(predicted.len());
        let predicted = &predicted[..n_views];
        let ground_truth = &ground_truth[..n_views];

        let w = self.config.image_width;
        let h = self.config.image_height;
        let patch = self.config.lpips_patch_size;

        if self.config.save_renders {
            self.write_render_pngs(step, predicted, w, h)?;
        }

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

    /// Write each of `predicted`'s views as a PNG into
    /// `config.render_output_dir`, named `step{step:08}_view{idx:04}.png`.
    ///
    /// Previously `save_renders` was accepted but never actually wrote
    /// anything ("handled externally"). Logs a `tracing::warn!` and
    /// returns `Ok(())` without writing anything if `render_output_dir` is
    /// `None` — `save_renders: true` with no configured destination is a
    /// config gap the caller should notice, not a silent no-op or a hard
    /// evaluation failure.
    fn write_render_pngs(
        &self,
        step: usize,
        predicted: &[Vec<f32>],
        width: usize,
        height: usize,
    ) -> Result<(), EvalError> {
        let Some(dir) = self.config.render_output_dir.as_ref() else {
            tracing::warn!(
                "EvalConfig::save_renders is true but render_output_dir is None \
                 — skipping render save"
            );
            return Ok(());
        };
        std::fs::create_dir_all(dir).map_err(|e| {
            EvalError::InvalidParam(format!(
                "failed to create render_output_dir '{}': {e}",
                dir.display()
            ))
        })?;
        for (view_id, pred) in predicted.iter().enumerate() {
            let path = dir.join(format!("step{step:08}_view{view_id:04}.png"));
            save_render_png(pred, width, height, &path)?;
        }
        Ok(())
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

/// Encode a flat `H×W×3`, `[0,1]`-range RGB buffer as a PNG and write it to
/// `path`, creating/overwriting the file.
fn save_render_png(
    data: &[f32],
    width: usize,
    height: usize,
    path: &Path,
) -> Result<(), EvalError> {
    let expected = width * height * 3;
    if data.len() != expected {
        return Err(EvalError::SizeMismatch {
            pred_w: width,
            pred_h: height,
            gt_w: data.len() / 3,
            gt_h: 1,
        });
    }
    let pixels: Vec<u8> = data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let img = image::RgbImage::from_raw(width as u32, height as u32, pixels).ok_or_else(|| {
        EvalError::InvalidParam("failed to build image buffer from pixel data".into())
    })?;
    img.save(path).map_err(|e| {
        EvalError::InvalidParam(format!("failed to save render '{}': {e}", path.display()))
    })?;
    Ok(())
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

/// SSIM (Structural Similarity Index), delegating to the exact SSIM the
/// trainer optimises against ([`crate::loss::ssim_loss`]: an 11×11 Gaussian
/// window, σ=1.5, replicate-padded so every pixel gets full coverage).
///
/// This fixes two prior defects in a standalone box-window implementation
/// that lived here:
/// - It hardcoded a constant `0.0` (the worst possible SSIM) for any image
///   smaller than the window, regardless of how similar the images
///   actually were, because no 11×11 box window could fit at all.
///   Replicate padding means every pixel always has a well-defined window
///   here, with no small-image special case needed.
/// - It used a *different* window (an unweighted box, strided by 4 pixels)
///   than [`crate::loss::ssim_loss`]'s Gaussian window, so this evaluation
///   metric could silently disagree with the training objective it's
///   supposed to be measuring.
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

    let kernel = crate::loss::gaussian_kernel_1d(11, 1.5);
    let dissimilarity = crate::loss::ssim_loss(predicted, ground_truth, width, height, &kernel);
    Ok(1.0 - dissimilarity)
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

/// Mean of per-patch mean absolute differences between two same-length
/// gradient-magnitude maps, tiling `width × height` into non-overlapping
/// `patch_size × patch_size` blocks (boundary blocks are smaller, not
/// dropped). Every patch counts equally regardless of its pixel count. When
/// every patch happens to be the same size (`patch_size` evenly divides
/// both `width` and `height`) this is arithmetically identical to a flat
/// per-pixel mean; it differs only when boundary patches are smaller,
/// weighting their (typically sparser) pixels more heavily per-pixel than
/// a full interior patch. `patch_size == 0` is treated as `1` (per-pixel
/// patches, always equivalent to a flat mean).
fn patch_mean_abs_diff(
    a: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    patch_size: usize,
) -> f32 {
    let patch_size = patch_size.max(1);
    let mut patch_means = Vec::new();
    let mut py = 0;
    while py < height {
        let y_end = (py + patch_size).min(height);
        let mut px = 0;
        while px < width {
            let x_end = (px + patch_size).min(width);
            let mut sum = 0.0_f32;
            let mut count = 0usize;
            for y in py..y_end {
                for x in px..x_end {
                    let idx = y * width + x;
                    sum += (a[idx] - b[idx]).abs();
                    count += 1;
                }
            }
            if count > 0 {
                patch_means.push(sum / count as f32);
            }
            px += patch_size;
        }
        py += patch_size;
    }
    if patch_means.is_empty() {
        0.0
    } else {
        patch_means.iter().sum::<f32>() / patch_means.len() as f32
    }
}

/// Simplified LPIPS approximation using per-channel Sobel gradient magnitudes.
///
/// For each channel, computes `|grad_mag_pred - grad_mag_gt|` per pixel and
/// aggregates it via `patch_mean_abs_diff` with `patch_size × patch_size`
/// tiles (previously `patch_size` was accepted but silently ignored,
/// always using a flat per-pixel mean instead — every patch now counts
/// equally, which matters when error is spatially uneven or the image
/// doesn't tile evenly), then averages across channels.
pub fn eval_lpips_approx(
    predicted: &[f32],
    ground_truth: &[f32],
    width: usize,
    height: usize,
    patch_size: usize,
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

    let mut total = 0.0_f32;

    for ch in 0..3_usize {
        let pred_ch = extract_channel(predicted, width, height, ch);
        let gt_ch = extract_channel(ground_truth, width, height, ch);

        let pred_mag = sobel_magnitude(&pred_ch, width, height);
        let gt_mag = sobel_magnitude(&gt_ch, width, height);

        total += patch_mean_abs_diff(&pred_mag, &gt_mag, width, height, patch_size);
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
        // 4×4 image — smaller than an 11×11 window, but replicate-padded
        // convolution (see eval_ssim's doc) gives every pixel full coverage
        // regardless of image size, so no small-image special case is
        // needed — this used to hit a hardcoded-0.0 fallback instead.
        let w = 4;
        let h = 4;
        let a = make_image(w, h, 0.5);
        let ssim = eval_ssim(&a, &a, w, h).unwrap();
        // Identical images → SSIM=1 exactly, at any size.
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

    #[test]
    fn eval_ssim_small_nearly_identical_images_not_hardcoded_zero() {
        // Regression: two nearly-identical small thumbnails used to report
        // SSIM = 0.0 (the worst possible score) purely because no 11×11 box
        // window fit — regardless of how similar the images actually were.
        let w = 8;
        let h = 8;
        let a = make_image(w, h, 0.5);
        let b: Vec<f32> = a.iter().map(|&v| v + 0.001).collect();
        let ssim = eval_ssim(&a, &b, w, h).unwrap();
        assert!(
            ssim > 0.9,
            "nearly-identical 8x8 images should score near 1.0, got {ssim}"
        );
    }

    #[test]
    fn eval_ssim_matches_training_time_ssim_loss() {
        // Regression: this module's SSIM used to diverge from
        // `loss::ssim_loss` (the one actually optimised during training) —
        // different window shape and stride entirely. They must now agree
        // exactly, since `eval_ssim` delegates to it directly.
        let w = 32;
        let h = 32;
        let a = make_gradient_image(w, h);
        let b: Vec<f32> = a.iter().map(|&v| (v * 1.3).min(1.0)).collect();
        let ssim = eval_ssim(&a, &b, w, h).unwrap();
        let kernel = crate::loss::gaussian_kernel_1d(11, 1.5);
        let dissimilarity = crate::loss::ssim_loss(&a, &b, w, h, &kernel);
        assert!(
            (ssim - (1.0 - dissimilarity)).abs() < 1e-6,
            "eval_ssim ({ssim}) must equal 1.0 - loss::ssim_loss ({dissimilarity})"
        );
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

    #[test]
    fn patch_mean_abs_diff_weights_boundary_patches_equally() {
        // Regression: `lpips_patch_size` used to be accepted but silently
        // ignored (always a flat per-pixel mean). With patch_size=2 over a
        // 3x1 row, tiling gives an unequal-sized final patch: [0,0] (mean
        // 0) and [6] (mean 6) -> mean of patch means = 3.0, which differs
        // from the flat per-pixel mean (0+0+6)/3 = 2.0.
        let a = [0.0_f32, 0.0, 6.0];
        let b = [0.0_f32, 0.0, 0.0];
        let flat = patch_mean_abs_diff(&a, &b, 3, 1, 1);
        let patched = patch_mean_abs_diff(&a, &b, 3, 1, 2);
        assert!(
            (flat - 2.0).abs() < 1e-6,
            "flat mean should be 2.0, got {flat}"
        );
        assert!(
            (patched - 3.0).abs() < 1e-6,
            "patch mean should be 3.0, got {patched}"
        );
    }

    #[test]
    fn eval_lpips_approx_matches_manual_patch_aggregation() {
        // End-to-end: `lpips_patch_size` must actually reach the
        // aggregation, not just be accepted and discarded. Reproduces
        // eval_lpips_approx's own pipeline manually (extract channel,
        // Sobel, patch-aggregate) with a patch_size that does *not* evenly
        // divide the image (so patch weighting is actually exercised, per
        // patch_mean_abs_diff's doc) and checks they agree — this would
        // fail if `patch_size` were silently ignored in favour of a flat
        // per-pixel mean.
        let w = 8;
        let h = 8;
        let patch_size = 3; // 8 does not divide evenly by 3
        let pred = make_gradient_image(w, h);
        let gt = make_image(w, h, 0.5);

        let mut expected = 0.0_f32;
        for ch in 0..3 {
            let pred_ch = extract_channel(&pred, w, h, ch);
            let gt_ch = extract_channel(&gt, w, h, ch);
            let pred_mag = sobel_magnitude(&pred_ch, w, h);
            let gt_mag = sobel_magnitude(&gt_ch, w, h);
            expected += patch_mean_abs_diff(&pred_mag, &gt_mag, w, h, patch_size);
        }
        expected /= 3.0;

        let actual = eval_lpips_approx(&pred, &gt, w, h, patch_size).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );

        // And patch_size=1 (equal-sized 1x1 patches) must match the old
        // flat per-pixel-mean behaviour exactly.
        let mut flat_expected = 0.0_f32;
        for ch in 0..3 {
            let pred_ch = extract_channel(&pred, w, h, ch);
            let gt_ch = extract_channel(&gt, w, h, ch);
            let pred_mag = sobel_magnitude(&pred_ch, w, h);
            let gt_mag = sobel_magnitude(&gt_ch, w, h);
            let n_pixels = w * h;
            let channel_sum: f32 = pred_mag
                .iter()
                .zip(gt_mag.iter())
                .map(|(pm, gm)| (pm - gm).abs())
                .sum();
            flat_expected += channel_sum / n_pixels as f32;
        }
        flat_expected /= 3.0;
        let actual_flat = eval_lpips_approx(&pred, &gt, w, h, 1).unwrap();
        assert!(
            (actual_flat - flat_expected).abs() < 1e-6,
            "patch_size=1 should equal the flat per-pixel mean"
        );
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

    #[test]
    fn evaluate_honors_n_eval_views_cap() {
        // Regression: n_eval_views was accepted but never read — evaluate
        // always processed every view it was handed, regardless of this
        // field.
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            n_eval_views: 2,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);
        let (pred, gt) = make_eval_views(5, 16, 16); // hand it 5, cap is 2
        let metrics = ev.evaluate(1000, &pred, &gt).unwrap();
        assert_eq!(
            metrics.total_views, 2,
            "must cap at n_eval_views, not use all 5 given"
        );
        assert_eq!(ev.latest_metrics().unwrap().per_view.len(), 2);
    }

    #[test]
    fn evaluate_save_renders_writes_png_files() {
        // Regression: save_renders was accepted but evaluate() never wrote
        // anything ("handled externally").
        let dir = std::env::temp_dir().join(format!(
            "oxigaf_view_synth_eval_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let config = EvalConfig {
            image_width: 4,
            image_height: 4,
            n_eval_views: 2,
            save_renders: true,
            render_output_dir: Some(dir.clone()),
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);
        let (pred, gt) = make_eval_views(2, 4, 4);
        ev.evaluate(7, &pred, &gt).unwrap();

        assert!(dir.join("step00000007_view0000.png").exists());
        assert!(dir.join("step00000007_view0001.png").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn evaluate_save_renders_without_dir_warns_and_does_not_error() {
        let config = EvalConfig {
            image_width: 16,
            image_height: 16,
            save_renders: true,
            render_output_dir: None,
            ..EvalConfig::default()
        };
        let mut ev = ViewSynthesisEvaluator::new(config);
        let (pred, gt) = make_eval_views(1, 16, 16);
        // A missing destination is logged, not fatal.
        assert!(ev.evaluate(1, &pred, &gt).is_ok());
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
