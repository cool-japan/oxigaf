//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;

/// Identifies a specific image-quality metric.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EvalMetricKind {
    /// Peak signal-to-noise ratio (dB, higher is better).
    Psnr,
    /// Structural similarity index (higher is better).
    Ssim,
    /// Gradient-based perceptual proxy for LPIPS (lower is better).
    LpipsApprox,
    /// Mean absolute error (lower is better).
    Mae,
    /// Root mean square error (lower is better).
    Rmse,
    /// Multi-scale structural similarity index (higher is better).
    SsimMs,
}
impl EvalMetricKind {
    /// Short identifier used in reports and CSV headers.
    pub fn name(&self) -> &'static str {
        match self {
            EvalMetricKind::Psnr => "PSNR",
            EvalMetricKind::Ssim => "SSIM",
            EvalMetricKind::LpipsApprox => "LPIPS_APPROX",
            EvalMetricKind::Mae => "MAE",
            EvalMetricKind::Rmse => "RMSE",
            EvalMetricKind::SsimMs => "SSIM_MS",
        }
    }
    /// Returns `true` when a *higher* value means a *better* result.
    pub fn higher_is_better(&self) -> bool {
        match self {
            EvalMetricKind::Psnr => true,
            EvalMetricKind::Ssim => true,
            EvalMetricKind::LpipsApprox => false,
            EvalMetricKind::Mae => false,
            EvalMetricKind::Rmse => false,
            EvalMetricKind::SsimMs => true,
        }
    }
}
/// A single test item consisting of a view ID, predicted image, and ground truth.
pub struct EvalTestItem {
    /// Unique identifier for this view.
    pub view_id: String,
    /// Predicted (rendered) pixel data — flat interleaved RGB, [0, 1].
    pub pred: Vec<f32>,
    /// Ground-truth pixel data — flat interleaved RGB, [0, 1].
    pub gt: Vec<f32>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}
/// Evaluation results for a single rendered view vs. ground truth.
#[derive(Debug, Clone)]
pub struct ViewEvalResult {
    /// Identifier for this view (e.g. file name or camera index).
    pub view_id: String,
    /// PSNR in dB; `f32::INFINITY` when images are identical.
    pub psnr: f32,
    /// SSIM in [−1, 1] (ideally ≈1.0 for good quality).
    pub ssim: f32,
    /// LPIPS approximation; lower means perceptually closer.
    pub lpips_approx: f32,
    /// Mean absolute error.
    pub mae: f32,
    /// Root mean square error.
    pub rmse: f32,
    /// Multi-scale SSIM.
    pub ssim_ms: f32,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Set by the aggregator when this view is among the worst-N by PSNR.
    pub is_worst: bool,
    /// Set by the aggregator when this view is among the best-N by PSNR.
    pub is_best: bool,
}
/// Configuration for a batch evaluation run.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// Which metrics to compute (all by default).
    pub metrics: Vec<EvalMetricKind>,
    /// Whether to store per-view results in the returned suite result.
    pub save_per_view_results: bool,
    /// Number of worst-performing views to report (by PSNR).
    pub n_worst_views: usize,
    /// Number of best-performing views to report (by PSNR).
    pub n_best_views: usize,
}
/// Aggregate evaluation statistics across a full test set.
pub struct EvalSuiteResult {
    /// Per-view evaluation results (populated regardless of `save_per_view_results`).
    pub per_view: Vec<ViewEvalResult>,
    /// Mean PSNR across all views.
    pub mean_psnr: f32,
    /// Mean SSIM across all views.
    pub mean_ssim: f32,
    /// Mean LPIPS approximation across all views.
    pub mean_lpips: f32,
    /// Mean MAE across all views.
    pub mean_mae: f32,
    /// Standard deviation of PSNR across views.
    pub std_psnr: f32,
    /// Minimum PSNR across views.
    pub min_psnr: f32,
    /// Maximum PSNR across views.
    pub max_psnr: f32,
    /// Total number of evaluated views.
    pub n_views: usize,
    /// View IDs of the worst-performing views, sorted ascending by PSNR.
    pub worst_views: Vec<String>,
    /// View IDs of the best-performing views, sorted descending by PSNR.
    pub best_views: Vec<String>,
}
/// Side-by-side comparison of a baseline and a candidate evaluation result.
pub struct EvalComparison {
    /// Mean PSNR of the baseline model.
    pub baseline_mean_psnr: f32,
    /// Mean PSNR of the candidate model.
    pub candidate_mean_psnr: f32,
    /// PSNR improvement: `candidate − baseline` (positive = better candidate).
    pub delta_psnr: f32,
    /// SSIM improvement: `candidate − baseline` (positive = better candidate).
    pub delta_ssim: f32,
    /// LPIPS change: `candidate − baseline` (negative = better candidate).
    pub delta_lpips: f32,
    /// Number of views where the candidate achieved a higher PSNR.
    pub n_views_improved: usize,
    /// Number of views where the candidate achieved a lower PSNR.
    pub n_views_degraded: usize,
    /// Overall judgment: `true` if the candidate is better on average.
    pub is_candidate_better: bool,
}
/// Errors that can occur during evaluation.
#[derive(Debug, Error)]
pub enum EvalError {
    /// Wraps any [`std::io::Error`].
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The supplied test set contains zero items.
    #[error("empty test set")]
    EmptyTestSet,
    /// Prediction and ground-truth pixel buffers have different lengths.
    #[error("dimension mismatch: predicted has {pred} pixels, ground truth has {gt}")]
    DimensionMismatch { pred: usize, gt: usize },
    /// A configuration field has an invalid value.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// A requested view ID could not be found.
    #[error("view not found: {0}")]
    ViewNotFound(String),
    /// A metric computation produced an invalid result.
    #[error("metric computation failed: {0}")]
    MetricFailed(String),
}
