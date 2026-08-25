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
    ///
    /// Always computed, even when [`EvalConfig::metrics`] omits
    /// [`EvalMetricKind::Psnr`] — it is the ranking key for every aggregate.
    pub psnr: f32,
    /// SSIM in [−1, 1] (ideally ≈1.0 for good quality).
    ///
    /// `f32::NAN` when [`EvalConfig::metrics`] did not select
    /// [`EvalMetricKind::Ssim`].
    pub ssim: f32,
    /// LPIPS approximation; lower means perceptually closer.
    ///
    /// `f32::NAN` when [`EvalConfig::metrics`] did not select
    /// [`EvalMetricKind::LpipsApprox`].
    pub lpips_approx: f32,
    /// Mean absolute error.
    ///
    /// `f32::NAN` when [`EvalConfig::metrics`] did not select
    /// [`EvalMetricKind::Mae`].
    pub mae: f32,
    /// Root mean square error.
    ///
    /// `f32::NAN` when [`EvalConfig::metrics`] did not select
    /// [`EvalMetricKind::Rmse`].
    pub rmse: f32,
    /// Multi-scale SSIM.
    ///
    /// `f32::NAN` when [`EvalConfig::metrics`] did not select
    /// [`EvalMetricKind::SsimMs`].
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
    ///
    /// Honoured by `eval_suite`: an unselected metric is not computed at all
    /// (skipping SSIM *and* MS-SSIM avoids the Gaussian-window convolutions,
    /// the dominant cost of a full evaluation) and is reported as `f32::NAN`
    /// on every [`ViewEvalResult`], which propagates to the corresponding
    /// aggregate mean.
    ///
    /// [`EvalMetricKind::Psnr`] is computed unconditionally — it is a single
    /// cheap pass and the ranking key behind `worst_views`, `best_views`,
    /// `std_psnr`, `min_psnr`/`max_psnr`, and the histogram/percentile
    /// helpers — so listing it is implied.
    ///
    /// Must name at least one kind; an empty list is rejected with
    /// [`EvalError::InvalidConfig`].
    pub metrics: Vec<EvalMetricKind>,
    /// Whether to store per-view results in the returned suite result.
    ///
    /// Not currently consulted: `eval_suite` always populates
    /// [`EvalSuiteResult::per_view`], because `eval_compare` needs the
    /// per-view list to count improved/degraded views and the `analyze eval`
    /// command reads it for `--per-view`. Honouring this flag therefore has
    /// to change its default to `true` in the same commit, or those callers
    /// silently lose their data.
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
    /// Standard deviation of PSNR across views, over the finite (imperfect)
    /// views only — a spread is undefined once an infinity enters it. `0.0`
    /// when fewer than two views are finite.
    pub std_psnr: f32,
    /// Minimum PSNR across views; `f32::INFINITY` only when *every* view is a
    /// pixel-perfect match.
    pub min_psnr: f32,
    /// Maximum PSNR across views, infinities included: `f32::INFINITY` when
    /// at least one view is a pixel-perfect match.
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
