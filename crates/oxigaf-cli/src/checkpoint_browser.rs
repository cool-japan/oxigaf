//! Checkpoint browser — discover, analyze, compare, and select training checkpoints.
//!
//! This module provides tools for browsing and comparing training checkpoints
//! without performing file I/O beyond path inspection.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during checkpoint browsing operations.
///
/// [`BrowserError::ParseError`] is returned by [`BrowserCheckpoint::try_from_path`]
/// and [`BrowserError::CheckpointNotFound`] by
/// [`CheckpointBrowser::find_at_step_exact`]. The remaining variants —
/// [`BrowserError::NoCheckpoints`], [`BrowserError::InvalidParam`], and
/// [`BrowserError::TooFewCheckpoints`] — are reserved for the CLI command
/// layer that will eventually scan a directory and construct a
/// [`CheckpointBrowser`] from it (that wiring does not exist yet —
/// `checkpoint_browser` is not reachable from any subcommand): this module
/// performs no directory I/O of its own (see the module doc), and the
/// existing `find_psnr_elbow`/`estimate_steps_to_psnr` query functions
/// already use `Option` idiomatically for "not enough data" rather than
/// needing a typed error.
#[derive(Debug, Error)]
pub enum BrowserError {
    #[error("No checkpoints found in {0}")]
    NoCheckpoints(String),
    #[error("Checkpoint {0} not found")]
    CheckpointNotFound(String),
    #[error("Parse error in checkpoint metadata: {0}")]
    ParseError(String),
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
    #[error("Insufficient checkpoints for comparison (need >= 2, have {0})")]
    TooFewCheckpoints(usize),
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Checkpoint metadata discovered from directory listing (no file I/O).
#[derive(Debug, Clone)]
pub struct BrowserCheckpoint {
    /// Absolute or relative path to the checkpoint file.
    pub path: String,
    /// Training step extracted from the path.
    pub step: usize,
    /// Epoch number (if available).
    pub epoch: Option<usize>,
    /// Peak signal-to-noise ratio (if available).
    pub psnr: Option<f32>,
    /// Training loss (if available).
    pub loss: Option<f32>,
    /// Number of 3D Gaussians (if available).
    pub n_gaussians: Option<usize>,
    /// Unix timestamp from filename or metadata.
    pub timestamp: Option<u64>,
    /// File size in bytes.
    pub file_size_bytes: usize,
    /// Tags extracted from the path (e.g. "best", "final", "epoch_10").
    pub tags: Vec<String>,
}

impl BrowserCheckpoint {
    /// Build a checkpoint record for `path` with an already-resolved `step`.
    fn build(path: &str, step: usize) -> Self {
        Self {
            path: path.to_string(),
            step,
            epoch: None,
            psnr: parse_psnr_from_path(path),
            loss: None,
            n_gaussians: None,
            timestamp: None,
            file_size_bytes: 0,
            tags: extract_tags_from_path(path),
        }
    }

    /// Construct a `BrowserCheckpoint` by parsing metadata from a path string.
    ///
    /// Only the filename component is used for parsing; no actual I/O is
    /// performed. When the training step cannot be determined from the
    /// path, this falls back to step `0` rather than failing — use
    /// [`Self::try_from_path`] when an unparseable step should be treated
    /// as an error instead.
    pub fn from_path(path: &str) -> Self {
        let step = parse_step_from_path(path).unwrap_or(0);
        Self::build(path, step)
    }

    /// Like [`Self::from_path`], but returns [`BrowserError::ParseError`]
    /// instead of silently defaulting to step `0` when no training step can
    /// be determined from the path.
    pub fn try_from_path(path: &str) -> Result<Self, BrowserError> {
        let step = parse_step_from_path(path).ok_or_else(|| {
            BrowserError::ParseError(format!(
                "could not determine a training step from path '{path}'"
            ))
        })?;
        Ok(Self::build(path, step))
    }

    /// Return `true` if this checkpoint is tagged or named as "best".
    pub fn is_best(&self) -> bool {
        let lower = self.path.to_ascii_lowercase();
        lower.contains("best") || self.tags.iter().any(|t| t == "best")
    }

    /// Return `true` if this checkpoint is tagged or named as "final".
    pub fn is_final(&self) -> bool {
        let lower = self.path.to_ascii_lowercase();
        lower.contains("final") || self.tags.iter().any(|t| t == "final")
    }

    /// Composite quality score, *typically* in `[0, 1]` but not clamped:
    ///
    /// - PSNR available: `psnr / 50.0` — unbounded above for PSNR > 50 dB
    ///   (routine for a well-converged synthetic scene) and negative for a
    ///   negative PSNR parsed from a filename.
    /// - Loss available: `1.0 - loss.min(1.0)` — negative for loss > 1.0.
    /// - Neither: `0.0`
    ///
    /// [`BrowserSort::ByQualityScore`] still orders correctly regardless,
    /// since it only compares these values against each other; a caller
    /// that needs a normalised `[0, 1]` fraction should clamp explicitly.
    pub fn quality_score(&self) -> f32 {
        if let Some(psnr) = self.psnr {
            psnr / 50.0
        } else if let Some(loss) = self.loss {
            1.0 - loss.min(1.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Browser configuration
// ---------------------------------------------------------------------------

/// Sort order for checkpoint browsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrowserSort {
    /// Ascending step (oldest first).
    ByStep,
    /// Descending step (most recent first).
    ByStepDesc,
    /// Descending PSNR (best first).
    ByPsnr,
    /// Ascending loss (best first).
    ByLoss,
    /// Descending file size (largest first).
    ByFileSize,
    /// Descending composite quality score (best first).
    ByQualityScore,
}

/// Filter criteria for checkpoint browsing.
#[derive(Debug, Clone, Default)]
pub struct BrowserFilter {
    /// Minimum step (inclusive).
    pub min_step: Option<usize>,
    /// Maximum step (inclusive).
    pub max_step: Option<usize>,
    /// Minimum PSNR.
    pub min_psnr: Option<f32>,
    /// Maximum loss.
    pub max_loss: Option<f32>,
    /// Tags that must all be present.
    pub tags_required: Vec<String>,
    /// Tags that must not be present.
    pub tags_excluded: Vec<String>,
}

impl BrowserFilter {
    /// Return `true` if the checkpoint passes all filter conditions.
    pub fn passes(&self, ckpt: &BrowserCheckpoint) -> bool {
        if let Some(min) = self.min_step {
            if ckpt.step < min {
                return false;
            }
        }
        if let Some(max) = self.max_step {
            if ckpt.step > max {
                return false;
            }
        }
        if let Some(min_psnr) = self.min_psnr {
            match ckpt.psnr {
                Some(p) if p >= min_psnr => {}
                Some(_) => return false,
                None => return false,
            }
        }
        if let Some(max_loss) = self.max_loss {
            match ckpt.loss {
                Some(l) if l <= max_loss => {}
                Some(_) => return false,
                None => return false,
            }
        }
        for req in &self.tags_required {
            if !ckpt.tags.contains(req) {
                return false;
            }
        }
        for excl in &self.tags_excluded {
            if ckpt.tags.contains(excl) {
                return false;
            }
        }
        true
    }
}

/// Configuration for `CheckpointBrowser`.
#[derive(Debug, Clone)]
pub struct BrowserConfig {
    /// Sort order.
    pub sort_by: BrowserSort,
    /// Filter criteria.
    pub filter: BrowserFilter,
    /// Maximum number of checkpoints to return from `browse()`.
    pub max_display: usize,
    /// Whether to include tags in formatted output.
    pub show_tags: bool,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            sort_by: BrowserSort::ByStep,
            filter: BrowserFilter::default(),
            max_display: 20,
            show_tags: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Main browser
// ---------------------------------------------------------------------------

/// Browser for discovering and selecting training checkpoints.
pub struct CheckpointBrowser {
    checkpoints: Vec<BrowserCheckpoint>,
    config: BrowserConfig,
}

impl CheckpointBrowser {
    /// Create a new browser from a pre-built list of checkpoints.
    pub fn new(checkpoints: Vec<BrowserCheckpoint>, config: BrowserConfig) -> Self {
        Self {
            checkpoints,
            config,
        }
    }

    /// Create a browser by parsing metadata from a list of paths.
    pub fn from_paths(paths: Vec<String>, config: BrowserConfig) -> Self {
        let checkpoints = paths
            .iter()
            .map(|p| BrowserCheckpoint::from_path(p))
            .collect();
        Self {
            checkpoints,
            config,
        }
    }

    /// Apply filter and sort, return matching checkpoints (up to `max_display`).
    pub fn browse(&self) -> Vec<&BrowserCheckpoint> {
        let mut filtered: Vec<&BrowserCheckpoint> = self
            .checkpoints
            .iter()
            .filter(|c| self.config.filter.passes(c))
            .collect();

        match self.config.sort_by {
            BrowserSort::ByStep => {
                filtered.sort_by_key(|c| c.step);
            }
            BrowserSort::ByStepDesc => {
                filtered.sort_by_key(|c| std::cmp::Reverse(c.step));
            }
            BrowserSort::ByPsnr => {
                filtered.sort_by(|a, b| {
                    let pa = a.psnr.unwrap_or(f32::NEG_INFINITY);
                    let pb = b.psnr.unwrap_or(f32::NEG_INFINITY);
                    pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            BrowserSort::ByLoss => {
                filtered.sort_by(|a, b| {
                    let la = a.loss.unwrap_or(f32::INFINITY);
                    let lb = b.loss.unwrap_or(f32::INFINITY);
                    la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            BrowserSort::ByFileSize => {
                filtered.sort_by_key(|c| std::cmp::Reverse(c.file_size_bytes));
            }
            BrowserSort::ByQualityScore => {
                filtered.sort_by(|a, b| {
                    let sa = a.quality_score();
                    let sb = b.quality_score();
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        filtered.into_iter().take(self.config.max_display).collect()
    }

    /// Find the best checkpoint by PSNR (if available) or lowest loss.
    pub fn find_best(&self) -> Option<&BrowserCheckpoint> {
        let with_psnr: Vec<&BrowserCheckpoint> = self
            .checkpoints
            .iter()
            .filter(|c| c.psnr.is_some())
            .collect();

        if !with_psnr.is_empty() {
            return with_psnr.into_iter().max_by(|a, b| {
                a.psnr
                    .unwrap_or(f32::NEG_INFINITY)
                    .partial_cmp(&b.psnr.unwrap_or(f32::NEG_INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let with_loss: Vec<&BrowserCheckpoint> = self
            .checkpoints
            .iter()
            .filter(|c| c.loss.is_some())
            .collect();

        if !with_loss.is_empty() {
            return with_loss.into_iter().min_by(|a, b| {
                a.loss
                    .unwrap_or(f32::INFINITY)
                    .partial_cmp(&b.loss.unwrap_or(f32::INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        None
    }

    /// Find the most recent checkpoint (highest step number).
    pub fn find_latest(&self) -> Option<&BrowserCheckpoint> {
        self.checkpoints.iter().max_by_key(|c| c.step)
    }

    /// Find the checkpoint at a specific step (exact match) or nearest.
    pub fn find_at_step(&self, step: usize) -> Option<&BrowserCheckpoint> {
        // Try exact match first
        if let Some(exact) = self.checkpoints.iter().find(|c| c.step == step) {
            return Some(exact);
        }
        // Fall back to nearest
        self.find_nearest_step(step)
    }

    /// Find the checkpoint at exactly `step`, without falling back to the
    /// nearest one.
    ///
    /// Unlike [`Self::find_at_step`] (which silently substitutes the
    /// nearest checkpoint when there is no exact match, giving the caller
    /// no way to tell the two cases apart), this returns
    /// [`BrowserError::CheckpointNotFound`] when `step` is not present.
    pub fn find_at_step_exact(&self, step: usize) -> Result<&BrowserCheckpoint, BrowserError> {
        self.checkpoints
            .iter()
            .find(|c| c.step == step)
            .ok_or_else(|| BrowserError::CheckpointNotFound(step.to_string()))
    }

    /// Find the checkpoint whose step is nearest to the target.
    pub fn find_nearest_step(&self, step: usize) -> Option<&BrowserCheckpoint> {
        self.checkpoints
            .iter()
            .min_by_key(|c| c.step.abs_diff(step))
    }

    /// Get the checkpoint at a percentile of training progress.
    ///
    /// `p = 0.0` → lowest step, `p = 1.0` → highest step, `p = 0.5` → midpoint.
    pub fn at_percentile(&self, p: f32) -> Option<&BrowserCheckpoint> {
        if self.checkpoints.is_empty() {
            return None;
        }
        let mut sorted: Vec<&BrowserCheckpoint> = self.checkpoints.iter().collect();
        sorted.sort_by_key(|c| c.step);
        let n = sorted.len();
        let idx = ((p.clamp(0.0, 1.0) * (n - 1) as f32).round() as usize).min(n - 1);
        sorted.into_iter().nth(idx)
    }

    /// Total number of checkpoints.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Return `true` if there are no checkpoints.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Sum of all checkpoint file sizes in bytes.
    pub fn total_size_bytes(&self) -> usize {
        self.checkpoints.iter().map(|c| c.file_size_bytes).sum()
    }

    /// Return `(min_step, max_step)` or `None` if there are no checkpoints.
    pub fn step_range(&self) -> Option<(usize, usize)> {
        if self.checkpoints.is_empty() {
            return None;
        }
        let min = self.checkpoints.iter().map(|c| c.step).min()?;
        let max = self.checkpoints.iter().map(|c| c.step).max()?;
        Some((min, max))
    }
}

// ---------------------------------------------------------------------------
// Comparison types
// ---------------------------------------------------------------------------

/// Differences between two checkpoints.
#[derive(Debug, Clone)]
pub struct CheckpointDiff {
    /// Step difference: `b.step - a.step` (signed).
    pub step_delta: i64,
    /// PSNR difference: `b.psnr - a.psnr` (positive = b is better).
    pub psnr_delta: Option<f32>,
    /// Loss difference: `b.loss - a.loss` (negative = b is better).
    pub loss_delta: Option<f32>,
    /// Gaussian count difference: `b.n_gaussians - a.n_gaussians` (signed).
    pub gaussians_delta: Option<i64>,
    /// File size difference in bytes (signed).
    pub size_delta: i64,
    /// Tags present in b but not in a.
    pub tags_added: Vec<String>,
    /// Tags present in a but not in b.
    pub tags_removed: Vec<String>,
}

/// Statistics about the spacing between checkpoints.
#[derive(Debug, Clone)]
pub struct SpacingStats {
    /// Mean step gap between consecutive checkpoints.
    pub mean_step_gap: f32,
    /// Minimum step gap.
    pub min_step_gap: usize,
    /// Maximum step gap.
    pub max_step_gap: usize,
    /// Whether gaps are approximately equal (std < 10% of mean).
    pub is_regular: bool,
    /// Total steps covered (last step - first step).
    pub total_steps: usize,
}

// ---------------------------------------------------------------------------
// Comparison functions
// ---------------------------------------------------------------------------

/// Compare two checkpoints and compute their differences.
pub fn compare_checkpoints(a: &BrowserCheckpoint, b: &BrowserCheckpoint) -> CheckpointDiff {
    let step_delta = b.step as i64 - a.step as i64;

    let psnr_delta = match (a.psnr, b.psnr) {
        (Some(pa), Some(pb)) => Some(pb - pa),
        _ => None,
    };

    let loss_delta = match (a.loss, b.loss) {
        (Some(la), Some(lb)) => Some(lb - la),
        _ => None,
    };

    let gaussians_delta = match (a.n_gaussians, b.n_gaussians) {
        (Some(ga), Some(gb)) => Some(gb as i64 - ga as i64),
        _ => None,
    };

    let size_delta = b.file_size_bytes as i64 - a.file_size_bytes as i64;

    let tags_added = b
        .tags
        .iter()
        .filter(|t| !a.tags.contains(t))
        .cloned()
        .collect();

    let tags_removed = a
        .tags
        .iter()
        .filter(|t| !b.tags.contains(t))
        .cloned()
        .collect();

    CheckpointDiff {
        step_delta,
        psnr_delta,
        loss_delta,
        gaussians_delta,
        size_delta,
        tags_added,
        tags_removed,
    }
}

/// Compute the PSNR progression trend from a sequence of checkpoints.
///
/// Returns `(step, psnr)` pairs for checkpoints that have PSNR, sorted by step.
pub fn psnr_trend(checkpoints: &[BrowserCheckpoint]) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = checkpoints
        .iter()
        .filter_map(|c| c.psnr.map(|p| (c.step, p)))
        .collect();
    pairs.sort_by_key(|(step, _)| *step);
    pairs
}

/// Find the elbow point in a PSNR curve (point of diminishing returns).
///
/// Computes slopes between consecutive PSNR values and returns the step
/// at which the slope decrease is greatest (second derivative minimum).
/// Returns `None` if fewer than 3 PSNR-bearing checkpoints are available.
pub fn find_psnr_elbow(checkpoints: &[BrowserCheckpoint]) -> Option<usize> {
    let trend = psnr_trend(checkpoints);
    if trend.len() < 3 {
        return None;
    }

    // Compute first differences (slopes between consecutive points)
    let mut slopes: Vec<f32> = Vec::with_capacity(trend.len() - 1);
    for i in 1..trend.len() {
        let step_gap = (trend[i].0 as f32 - trend[i - 1].0 as f32).max(1.0);
        let psnr_gap = trend[i].1 - trend[i - 1].1;
        slopes.push(psnr_gap / step_gap);
    }

    // Find the index where slope drops most (largest decrease in slope)
    let mut max_drop = f32::NEG_INFINITY;
    let mut elbow_idx = 0usize;
    for i in 1..slopes.len() {
        let drop = slopes[i - 1] - slopes[i];
        if drop > max_drop {
            max_drop = drop;
            elbow_idx = i;
        }
    }

    // `elbow_idx` in slopes corresponds to `elbow_idx + 1` in trend
    Some(trend[elbow_idx + 1].0)
}

/// Ordinary least-squares fit of `psnr = slope * step + intercept` over
/// `trend`, extrapolated to find the number of *additional* steps (beyond
/// the last observed one) needed to reach `target_psnr`.
///
/// Returns `None` when the trend has fewer than 2 points, the step values
/// are degenerate (no variance — e.g. every checkpoint claims the same
/// step), or the fitted slope is not positive, not finite (a NaN PSNR
/// poisons the fit), or too flat to reach the target in a finite number of
/// steps — in short, whenever the extrapolation would be a fiction.
///
/// The count is rounded up, but only past a tolerance derived from the f32
/// quantisation of the PSNR samples: without it an exact answer of `k`
/// steps reports `k + 1` whenever the fit lands a few ulps above `k`.
///
/// Accumulates in `f64` and centres `step` on its mean before forming the
/// normal-equation denominator. The naive single-pass f32 formula
/// (`n*Σx² - (Σx)²`) subtracts two nearly-equal ~1e10+ magnitude
/// quantities for realistic step values (up to hundreds of thousands),
/// which loses nearly all significant digits in f32's 24-bit mantissa —
/// and the resulting near-zero-magnitude `f32::EPSILON` guard then fails
/// to catch anything but an exactly-zero denominator. Centring computes the
/// same quantity (`n * Σ(x-x̄)²`, by the standard sum-of-squares identity)
/// directly, without cancellation, so the degeneracy guard below is
/// actually meaningful.
fn fit_steps_to_psnr(trend: &[(usize, f32)], target_psnr: f32) -> Option<usize> {
    if trend.len() < 2 {
        return None;
    }

    let n = trend.len() as f64;
    let x_mean = trend.iter().map(|(s, _)| *s as f64).sum::<f64>() / n;
    let y_mean = trend.iter().map(|(_, p)| *p as f64).sum::<f64>() / n;

    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    for (step, psnr) in trend {
        let dx = *step as f64 - x_mean;
        let dy = *psnr as f64 - y_mean;
        sxx += dx * dx;
        sxy += dx * dy;
    }

    // Relative-to-magnitude guard: catches "all steps identical" (sxx
    // exactly 0) without being fooled by the huge absolute scale of real
    // step counts.
    if sxx <= f64::EPSILON * x_mean.abs().max(1.0) {
        return None;
    }

    let slope = sxy / sxx;
    let intercept = y_mean - slope * x_mean;

    // `slope <= 0.0` is *false* for NaN, so a checkpoint carrying a NaN PSNR
    // would otherwise sail through this guard and produce a NaN
    // `target_step` — which `as usize` silently saturates to 0, i.e. "you
    // are already there". An unusable fit has to say so.
    if !slope.is_finite() || slope <= 0.0 {
        return None;
    }

    // target_psnr = slope * step + intercept → step = (target - intercept) / slope
    let target_step = (target_psnr as f64 - intercept) / slope;
    if !target_step.is_finite() {
        return None;
    }
    let last_step = trend.last().map(|(s, _)| *s as f64).unwrap_or(0.0);

    // Extrapolating a fit of f32 inputs and then rounding *up* turns the
    // quantisation error into a whole extra step: a perfect line whose exact
    // answer is 6 steps computes as 6.0000153 and reports 7. The tolerance
    // below is that quantisation expressed in steps, so only genuine
    // fractional demand survives the ceil.
    //
    // A PSNR sample carries an error of about one f32 ulp, `eps * |y|`. The
    // OLS line inherits it at abscissa `x` scaled by the usual
    // `sqrt(1/n + (x - x̄)²/sxx)` lever (1/n from the mean, the second term
    // from the slope over the observed spread), the target value contributes
    // one more ulp, and dividing by the slope converts PSNR into steps.
    let y_scale = (target_psnr as f64).abs().max(y_mean.abs()).max(1.0);
    let psnr_tolerance = f64::from(f32::EPSILON) * y_scale;
    let lever = (1.0 / n + (target_step - x_mean).powi(2) / sxx).sqrt();
    let step_tolerance = psnr_tolerance * (1.0 + lever) / slope;
    // An absurdly distant extrapolation can overflow the lever to infinity;
    // no tolerance at all is the safe reading there (it can only add steps,
    // never invent them).
    let step_tolerance = if step_tolerance.is_finite() {
        step_tolerance
    } else {
        0.0
    };

    let extra = (target_step - last_step - step_tolerance).ceil();
    if extra <= 0.0 {
        // The target is already reached (or within one ulp of it).
        return Some(0);
    }
    Some(extra as usize)
}

/// Estimate the number of additional steps required to reach a target PSNR.
///
/// Uses linear extrapolation through the last available PSNR points.
/// Returns `None` if the trend is not improving (slope <= 0) or if fewer
/// than 2 PSNR-bearing checkpoints are available.
pub fn estimate_steps_to_psnr(
    checkpoints: &[BrowserCheckpoint],
    target_psnr: f32,
) -> Option<usize> {
    let trend = psnr_trend(checkpoints);
    fit_steps_to_psnr(&trend, target_psnr)
}

/// Compute statistics about the step spacing between checkpoints.
pub fn checkpoint_spacing_stats(checkpoints: &[BrowserCheckpoint]) -> SpacingStats {
    if checkpoints.len() < 2 {
        return SpacingStats {
            mean_step_gap: 0.0,
            min_step_gap: 0,
            max_step_gap: 0,
            is_regular: true,
            total_steps: 0,
        };
    }

    let mut steps: Vec<usize> = checkpoints.iter().map(|c| c.step).collect();
    steps.sort_unstable();

    let gaps: Vec<usize> = steps.windows(2).map(|w| w[1] - w[0]).collect();
    let n = gaps.len() as f32;
    let sum: usize = gaps.iter().sum();
    let mean = sum as f32 / n;
    let min_gap = *gaps.iter().min().unwrap_or(&0);
    let max_gap = *gaps.iter().max().unwrap_or(&0);

    let variance = gaps.iter().map(|&g| (g as f32 - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();
    let is_regular = mean < f32::EPSILON || std_dev < 0.1 * mean;

    let total_steps = steps.last().copied().unwrap_or(0) - steps.first().copied().unwrap_or(0);

    SpacingStats {
        mean_step_gap: mean,
        min_step_gap: min_gap,
        max_step_gap: max_gap,
        is_regular,
        total_steps,
    }
}

// ---------------------------------------------------------------------------
// Parsing utilities
// ---------------------------------------------------------------------------

/// Returns `true` when `tok` is a poor step-number candidate because it
/// looks like a date or a raw timestamp rather than a training step: purely
/// numeric and either longer than 8 digits (implausible as a step count —
/// more likely a Unix timestamp), or exactly 8 digits forming a plausible
/// `YYYYMMDD` date.
fn looks_like_date_or_timestamp(tok: &str) -> bool {
    if tok.is_empty() || !tok.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if tok.len() > 8 {
        return true;
    }
    if tok.len() == 8 {
        let year = tok[0..4].parse::<u32>().unwrap_or(0);
        let month = tok[4..6].parse::<u32>().unwrap_or(0);
        let day = tok[6..8].parse::<u32>().unwrap_or(0);
        return (1970..=9999).contains(&year)
            && (1..=12).contains(&month)
            && (1..=31).contains(&day);
    }
    false
}

/// Returns `true` if some token other than `tokens[skip]` is a plausible
/// (non-date/timestamp-shaped) numeric step candidate.
fn has_better_step_candidate(tokens: &[&str], skip: usize) -> bool {
    tokens.iter().enumerate().any(|(idx, tok)| {
        idx != skip
            && !tok.is_empty()
            && tok.chars().all(|c| c.is_ascii_digit())
            && !looks_like_date_or_timestamp(tok)
    })
}

/// Parse `tokens[idx]` as a step number, rejecting it in favour of `None`
/// only when it looks like a date/timestamp *and* a better candidate exists
/// elsewhere in `tokens` — a lone date-shaped number is still accepted
/// (some evidence beats none), it just loses to a better alternative when
/// one is available.
fn accept_step_candidate(tokens: &[&str], tok: &str, idx: usize) -> Option<usize> {
    let n = tok.parse::<usize>().ok()?;
    if looks_like_date_or_timestamp(tok) && has_better_step_candidate(tokens, idx) {
        None
    } else {
        Some(n)
    }
}

/// Parse a step number from a checkpoint filename or path.
///
/// Recognises patterns like: `"ckpt_1000"`, `"step_1000"`, `"checkpoint-1000"`,
/// `"model_1000.json"`. Takes the last occurring number after a recognised prefix.
///
/// A numeric token that looks like a date (`YYYYMMDD`) or a raw timestamp
/// (more than 8 digits) is treated as a poor step candidate and skipped
/// whenever a more plausible numeric token exists elsewhere in the
/// filename — e.g. `"checkpoint_20260101_1000.json"` yields `1000`, not the
/// embedded date.
pub fn parse_step_from_path(path: &str) -> Option<usize> {
    // Extract the filename component
    let filename = path.rsplit(['/', '\\']).next().unwrap_or(path);

    // Strip extension
    let stem = if let Some(pos) = filename.rfind('.') {
        &filename[..pos]
    } else {
        filename
    };

    // Tokenise on '_' and '-'
    let tokens: Vec<&str> = stem.split(['_', '-']).collect();
    let keywords = ["ckpt", "checkpoint", "step", "model"];

    // Keyword-driven search
    let mut i = 0;
    while i < tokens.len() {
        let lower = tokens[i].to_ascii_lowercase();
        if keywords.iter().any(|k| *k == lower) {
            if let Some(next) = tokens.get(i + 1) {
                if let Some(n) = accept_step_candidate(&tokens, next, i + 1) {
                    return Some(n);
                }
                // Skip over another keyword, or a rejected date/timestamp
                // ("checkpoint_step_1000", "checkpoint_20260101_1000")
                if let Some(after) = tokens.get(i + 2) {
                    if let Some(n) = accept_step_candidate(&tokens, after, i + 2) {
                        return Some(n);
                    }
                }
            }
        }
        i += 1;
    }

    // Fallback: last purely-numeric token after at least one non-numeric
    // token, preferring one that isn't date/timestamp-shaped.
    let mut found_non_numeric = false;
    let mut last_number: Option<usize> = None;
    let mut last_plausible_number: Option<usize> = None;
    for tok in &tokens {
        let is_numeric = !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit());
        if is_numeric {
            if found_non_numeric {
                if let Ok(n) = tok.parse::<usize>() {
                    last_number = Some(n);
                    if !looks_like_date_or_timestamp(tok) {
                        last_plausible_number = Some(n);
                    }
                }
            }
        } else if !tok.is_empty() {
            found_non_numeric = true;
        }
    }
    last_plausible_number.or(last_number)
}

/// Parse a PSNR value from a checkpoint filename.
///
/// Recognises patterns like: `"ckpt_1000_psnr_28.5.json"`, `"model_psnr28.5.bin"`.
pub fn parse_psnr_from_path(path: &str) -> Option<f32> {
    let filename = path.rsplit(['/', '\\']).next().unwrap_or(path);

    let lower = filename.to_ascii_lowercase();

    // Find "psnr" in the filename
    let psnr_pos = lower.find("psnr")?;
    let after = &lower[psnr_pos + 4..];

    // Strip leading separators
    let after = after.trim_start_matches(['_', '-']);

    // Collect digits and decimal point, stopping at non-numeric characters.
    // We do this manually to handle cases like "28.5.json" correctly:
    // allow only one decimal point.
    let mut num_str = String::new();
    let mut seen_dot = false;
    for ch in after.chars() {
        if ch.is_ascii_digit() {
            num_str.push(ch);
        } else if ch == '.' && !seen_dot {
            // Peek ahead: if the character after the dot is a digit, this is a decimal point.
            // Otherwise it is the extension separator — stop.
            seen_dot = true;
            // We'll push it and try; if parsing fails we'll drop the trailing dot below.
            num_str.push(ch);
        } else {
            break;
        }
    }

    // Strip a trailing '.' that was actually the extension separator.
    let num_str = num_str.trim_end_matches('.');

    if num_str.is_empty() {
        return None;
    }

    num_str.parse::<f32>().ok()
}

/// Extract semantic tags from a checkpoint path or filename.
///
/// Recognised tags: `"best"`, `"final"`, `"latest"`, `"epoch_N"`, `"last"`.
pub fn extract_tags_from_path(path: &str) -> Vec<String> {
    let lower = path.to_ascii_lowercase();
    let mut tags = Vec::new();

    if lower.contains("best") {
        tags.push("best".to_string());
    }
    if lower.contains("final") {
        tags.push("final".to_string());
    }
    if lower.contains("latest") {
        tags.push("latest".to_string());
    }
    if lower.contains("last") && !lower.contains("latest") {
        tags.push("last".to_string());
    }

    // epoch_N tags
    let filename = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let stem = if let Some(pos) = filename.rfind('.') {
        &filename[..pos]
    } else {
        filename
    };
    let tokens: Vec<&str> = stem.split(['_', '-']).collect();
    for (i, tok) in tokens.iter().enumerate() {
        if tok.eq_ignore_ascii_case("epoch") {
            if let Some(next) = tokens.get(i + 1) {
                if next.parse::<usize>().is_ok() {
                    tags.push(format!("epoch_{}", next));
                }
            }
        }
    }

    tags
}

/// Generate a descriptive human-readable label for a checkpoint.
pub fn describe_checkpoint(ckpt: &BrowserCheckpoint) -> String {
    let mut parts = vec![format!("step={}", ckpt.step)];

    if let Some(psnr) = ckpt.psnr {
        parts.push(format!("psnr={:.2}", psnr));
    }
    if let Some(loss) = ckpt.loss {
        parts.push(format!("loss={:.4}", loss));
    }
    if let Some(ng) = ckpt.n_gaussians {
        parts.push(format!("gaussians={}", ng));
    }
    if !ckpt.tags.is_empty() {
        parts.push(format!("tags=[{}]", ckpt.tags.join(",")));
    }

    let size_kb = ckpt.file_size_bytes / 1024;
    if size_kb > 0 {
        parts.push(format!("size={}KB", size_kb));
    }

    parts.join(" ")
}

/// Format a list of checkpoints as an ASCII table.
pub fn format_checkpoint_table(checkpoints: &[&BrowserCheckpoint]) -> String {
    let header = format!(
        "{:<8} {:<10} {:<8} {:<10} {:<12} {:<10}",
        "Step", "PSNR", "Loss", "Gaussians", "Size(bytes)", "Tags"
    );
    let separator = "-".repeat(header.len());

    let mut lines = vec![header, separator];

    for ckpt in checkpoints {
        let psnr_str = ckpt
            .psnr
            .map(|p| format!("{:.2}", p))
            .unwrap_or_else(|| "-".to_string());
        let loss_str = ckpt
            .loss
            .map(|l| format!("{:.4}", l))
            .unwrap_or_else(|| "-".to_string());
        let gauss_str = ckpt
            .n_gaussians
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".to_string());
        let tags_str = if ckpt.tags.is_empty() {
            "-".to_string()
        } else {
            ckpt.tags.join(",")
        };

        lines.push(format!(
            "{:<8} {:<10} {:<8} {:<10} {:<12} {:<10}",
            ckpt.step, psnr_str, loss_str, gauss_str, ckpt.file_size_bytes, tags_str
        ));
    }

    lines.join("\n")
}

/// Format a checkpoint diff as a human-readable string.
pub fn format_checkpoint_diff(diff: &CheckpointDiff) -> String {
    let mut lines = Vec::new();

    lines.push(format!("Step delta:     {:+}", diff.step_delta));

    match diff.psnr_delta {
        Some(d) => lines.push(format!("PSNR delta:     {:+.2} dB", d)),
        None => lines.push("PSNR delta:     n/a".to_string()),
    }
    match diff.loss_delta {
        Some(d) => lines.push(format!("Loss delta:     {:+.6}", d)),
        None => lines.push("Loss delta:     n/a".to_string()),
    }
    match diff.gaussians_delta {
        Some(d) => lines.push(format!("Gaussians delta:{:+}", d)),
        None => lines.push("Gaussians delta:n/a".to_string()),
    }
    lines.push(format!("Size delta:     {:+} bytes", diff.size_delta));

    if !diff.tags_added.is_empty() {
        lines.push(format!("Tags added:     {}", diff.tags_added.join(", ")));
    }
    if !diff.tags_removed.is_empty() {
        lines.push(format!("Tags removed:   {}", diff.tags_removed.join(", ")));
    }

    lines.join("\n")
}

/// Format spacing statistics as a human-readable string.
pub fn format_spacing_stats(stats: &SpacingStats) -> String {
    format!(
        "Step gap — mean: {:.1}, min: {}, max: {}, regular: {}, total: {}",
        stats.mean_step_gap,
        stats.min_step_gap,
        stats.max_step_gap,
        stats.is_regular,
        stats.total_steps
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
