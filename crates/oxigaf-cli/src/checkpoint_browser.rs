//! Checkpoint browser — discover, analyze, compare, and select training checkpoints.
//!
//! This module provides tools for browsing and comparing training checkpoints
//! without performing file I/O beyond path inspection.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during checkpoint browsing operations.
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
    /// Construct a `BrowserCheckpoint` by parsing metadata from a path string.
    ///
    /// Only the filename component is used for parsing; no actual I/O is performed.
    pub fn from_path(path: &str) -> Self {
        let step = parse_step_from_path(path).unwrap_or(0);
        let psnr = parse_psnr_from_path(path);
        let tags = extract_tags_from_path(path);
        Self {
            path: path.to_string(),
            step,
            epoch: None,
            psnr,
            loss: None,
            n_gaussians: None,
            timestamp: None,
            file_size_bytes: 0,
            tags,
        }
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

    /// Composite quality score in `[0, 1]`.
    ///
    /// - PSNR available: `psnr / 50.0`
    /// - Loss available: `1.0 - loss.min(1.0)`
    /// - Neither: `0.0`
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
    if trend.len() < 2 {
        return None;
    }

    // Use all points for a simple linear regression (least squares)
    let n = trend.len() as f32;
    let sum_x: f32 = trend.iter().map(|(s, _)| *s as f32).sum();
    let sum_y: f32 = trend.iter().map(|(_, p)| *p).sum();
    let sum_xx: f32 = trend.iter().map(|(s, _)| (*s as f32).powi(2)).sum();
    let sum_xy: f32 = trend.iter().map(|(s, p)| *s as f32 * *p).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < f32::EPSILON {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    if slope <= 0.0 {
        return None;
    }

    // target_psnr = slope * step + intercept → step = (target - intercept) / slope
    let target_step = (target_psnr - intercept) / slope;
    let last_step = trend.last().map(|(s, _)| *s as f32).unwrap_or(0.0);
    if target_step <= last_step {
        return Some(0);
    }

    Some((target_step - last_step).ceil() as usize)
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

/// Parse a step number from a checkpoint filename or path.
///
/// Recognises patterns like: `"ckpt_1000"`, `"step_1000"`, `"checkpoint-1000"`,
/// `"model_1000.json"`. Takes the last occurring number after a recognised prefix.
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
                if let Ok(n) = next.parse::<usize>() {
                    return Some(n);
                }
                // Skip over another keyword ("checkpoint_step_1000")
                if let Some(after) = tokens.get(i + 2) {
                    if let Ok(n) = after.parse::<usize>() {
                        return Some(n);
                    }
                }
            }
        }
        i += 1;
    }

    // Fallback: last purely-numeric token after at least one non-numeric token
    let mut found_non_numeric = false;
    let mut last_number: Option<usize> = None;
    for tok in &tokens {
        let is_numeric = !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit());
        if is_numeric {
            if found_non_numeric {
                if let Ok(n) = tok.parse::<usize>() {
                    last_number = Some(n);
                }
            }
        } else if !tok.is_empty() {
            found_non_numeric = true;
        }
    }
    last_number
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
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // parse_step_from_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_step_ckpt_prefix() {
        assert_eq!(parse_step_from_path("ckpt_1000.json"), Some(1000));
    }

    #[test]
    fn test_parse_step_step_prefix() {
        assert_eq!(parse_step_from_path("step_500"), Some(500));
    }

    #[test]
    fn test_parse_step_checkpoint_dash() {
        assert_eq!(parse_step_from_path("checkpoint-200"), Some(200));
    }

    #[test]
    fn test_parse_step_checkpoint_underscore() {
        assert_eq!(parse_step_from_path("checkpoint_300.bin"), Some(300));
    }

    #[test]
    fn test_parse_step_model_prefix() {
        assert_eq!(parse_step_from_path("model_1000.json"), Some(1000));
    }

    #[test]
    fn test_parse_step_checkpoint_step_compound() {
        assert_eq!(
            parse_step_from_path("checkpoint_step_12345.json"),
            Some(12345)
        );
    }

    #[test]
    fn test_parse_step_no_number() {
        assert_eq!(parse_step_from_path("final_model.json"), None);
    }

    #[test]
    fn test_parse_step_empty() {
        assert_eq!(parse_step_from_path(""), None);
    }

    #[test]
    fn test_parse_step_with_directory() {
        assert_eq!(
            parse_step_from_path("/run/train/ckpt_2000.json"),
            Some(2000)
        );
    }

    #[test]
    fn test_parse_step_fallback_trailing_number() {
        // "model_best_500" — last number after non-numeric tokens
        assert_eq!(parse_step_from_path("model_best_500"), Some(500));
    }

    // -----------------------------------------------------------------------
    // parse_psnr_from_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_psnr_basic() {
        let result = parse_psnr_from_path("ckpt_1000_psnr_28.5.json");
        assert!(result.is_some());
        let v = result.unwrap();
        assert!((v - 28.5).abs() < 0.01, "expected 28.5, got {}", v);
    }

    #[test]
    fn test_parse_psnr_no_psnr() {
        assert!(parse_psnr_from_path("ckpt_1000.json").is_none());
    }

    #[test]
    fn test_parse_psnr_attached() {
        // "model_psnr28.5.bin"
        let result = parse_psnr_from_path("model_psnr28.5.bin");
        assert!(result.is_some());
        let v = result.unwrap();
        assert!((v - 28.5).abs() < 0.01, "expected 28.5, got {}", v);
    }

    #[test]
    fn test_parse_psnr_dash_separator() {
        let result = parse_psnr_from_path("ckpt-psnr-32.1.json");
        assert!(result.is_some());
        let v = result.unwrap();
        assert!((v - 32.1).abs() < 0.01);
    }

    #[test]
    fn test_parse_psnr_integer() {
        let result = parse_psnr_from_path("ckpt_psnr_30.json");
        assert!(result.is_some());
        let v = result.unwrap();
        assert!((v - 30.0).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // extract_tags_from_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_tags_best() {
        let tags = extract_tags_from_path("ckpt_best_1000.json");
        assert!(tags.contains(&"best".to_string()));
    }

    #[test]
    fn test_extract_tags_final() {
        let tags = extract_tags_from_path("model_final.bin");
        assert!(tags.contains(&"final".to_string()));
    }

    #[test]
    fn test_extract_tags_latest() {
        let tags = extract_tags_from_path("checkpoint_latest.json");
        assert!(tags.contains(&"latest".to_string()));
    }

    #[test]
    fn test_extract_tags_epoch() {
        let tags = extract_tags_from_path("ckpt_epoch_10_step_1000.json");
        assert!(tags.contains(&"epoch_10".to_string()));
    }

    #[test]
    fn test_extract_tags_empty() {
        let tags = extract_tags_from_path("ckpt_1000.json");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_extract_tags_multiple() {
        let tags = extract_tags_from_path("model_best_final.json");
        assert!(tags.contains(&"best".to_string()));
        assert!(tags.contains(&"final".to_string()));
    }

    // -----------------------------------------------------------------------
    // BrowserCheckpoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_path_step_parsed() {
        let c = BrowserCheckpoint::from_path("ckpt_step_5000.json");
        assert_eq!(c.step, 5000);
    }

    #[test]
    fn test_from_path_zero_step_fallback() {
        let c = BrowserCheckpoint::from_path("model.json");
        assert_eq!(c.step, 0);
    }

    #[test]
    fn test_is_best_true() {
        let c = BrowserCheckpoint::from_path("/checkpoints/ckpt_best_1000.json");
        assert!(c.is_best());
    }

    #[test]
    fn test_is_best_false() {
        let c = BrowserCheckpoint::from_path("ckpt_1000.json");
        assert!(!c.is_best());
    }

    #[test]
    fn test_is_final_true() {
        let c = BrowserCheckpoint::from_path("model_final.json");
        assert!(c.is_final());
    }

    #[test]
    fn test_is_final_false() {
        let c = BrowserCheckpoint::from_path("ckpt_1000.json");
        assert!(!c.is_final());
    }

    #[test]
    fn test_quality_score_with_psnr() {
        let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
        c.psnr = Some(30.0);
        let score = c.quality_score();
        assert!((score - 0.6).abs() < 1e-5, "expected 0.6, got {}", score);
    }

    #[test]
    fn test_quality_score_with_loss() {
        let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
        c.loss = Some(0.5);
        let score = c.quality_score();
        assert!((score - 0.5).abs() < 1e-5, "expected 0.5, got {}", score);
    }

    #[test]
    fn test_quality_score_no_metrics() {
        let c = BrowserCheckpoint::from_path("ckpt_1000.json");
        assert!((c.quality_score() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_quality_score_psnr_priority_over_loss() {
        let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
        c.psnr = Some(25.0);
        c.loss = Some(0.1);
        // Should use PSNR: 25/50 = 0.5, not 1-0.1 = 0.9
        let score = c.quality_score();
        assert!((score - 0.5).abs() < 1e-5, "expected 0.5, got {}", score);
    }

    // -----------------------------------------------------------------------
    // CheckpointBrowser construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_browser_empty() {
        let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
        assert!(browser.is_empty());
        assert_eq!(browser.len(), 0);
    }

    #[test]
    fn test_browser_from_paths() {
        let paths = vec!["ckpt_100.json".to_string(), "ckpt_200.json".to_string()];
        let browser = CheckpointBrowser::from_paths(paths, BrowserConfig::default());
        assert_eq!(browser.len(), 2);
    }

    #[test]
    fn test_browser_total_size_bytes() {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
        c1.file_size_bytes = 1024;
        let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
        c2.file_size_bytes = 2048;
        let browser = CheckpointBrowser::new(vec![c1, c2], BrowserConfig::default());
        assert_eq!(browser.total_size_bytes(), 3072);
    }

    #[test]
    fn test_browser_step_range() {
        let c1 = BrowserCheckpoint::from_path("ckpt_100.json");
        let c2 = BrowserCheckpoint::from_path("ckpt_500.json");
        let c3 = BrowserCheckpoint::from_path("ckpt_300.json");
        let browser = CheckpointBrowser::new(vec![c1, c2, c3], BrowserConfig::default());
        assert_eq!(browser.step_range(), Some((100, 500)));
    }

    #[test]
    fn test_browser_step_range_empty() {
        let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
        assert!(browser.step_range().is_none());
    }

    // -----------------------------------------------------------------------
    // browse() — sort
    // -----------------------------------------------------------------------

    fn make_checkpoints_with_psnr() -> Vec<BrowserCheckpoint> {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
        c1.psnr = Some(25.0);
        let mut c2 = BrowserCheckpoint::from_path("ckpt_300.json");
        c2.psnr = Some(30.0);
        let mut c3 = BrowserCheckpoint::from_path("ckpt_200.json");
        c3.psnr = Some(27.5);
        vec![c1, c2, c3]
    }

    #[test]
    fn test_browse_sort_by_step() {
        let config = BrowserConfig {
            sort_by: BrowserSort::ByStep,
            ..Default::default()
        };
        let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
        let result = browser.browse();
        assert_eq!(result[0].step, 100);
        assert_eq!(result[1].step, 200);
        assert_eq!(result[2].step, 300);
    }

    #[test]
    fn test_browse_sort_by_step_desc() {
        let config = BrowserConfig {
            sort_by: BrowserSort::ByStepDesc,
            ..Default::default()
        };
        let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
        let result = browser.browse();
        assert_eq!(result[0].step, 300);
        assert_eq!(result[1].step, 200);
        assert_eq!(result[2].step, 100);
    }

    #[test]
    fn test_browse_sort_by_psnr() {
        let config = BrowserConfig {
            sort_by: BrowserSort::ByPsnr,
            ..Default::default()
        };
        let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
        let result = browser.browse();
        assert!((result[0].psnr.unwrap() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_browse_max_display() {
        let config = BrowserConfig {
            max_display: 2,
            ..Default::default()
        };
        let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
        let result = browser.browse();
        assert_eq!(result.len(), 2);
    }

    // -----------------------------------------------------------------------
    // browse() — filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_browse_filter_step_range() {
        let mut config = BrowserConfig::default();
        config.filter.min_step = Some(150);
        config.filter.max_step = Some(250);
        let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
        let result = browser.browse();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].step, 200);
    }

    #[test]
    fn test_browse_filter_min_psnr() {
        let mut config = BrowserConfig::default();
        config.filter.min_psnr = Some(28.0);
        let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
        let result = browser.browse();
        // Only step=300 (psnr=30.0) passes; step=200 (27.5) does not
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].step, 300);
    }

    #[test]
    fn test_browse_filter_max_loss() {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
        c1.loss = Some(0.8);
        let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
        c2.loss = Some(0.3);
        let mut config = BrowserConfig::default();
        config.filter.max_loss = Some(0.5);
        let browser = CheckpointBrowser::new(vec![c1, c2], config);
        let result = browser.browse();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].step, 200);
    }

    #[test]
    fn test_browse_filter_tags_required() {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_best_100.json");
        c1.psnr = Some(25.0);
        let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
        c2.psnr = Some(27.0);
        let mut config = BrowserConfig::default();
        config.filter.tags_required = vec!["best".to_string()];
        let browser = CheckpointBrowser::new(vec![c1, c2], config);
        let result = browser.browse();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_best());
    }

    #[test]
    fn test_browse_filter_tags_excluded() {
        let c1 = BrowserCheckpoint::from_path("ckpt_best_100.json");
        let c2 = BrowserCheckpoint::from_path("ckpt_200.json");
        let mut config = BrowserConfig::default();
        config.filter.tags_excluded = vec!["best".to_string()];
        let browser = CheckpointBrowser::new(vec![c1, c2], config);
        let result = browser.browse();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].step, 200);
    }

    // -----------------------------------------------------------------------
    // find_best
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_best_by_psnr() {
        let browser =
            CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
        let best = browser.find_best().expect("should find best");
        assert!((best.psnr.unwrap() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_find_best_by_loss_when_no_psnr() {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
        c1.loss = Some(0.8);
        let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
        c2.loss = Some(0.2);
        let browser = CheckpointBrowser::new(vec![c1, c2], BrowserConfig::default());
        let best = browser.find_best().expect("should find best");
        assert_eq!(best.step, 200);
    }

    #[test]
    fn test_find_best_empty() {
        let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
        assert!(browser.find_best().is_none());
    }

    // -----------------------------------------------------------------------
    // find_latest
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_latest() {
        let browser =
            CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
        let latest = browser.find_latest().expect("should find latest");
        assert_eq!(latest.step, 300);
    }

    #[test]
    fn test_find_latest_empty() {
        let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
        assert!(browser.find_latest().is_none());
    }

    // -----------------------------------------------------------------------
    // find_at_step / find_nearest_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_at_step_exact() {
        let browser =
            CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
        let found = browser.find_at_step(200).expect("should find step 200");
        assert_eq!(found.step, 200);
    }

    #[test]
    fn test_find_at_step_nearest() {
        let browser =
            CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
        // 175 is between 100 and 200; nearest is 200
        let found = browser.find_at_step(175).expect("should find nearest");
        assert_eq!(found.step, 200);
    }

    #[test]
    fn test_find_nearest_step() {
        let browser =
            CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
        let found = browser.find_nearest_step(260).expect("should find nearest");
        assert_eq!(found.step, 300);
    }

    // -----------------------------------------------------------------------
    // at_percentile
    // -----------------------------------------------------------------------

    #[test]
    fn test_at_percentile_zero() {
        let browser =
            CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
        let ckpt = browser.at_percentile(0.0).expect("should return first");
        assert_eq!(ckpt.step, 100);
    }

    #[test]
    fn test_at_percentile_one() {
        let browser =
            CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
        let ckpt = browser.at_percentile(1.0).expect("should return last");
        assert_eq!(ckpt.step, 300);
    }

    #[test]
    fn test_at_percentile_half() {
        let browser =
            CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
        let ckpt = browser.at_percentile(0.5).expect("should return middle");
        assert_eq!(ckpt.step, 200);
    }

    #[test]
    fn test_at_percentile_empty() {
        let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
        assert!(browser.at_percentile(0.5).is_none());
    }

    // -----------------------------------------------------------------------
    // compare_checkpoints
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_step_delta() {
        let a = BrowserCheckpoint::from_path("ckpt_100.json");
        let b = BrowserCheckpoint::from_path("ckpt_300.json");
        let diff = compare_checkpoints(&a, &b);
        assert_eq!(diff.step_delta, 200);
    }

    #[test]
    fn test_compare_psnr_delta() {
        let mut a = BrowserCheckpoint::from_path("ckpt_100.json");
        a.psnr = Some(25.0);
        let mut b = BrowserCheckpoint::from_path("ckpt_200.json");
        b.psnr = Some(28.0);
        let diff = compare_checkpoints(&a, &b);
        assert!(diff.psnr_delta.is_some());
        assert!((diff.psnr_delta.unwrap() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_compare_loss_delta() {
        let mut a = BrowserCheckpoint::from_path("ckpt_100.json");
        a.loss = Some(0.5);
        let mut b = BrowserCheckpoint::from_path("ckpt_200.json");
        b.loss = Some(0.3);
        let diff = compare_checkpoints(&a, &b);
        assert!(diff.loss_delta.is_some());
        assert!((diff.loss_delta.unwrap() - (-0.2)).abs() < 0.001);
    }

    #[test]
    fn test_compare_tags_added_removed() {
        let a = BrowserCheckpoint::from_path("ckpt_best_100.json");
        let b = BrowserCheckpoint::from_path("ckpt_final_200.json");
        let diff = compare_checkpoints(&a, &b);
        assert!(diff.tags_added.contains(&"final".to_string()));
        assert!(diff.tags_removed.contains(&"best".to_string()));
    }

    #[test]
    fn test_compare_size_delta() {
        let mut a = BrowserCheckpoint::from_path("ckpt_100.json");
        a.file_size_bytes = 1000;
        let mut b = BrowserCheckpoint::from_path("ckpt_200.json");
        b.file_size_bytes = 2500;
        let diff = compare_checkpoints(&a, &b);
        assert_eq!(diff.size_delta, 1500);
    }

    // -----------------------------------------------------------------------
    // psnr_trend
    // -----------------------------------------------------------------------

    #[test]
    fn test_psnr_trend_sorted_by_step() {
        let ckpts = make_checkpoints_with_psnr();
        let trend = psnr_trend(&ckpts);
        assert_eq!(trend.len(), 3);
        // Should be sorted by step ascending
        assert!(trend[0].0 < trend[1].0);
        assert!(trend[1].0 < trend[2].0);
    }

    #[test]
    fn test_psnr_trend_excludes_no_psnr() {
        let mut ckpts = make_checkpoints_with_psnr();
        let no_psnr = BrowserCheckpoint::from_path("ckpt_400.json");
        ckpts.push(no_psnr);
        let trend = psnr_trend(&ckpts);
        assert_eq!(trend.len(), 3); // Only 3 have PSNR
    }

    #[test]
    fn test_psnr_trend_empty() {
        let trend = psnr_trend(&[]);
        assert!(trend.is_empty());
    }

    // -----------------------------------------------------------------------
    // find_psnr_elbow
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_psnr_elbow_basic() {
        // Create a set with diminishing returns
        let mut ckpts = Vec::new();
        let psnrs = [20.0f32, 25.0, 28.0, 29.5, 29.9, 30.0];
        for (i, &p) in psnrs.iter().enumerate() {
            let mut c = BrowserCheckpoint::from_path(&format!("ckpt_{}.json", (i + 1) * 1000));
            c.psnr = Some(p);
            ckpts.push(c);
        }
        let elbow = find_psnr_elbow(&ckpts);
        assert!(elbow.is_some());
        // The elbow should be somewhere in the middle of training
        let e = elbow.unwrap();
        assert!(e > 0);
    }

    #[test]
    fn test_find_psnr_elbow_too_few() {
        let mut ckpts = Vec::new();
        for i in 0..2 {
            let mut c = BrowserCheckpoint::from_path(&format!("ckpt_{}.json", (i + 1) * 1000));
            c.psnr = Some(25.0 + i as f32);
            ckpts.push(c);
        }
        assert!(find_psnr_elbow(&ckpts).is_none());
    }

    // -----------------------------------------------------------------------
    // estimate_steps_to_psnr
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_steps_to_psnr_improving() {
        let mut ckpts = Vec::new();
        // Linear PSNR: step=1000→psnr=20, step=2000→psnr=25
        let mut c1 = BrowserCheckpoint::from_path("ckpt_1000.json");
        c1.psnr = Some(20.0);
        let mut c2 = BrowserCheckpoint::from_path("ckpt_2000.json");
        c2.psnr = Some(25.0);
        ckpts.push(c1);
        ckpts.push(c2);
        // slope = 5/1000 per step, to reach 30 need 1000 more steps
        let estimate = estimate_steps_to_psnr(&ckpts, 30.0);
        assert!(estimate.is_some());
        let extra = estimate.unwrap();
        // Should be approximately 1000 more steps
        assert!(
            extra > 500 && extra < 2000,
            "expected ~1000 extra steps, got {}",
            extra
        );
    }

    #[test]
    fn test_estimate_steps_to_psnr_declining() {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_1000.json");
        c1.psnr = Some(30.0);
        let mut c2 = BrowserCheckpoint::from_path("ckpt_2000.json");
        c2.psnr = Some(25.0);
        let ckpts = vec![c1, c2];
        // Declining PSNR → None
        assert!(estimate_steps_to_psnr(&ckpts, 35.0).is_none());
    }

    #[test]
    fn test_estimate_steps_to_psnr_too_few() {
        let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
        c.psnr = Some(25.0);
        assert!(estimate_steps_to_psnr(&[c], 30.0).is_none());
    }

    #[test]
    fn test_estimate_steps_already_reached() {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_1000.json");
        c1.psnr = Some(20.0);
        let mut c2 = BrowserCheckpoint::from_path("ckpt_2000.json");
        c2.psnr = Some(30.0);
        let ckpts = vec![c1, c2];
        // Target already reached
        let estimate = estimate_steps_to_psnr(&ckpts, 25.0);
        assert_eq!(estimate, Some(0));
    }

    // -----------------------------------------------------------------------
    // checkpoint_spacing_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_spacing_stats_regular() {
        let paths = [
            "ckpt_100.json",
            "ckpt_200.json",
            "ckpt_300.json",
            "ckpt_400.json",
        ];
        let ckpts: Vec<_> = paths
            .iter()
            .map(|p| BrowserCheckpoint::from_path(p))
            .collect();
        let stats = checkpoint_spacing_stats(&ckpts);
        assert!(stats.is_regular, "evenly spaced should be regular");
        assert!((stats.mean_step_gap - 100.0).abs() < 0.01);
        assert_eq!(stats.min_step_gap, 100);
        assert_eq!(stats.max_step_gap, 100);
        assert_eq!(stats.total_steps, 300);
    }

    #[test]
    fn test_spacing_stats_irregular() {
        let paths = ["ckpt_100.json", "ckpt_200.json", "ckpt_800.json"];
        let ckpts: Vec<_> = paths
            .iter()
            .map(|p| BrowserCheckpoint::from_path(p))
            .collect();
        let stats = checkpoint_spacing_stats(&ckpts);
        assert!(!stats.is_regular, "irregular spacing should not be regular");
    }

    #[test]
    fn test_spacing_stats_single() {
        let ckpts = vec![BrowserCheckpoint::from_path("ckpt_100.json")];
        let stats = checkpoint_spacing_stats(&ckpts);
        assert_eq!(stats.total_steps, 0);
    }

    #[test]
    fn test_spacing_stats_empty() {
        let stats = checkpoint_spacing_stats(&[]);
        assert!(stats.is_regular);
        assert_eq!(stats.total_steps, 0);
    }

    // -----------------------------------------------------------------------
    // describe_checkpoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_describe_checkpoint_non_empty() {
        let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
        c.psnr = Some(28.5);
        let desc = describe_checkpoint(&c);
        assert!(!desc.is_empty());
        assert!(desc.contains("step=1000"));
        assert!(desc.contains("psnr=28.50"));
    }

    #[test]
    fn test_describe_checkpoint_minimal() {
        let c = BrowserCheckpoint::from_path("ckpt_0.json");
        let desc = describe_checkpoint(&c);
        assert!(!desc.is_empty());
        assert!(desc.contains("step=0"));
    }

    // -----------------------------------------------------------------------
    // format_checkpoint_table
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_checkpoint_table_non_empty() {
        let ckpts = make_checkpoints_with_psnr();
        let refs: Vec<&BrowserCheckpoint> = ckpts.iter().collect();
        let table = format_checkpoint_table(&refs);
        assert!(!table.is_empty());
        assert!(table.contains("Step"));
        assert!(table.contains("PSNR"));
    }

    #[test]
    fn test_format_checkpoint_table_empty() {
        let table = format_checkpoint_table(&[]);
        assert!(!table.is_empty()); // Still has header
        assert!(table.contains("Step"));
    }

    // -----------------------------------------------------------------------
    // format_checkpoint_diff
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_checkpoint_diff_non_empty() {
        let a = BrowserCheckpoint::from_path("ckpt_100.json");
        let b = BrowserCheckpoint::from_path("ckpt_200.json");
        let diff = compare_checkpoints(&a, &b);
        let formatted = format_checkpoint_diff(&diff);
        assert!(!formatted.is_empty());
        assert!(formatted.contains("Step delta"));
    }

    // -----------------------------------------------------------------------
    // format_spacing_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_spacing_stats_non_empty() {
        let paths = ["ckpt_100.json", "ckpt_200.json", "ckpt_300.json"];
        let ckpts: Vec<_> = paths
            .iter()
            .map(|p| BrowserCheckpoint::from_path(p))
            .collect();
        let stats = checkpoint_spacing_stats(&ckpts);
        let formatted = format_spacing_stats(&stats);
        assert!(!formatted.is_empty());
        assert!(formatted.contains("mean"));
    }

    // -----------------------------------------------------------------------
    // BrowserError display
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_no_checkpoints_display() {
        let e = BrowserError::NoCheckpoints("/tmp/ckpts".to_string());
        let msg = format!("{}", e);
        assert!(msg.contains("/tmp/ckpts"));
    }

    #[test]
    fn test_error_too_few_checkpoints_display() {
        let e = BrowserError::TooFewCheckpoints(1);
        let msg = format!("{}", e);
        assert!(msg.contains("1"));
    }

    #[test]
    fn test_error_checkpoint_not_found() {
        let e = BrowserError::CheckpointNotFound("ckpt_999.json".to_string());
        let msg = format!("{}", e);
        assert!(msg.contains("ckpt_999.json"));
    }

    // -----------------------------------------------------------------------
    // browse() — sort by loss and quality score
    // -----------------------------------------------------------------------

    #[test]
    fn test_browse_sort_by_loss() {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
        c1.loss = Some(0.8);
        let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
        c2.loss = Some(0.2);
        let mut c3 = BrowserCheckpoint::from_path("ckpt_300.json");
        c3.loss = Some(0.5);
        let config = BrowserConfig {
            sort_by: BrowserSort::ByLoss,
            ..Default::default()
        };
        let browser = CheckpointBrowser::new(vec![c1, c2, c3], config);
        let result = browser.browse();
        assert!((result[0].loss.unwrap() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_browse_sort_by_quality_score() {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
        c1.psnr = Some(20.0);
        let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
        c2.psnr = Some(35.0);
        let config = BrowserConfig {
            sort_by: BrowserSort::ByQualityScore,
            ..Default::default()
        };
        let browser = CheckpointBrowser::new(vec![c1, c2], config);
        let result = browser.browse();
        assert_eq!(result[0].step, 200); // higher PSNR = higher quality
    }

    #[test]
    fn test_browse_sort_by_file_size() {
        let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
        c1.file_size_bytes = 500;
        let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
        c2.file_size_bytes = 2000;
        let config = BrowserConfig {
            sort_by: BrowserSort::ByFileSize,
            ..Default::default()
        };
        let browser = CheckpointBrowser::new(vec![c1, c2], config);
        let result = browser.browse();
        assert_eq!(result[0].step, 200); // largest first
    }
}
