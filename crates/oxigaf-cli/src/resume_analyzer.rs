//! Checkpoint resume analysis for training recovery.
//!
//! Scans a directory for checkpoint files, scores them based on PSNR,
//! loss stability, and Gaussian count stability, then recommends the
//! best checkpoint to resume from.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during checkpoint analysis.
#[derive(Debug)]
pub enum ResumeError {
    /// No checkpoint files were found in the scanned directory.
    NoCheckpointsFound,
    /// The specified directory does not exist.
    DirectoryNotFound(PathBuf),
    /// An I/O error occurred during scanning or metadata reads.
    IoError(std::io::Error),
    /// The scorer produced an empty scores vector when checkpoints exist.
    EmptyScores,
}

impl std::fmt::Display for ResumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCheckpointsFound => write!(f, "No checkpoint files found in directory"),
            Self::DirectoryNotFound(p) => {
                write!(f, "Directory not found: {}", p.display())
            }
            Self::IoError(e) => write!(f, "I/O error: {e}"),
            Self::EmptyScores => write!(
                f,
                "Scorer returned empty scores for non-empty checkpoint list"
            ),
        }
    }
}

impl std::error::Error for ResumeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ResumeError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

// ---------------------------------------------------------------------------
// CheckpointMetadata
// ---------------------------------------------------------------------------

/// Metadata for a single checkpoint file.
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Absolute path to the checkpoint file.
    pub path: PathBuf,
    /// Training step extracted from the filename.
    pub step: usize,
    /// Peak signal-to-noise ratio from filename or header; -1.0 if unknown.
    pub psnr: f32,
    /// Total training loss; -1.0 if unknown.
    pub loss: f32,
    /// Number of 3D Gaussians; 0 if unknown.
    pub num_gaussians: usize,
    /// File modification time as Unix seconds; 0 if unavailable.
    pub timestamp_secs: u64,
    /// File size in bytes.
    pub file_size_bytes: u64,
}

impl CheckpointMetadata {
    /// Parse a training step from common checkpoint filename patterns.
    ///
    /// Supported patterns (case-insensitive component matching):
    /// - `"checkpoint_step_12345.json"` → `Some(12345)`
    /// - `"checkpoint_12345.bin"`       → `Some(12345)`
    /// - `"ckpt_12345"`                 → `Some(12345)`
    /// - `"step_12345"`                 → `Some(12345)`
    ///
    /// Returns `None` if no step number can be extracted.
    pub fn parse_step_from_filename(name: &str) -> Option<usize> {
        // Strip the extension (everything after the last '.')
        let stem = if let Some(pos) = name.rfind('.') {
            &name[..pos]
        } else {
            name
        };

        // Tokenise on '_' and '-'
        let tokens: Vec<&str> = stem.split(['_', '-']).collect();

        // Walk forward; when we see a keyword token, try to parse the *next* token
        // as a decimal integer. If no keyword precedes a number, accept the last
        // bare number token that follows any recognised prefix at position ≥ 1.
        let keywords = ["checkpoint", "ckpt", "step", "epoch"];

        let mut i = 0;
        while i < tokens.len() {
            let tok_lower = tokens[i].to_ascii_lowercase();
            if keywords.iter().any(|k| *k == tok_lower) {
                // Try the immediate next token first
                if let Some(next) = tokens.get(i + 1) {
                    if let Ok(n) = next.parse::<usize>() {
                        return Some(n);
                    }
                    // The next token might itself be a keyword (e.g. "checkpoint_step_12345")
                    // — skip it and check the one after
                    if let Some(after) = tokens.get(i + 2) {
                        if let Ok(n) = after.parse::<usize>() {
                            return Some(n);
                        }
                    }
                }
            }
            i += 1;
        }

        // Fallback: return the rightmost purely-numeric token that appears after
        // at least one non-numeric token (to avoid treating a bare number name
        // as if it carries step information without any keyword context).
        let mut found_non_numeric = false;
        let mut last_number: Option<usize> = None;
        for tok in &tokens {
            if tok.chars().all(|c| c.is_ascii_digit()) && !tok.is_empty() {
                if found_non_numeric {
                    if let Ok(n) = tok.parse::<usize>() {
                        last_number = Some(n);
                    }
                }
            } else {
                found_non_numeric = true;
            }
        }
        last_number
    }

    /// Format a one-line summary of this checkpoint.
    ///
    /// Example output:
    /// `"step=12345 psnr=28.3 loss=0.012 gaussians=50000 size=12.3MB"`
    pub fn format_summary_line(&self) -> String {
        let psnr_str = if self.psnr < 0.0 {
            "unknown".to_string()
        } else {
            format!("{:.1}", self.psnr)
        };

        let loss_str = if self.loss < 0.0 {
            "unknown".to_string()
        } else {
            format!("{:.4}", self.loss)
        };

        let size_mb = self.file_size_bytes as f64 / (1024.0 * 1024.0);

        format!(
            "step={} psnr={} loss={} gaussians={} size={:.1}MB",
            self.step, psnr_str, loss_str, self.num_gaussians, size_mb,
        )
    }
}

// ---------------------------------------------------------------------------
// CheckpointScanner
// ---------------------------------------------------------------------------

/// Scans a directory tree for checkpoint files.
pub struct CheckpointScanner {
    /// File extensions to consider as checkpoints.
    pub extensions: Vec<String>,
    /// Minimum step (inclusive) to include.
    pub min_step: usize,
    /// Maximum step (inclusive) to include; `usize::MAX` means no upper limit.
    pub max_step: usize,
}

impl Default for CheckpointScanner {
    fn default() -> Self {
        Self {
            extensions: vec![
                "json".to_string(),
                "bin".to_string(),
                "safetensors".to_string(),
                "ckpt".to_string(),
            ],
            min_step: 0,
            max_step: usize::MAX,
        }
    }
}

impl CheckpointScanner {
    /// Create a scanner with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the accepted file extensions.
    pub fn with_extensions(mut self, exts: Vec<String>) -> Self {
        self.extensions = exts;
        self
    }

    /// Restrict scanning to steps within `[min, max]`.
    pub fn with_step_range(mut self, min: usize, max: usize) -> Self {
        self.min_step = min;
        self.max_step = max;
        self
    }

    /// Scan `dir` for checkpoint files, returning them sorted by step ascending.
    ///
    /// Files whose step cannot be parsed from their name are silently skipped.
    pub fn scan(&self, dir: &Path) -> Result<Vec<CheckpointMetadata>, ResumeError> {
        if !dir.exists() {
            return Err(ResumeError::DirectoryNotFound(dir.to_path_buf()));
        }

        let entries = std::fs::read_dir(dir)?;
        let mut checkpoints: Vec<CheckpointMetadata> = Vec::new();

        for entry_result in entries {
            let entry = entry_result?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            // Check extension
            let ext_matches = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|ext| {
                    self.extensions
                        .iter()
                        .any(|allowed| allowed.eq_ignore_ascii_case(ext))
                })
                .unwrap_or(false);

            if !ext_matches {
                continue;
            }

            // Try to parse a step from the file name
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();

            let step = match CheckpointMetadata::parse_step_from_filename(file_name) {
                Some(s) => s,
                None => continue,
            };

            // Apply step range filter
            if step < self.min_step || step > self.max_step {
                continue;
            }

            // Gather filesystem metadata
            let fs_meta = std::fs::metadata(&path)?;
            let file_size_bytes = fs_meta.len();
            let timestamp_secs = fs_meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let (psnr, loss, num_gaussians) = Self::probe_metadata(&path, file_name);

            checkpoints.push(CheckpointMetadata {
                path,
                step,
                psnr,
                loss,
                num_gaussians,
                timestamp_secs,
                file_size_bytes,
            });
        }

        // Sort ascending by step
        checkpoints.sort_by_key(|c| c.step);
        Ok(checkpoints)
    }

    /// Best-effort extraction of `(psnr, loss, num_gaussians)` for one
    /// checkpoint file, returning the documented "unknown" sentinels
    /// (`(-1.0, -1.0, 0)`) for anything that cannot be determined.
    ///
    /// Real checkpoints written by `Trainer::save_checkpoint`
    /// (`oxigaf_trainer::checkpoint::CheckpointData`) are JSON with a
    /// `metrics_history` of `MetricEntry { iteration, psnr, ssim, loss }`
    /// and a `positions` array whose length is the Gaussian count — the
    /// most recent entry gives the checkpoint's PSNR/loss directly. When the
    /// file isn't a parseable `CheckpointData` JSON (a `.bin`/`.safetensors`
    /// checkpoint, or one predating `metrics_history`), fall back to a PSNR
    /// value embedded in the filename itself (`checkpoint_browser`'s
    /// `"..._psnr_28.5..."` convention).
    ///
    /// Cost note: `load_checkpoint` fully deserializes the checkpoint —
    /// positions/rotations/scales/opacities/SH coefficients *and* both Adam
    /// moment vectors — not just the three facts read here, so on a
    /// real multi-hundred-thousand-Gaussian run each `.json` checkpoint can
    /// cost hundreds of MB to parse. This is still the right trade-off for a
    /// resume-checkpoint scan (typically a handful of files, and the
    /// alternative is fabricating psnr/loss/num_gaussians), but avoid
    /// calling this over a directory with many large checkpoints in a hot
    /// path.
    fn probe_metadata(path: &Path, file_name: &str) -> (f32, f32, usize) {
        let is_json = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("json"))
            .unwrap_or(false);

        if is_json {
            if let Ok(data) = oxigaf_trainer::checkpoint::load_checkpoint(path) {
                let num_gaussians = data.positions.len();
                if let Some(latest) = data.metrics_history.last() {
                    return (latest.psnr, latest.loss, num_gaussians);
                }
                // Parsed fine but carries no metrics history (e.g. a
                // checkpoint saved before the first evaluation) — the
                // Gaussian count is still genuine even though psnr/loss
                // are not recorded anywhere in the file.
                let psnr =
                    crate::checkpoint_browser::parse_psnr_from_path(file_name).unwrap_or(-1.0);
                return (psnr, -1.0, num_gaussians);
            }
        }

        // Non-JSON checkpoint formats, or a JSON file that isn't a valid
        // `CheckpointData` (corrupted, or some other JSON payload entirely):
        // fall back to a PSNR embedded in the filename, if any.
        let psnr = crate::checkpoint_browser::parse_psnr_from_path(file_name).unwrap_or(-1.0);
        (psnr, -1.0, 0)
    }
}

// ---------------------------------------------------------------------------
// ScoringWeights / CheckpointScorer
// ---------------------------------------------------------------------------

/// Weights controlling how each quality signal contributes to the final score.
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Contribution from PSNR (higher PSNR → better).
    pub psnr_weight: f32,
    /// Contribution from loss stability (lower variance → better).
    pub loss_stability_weight: f32,
    /// Contribution from Gaussian-count stability (lower variance → better).
    pub gaussian_stability_weight: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            psnr_weight: 0.5,
            loss_stability_weight: 0.3,
            gaussian_stability_weight: 0.2,
        }
    }
}

/// Scores checkpoints so that callers can pick the best one to resume from.
#[derive(Default)]
pub struct CheckpointScorer {
    /// Weighting configuration.
    pub weights: ScoringWeights,
}

impl CheckpointScorer {
    /// Create a scorer with default weights.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a scorer with custom weights.
    pub fn with_weights(weights: ScoringWeights) -> Self {
        Self { weights }
    }

    /// Score all checkpoints. Returns a `Vec<f32>` of the same length.
    ///
    /// A higher score indicates a better starting point for resuming training.
    ///
    /// # Algorithm
    /// 1. **PSNR score** (normalised to \[0,1\]):
    ///    - If all PSNR values are -1.0 (unknown), fall back to step-based proxy
    ///      (later step → higher score).
    ///    - Otherwise `(psnr − min) / (max − min)`; unknown entries receive 0.
    /// 2. **Loss stability** (normalised to \[0,1\]):
    ///    - For each checkpoint compute variance of `loss` over ±3 neighbours.
    ///    - Stability = `1 / (1 + variance)`.  Normalize across all checkpoints.
    ///    - If all losses are -1.0, every entry gets 1.0 stability.
    /// 3. **Gaussian-count stability**: same window-variance approach.
    /// 4. Final score = weighted sum of the three normalised scores.
    pub fn score(&self, checkpoints: &[CheckpointMetadata]) -> Vec<f32> {
        let n = checkpoints.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![1.0];
        }

        // --- PSNR scores ---
        let psnr_scores = self.compute_psnr_scores(checkpoints);

        // --- Loss stability scores ---
        let loss_stability = self.compute_stability_scores(checkpoints, |c| {
            if c.loss >= 0.0 {
                c.loss
            } else {
                f32::NAN
            }
        });

        // --- Gaussian stability scores ---
        let gauss_stability = self.compute_stability_scores(checkpoints, |c| {
            if c.num_gaussians > 0 {
                c.num_gaussians as f32
            } else {
                f32::NAN
            }
        });

        let w = &self.weights;
        (0..n)
            .map(|i| {
                w.psnr_weight * psnr_scores[i]
                    + w.loss_stability_weight * loss_stability[i]
                    + w.gaussian_stability_weight * gauss_stability[i]
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn compute_psnr_scores(&self, checkpoints: &[CheckpointMetadata]) -> Vec<f32> {
        let _n = checkpoints.len();
        let known: Vec<f32> = checkpoints
            .iter()
            .filter(|c| c.psnr >= 0.0)
            .map(|c| c.psnr)
            .collect();

        if known.is_empty() {
            // Fall back to step-based proxy
            let max_step = checkpoints.iter().map(|c| c.step).max().unwrap_or(1);
            let min_step = checkpoints.iter().map(|c| c.step).min().unwrap_or(0);
            let range = (max_step - min_step) as f32;
            return checkpoints
                .iter()
                .map(|c| {
                    if range < f32::EPSILON {
                        0.5
                    } else {
                        (c.step - min_step) as f32 / range
                    }
                })
                .collect();
        }

        let min_psnr = known.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_psnr = known.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_psnr - min_psnr;

        checkpoints
            .iter()
            .map(|c| {
                if c.psnr < 0.0 {
                    0.0
                } else if range < f32::EPSILON {
                    // All known PSNR values are identical
                    1.0
                } else {
                    (c.psnr - min_psnr) / range
                }
            })
            .collect::<Vec<f32>>()
            .pipe(|v| normalize_to_unit(&v))
    }

    /// Compute per-checkpoint stability scores (normalised to [0,1]).
    ///
    /// `extractor` returns the value of interest, or `NaN` when unknown.
    fn compute_stability_scores<F>(
        &self,
        checkpoints: &[CheckpointMetadata],
        extractor: F,
    ) -> Vec<f32>
    where
        F: Fn(&CheckpointMetadata) -> f32,
    {
        let n = checkpoints.len();
        let values: Vec<f32> = checkpoints.iter().map(extractor).collect();

        // Check whether all values are unknown
        let all_unknown = values.iter().all(|v| v.is_nan());
        if all_unknown {
            return vec![1.0; n];
        }

        // Compute windowed variance for each position
        let raw_stability: Vec<f32> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(3);
                let hi = (i + 3).min(n - 1);
                let window: Vec<f32> = values[lo..=hi]
                    .iter()
                    .cloned()
                    .filter(|v| !v.is_nan())
                    .collect();

                let var = window_variance(&window);
                1.0 / (1.0 + var)
            })
            .collect();

        normalize_to_unit(&raw_stability)
    }
}

// ---------------------------------------------------------------------------
// Helper: variance of a slice
// ---------------------------------------------------------------------------

fn window_variance(vals: &[f32]) -> f32 {
    let n = vals.len();
    if n <= 1 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f32>() / n as f32;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    var
}

// ---------------------------------------------------------------------------
// Helper: normalise a Vec<f32> to [0,1]
// ---------------------------------------------------------------------------

fn normalize_to_unit(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range < f32::EPSILON {
        // All values are equal — give everyone 1.0 (best possible)
        return vec![1.0; v.len()];
    }
    v.iter().map(|x| (x - min) / range).collect()
}

// ---------------------------------------------------------------------------
// Pipeline helper (avoid repeating .pipe calls)
// ---------------------------------------------------------------------------

trait PipeExt: Sized {
    fn pipe<F, B>(self, f: F) -> B
    where
        F: FnOnce(Self) -> B;
}

impl<T> PipeExt for T {
    fn pipe<F, B>(self, f: F) -> B
    where
        F: FnOnce(Self) -> B,
    {
        f(self)
    }
}

// ---------------------------------------------------------------------------
// ResumeRecommendation
// ---------------------------------------------------------------------------

/// The recommended checkpoint to resume from, with context.
#[derive(Debug, Clone)]
pub struct ResumeRecommendation {
    /// The highest-scoring checkpoint.
    pub best_checkpoint: CheckpointMetadata,
    /// Score of the best checkpoint.
    pub best_score: f32,
    /// Confidence in the recommendation, in [0.0, 1.0].
    ///
    /// Computed as `(best_score − second_best_score) / (best_score + ε)`,
    /// clamped to [0.0, 1.0].
    pub confidence: f32,
    /// Human-readable explanation of why this checkpoint was chosen.
    pub reason: String,
    /// Up to 3 runner-up checkpoints by score.
    pub alternatives: Vec<CheckpointMetadata>,
    /// Total number of valid checkpoints that were scanned.
    pub total_scanned: usize,
}

impl ResumeRecommendation {
    /// Render a multi-line human-readable report using Unicode box-drawing characters.
    pub fn format_report(&self) -> String {
        let mut out = String::new();
        let best = &self.best_checkpoint;

        // Header
        out.push_str("╭─────────────────────────────────────────────────────────╮\n");
        out.push_str("│          Checkpoint Resume Analysis Report               │\n");
        out.push_str("├─────────────────────────────────────────────────────────┤\n");

        // Best checkpoint
        out.push_str(&format!(
            "│  Best checkpoint  : step {:<8}  score={:.3}          │\n",
            best.step, self.best_score
        ));

        let psnr_str = if best.psnr >= 0.0 {
            format!("{:.1} dB", best.psnr)
        } else {
            "unknown".to_string()
        };
        out.push_str(&format!("│  PSNR             : {:<38}│\n", psnr_str));

        let loss_str = if best.loss >= 0.0 {
            format!("{:.4}", best.loss)
        } else {
            "unknown".to_string()
        };
        out.push_str(&format!("│  Loss             : {:<38}│\n", loss_str));

        out.push_str(&format!(
            "│  Gaussians        : {:<38}│\n",
            best.num_gaussians
        ));

        let size_mb = best.file_size_bytes as f64 / (1024.0 * 1024.0);
        out.push_str(&format!(
            "│  File size        : {:.1} MB{:<33}│\n",
            size_mb, ""
        ));

        out.push_str(&format!(
            "│  Confidence       : {:.1}%{:<36}│\n",
            self.confidence * 100.0,
            ""
        ));

        out.push_str(&format!(
            "│  Total scanned    : {:<38}│\n",
            self.total_scanned
        ));

        out.push_str("├─────────────────────────────────────────────────────────┤\n");
        // Truncate by character, not by byte offset: `reason` is a public
        // field that may hold caller-supplied or localized text, and a byte
        // slice can land inside a multi-byte UTF-8 sequence and panic.
        let reason_display: String = self.reason.chars().take(49).collect();
        out.push_str(&format!("│  Reason: {:<49}│\n", reason_display));

        // Alternatives
        if !self.alternatives.is_empty() {
            out.push_str("├─────────────────────────────────────────────────────────┤\n");
            out.push_str("│  Alternatives:                                          │\n");
            for alt in &self.alternatives {
                let alt_psnr = if alt.psnr >= 0.0 {
                    format!("psnr={:.1}", alt.psnr)
                } else {
                    "psnr=N/A".to_string()
                };
                out.push_str(&format!("│    step={:<6}  {:<40}│\n", alt.step, alt_psnr));
            }
        }

        out.push_str("╰─────────────────────────────────────────────────────────╯\n");
        out
    }
}

// ---------------------------------------------------------------------------
// Main analysis functions
// ---------------------------------------------------------------------------

/// Scan `dir`, score the discovered checkpoints, and return a recommendation.
///
/// # Errors
///
/// - [`ResumeError::DirectoryNotFound`] if the directory does not exist.
/// - [`ResumeError::NoCheckpointsFound`] if no parseable checkpoints are found.
/// - [`ResumeError::IoError`] for any underlying filesystem errors.
/// - [`ResumeError::EmptyScores`] (should not occur in practice).
pub fn analyze_checkpoints(
    dir: &Path,
    scanner: &CheckpointScanner,
    scorer: &CheckpointScorer,
) -> Result<ResumeRecommendation, ResumeError> {
    let checkpoints = scanner.scan(dir)?;
    let total_scanned = checkpoints.len();

    if total_scanned == 0 {
        return Err(ResumeError::NoCheckpointsFound);
    }

    // Special case: only one checkpoint
    if total_scanned == 1 {
        let best = checkpoints
            .into_iter()
            .next()
            .ok_or(ResumeError::EmptyScores)?;
        let reason = format!(
            "Only checkpoint available at step {} — resuming from it directly",
            best.step
        );
        return Ok(ResumeRecommendation {
            best_checkpoint: best,
            best_score: 1.0,
            confidence: 1.0,
            reason,
            alternatives: Vec::new(),
            total_scanned,
        });
    }

    // Score all checkpoints
    let scores = scorer.score(&checkpoints);
    if scores.is_empty() {
        return Err(ResumeError::EmptyScores);
    }

    // Pair checkpoints with their scores and sort descending
    let mut paired: Vec<(CheckpointMetadata, f32)> = checkpoints.into_iter().zip(scores).collect();
    paired.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let best_score = paired[0].1;
    let second_score = paired[1].1;
    let confidence = ((best_score - second_score) / (best_score + 1e-8)).clamp(0.0, 1.0);

    let best_checkpoint = paired[0].0.clone();

    let reason = build_reason(&best_checkpoint, best_score, confidence);

    let alternatives: Vec<CheckpointMetadata> =
        paired[1..].iter().take(3).map(|(c, _)| c.clone()).collect();

    Ok(ResumeRecommendation {
        best_checkpoint,
        best_score,
        confidence,
        reason,
        alternatives,
        total_scanned,
    })
}

/// Convenience wrapper using default scanner and scorer settings.
pub fn analyze_checkpoints_default(dir: &Path) -> Result<ResumeRecommendation, ResumeError> {
    let scanner = CheckpointScanner::new();
    let scorer = CheckpointScorer::new();
    analyze_checkpoints(dir, &scanner, &scorer)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_reason(best: &CheckpointMetadata, score: f32, confidence: f32) -> String {
    let psnr_part = if best.psnr >= 0.0 {
        format!("Best PSNR ({:.1} dB)", best.psnr)
    } else {
        "Highest composite score".to_string()
    };

    let stability_part = if confidence >= 0.7 {
        "with stable training"
    } else if confidence >= 0.4 {
        "with moderate stability"
    } else {
        "though scores are close"
    };

    format!(
        "{} at step {} {} (score={:.3})",
        psnr_part, best.step, stability_part, score
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    // ------------------------------------------------------------------
    // Filename parsing
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_step_from_filename_patterns() {
        // Pattern 1: "checkpoint_step_12345.json"
        assert_eq!(
            CheckpointMetadata::parse_step_from_filename("checkpoint_step_12345.json"),
            Some(12345),
        );

        // Pattern 2: "checkpoint_12345.bin"
        assert_eq!(
            CheckpointMetadata::parse_step_from_filename("checkpoint_12345.bin"),
            Some(12345),
        );

        // Pattern 3: "ckpt_12345"
        assert_eq!(
            CheckpointMetadata::parse_step_from_filename("ckpt_12345"),
            Some(12345),
        );

        // Pattern 4: "step_12345"
        assert_eq!(
            CheckpointMetadata::parse_step_from_filename("step_12345"),
            Some(12345),
        );

        // None case: no number
        assert_eq!(
            CheckpointMetadata::parse_step_from_filename("model.safetensors"),
            None,
        );
    }

    #[test]
    fn test_parse_step_edge_cases() {
        // epoch keyword
        assert_eq!(
            CheckpointMetadata::parse_step_from_filename("epoch_100.bin"),
            Some(100),
        );
        // Step zero is valid
        assert_eq!(
            CheckpointMetadata::parse_step_from_filename("step_0.json"),
            Some(0),
        );
    }

    // ------------------------------------------------------------------
    // Summary formatting
    // ------------------------------------------------------------------

    #[test]
    fn test_checkpoint_metadata_format_summary() {
        let meta = CheckpointMetadata {
            path: PathBuf::from("/tmp/ckpt_1000.json"),
            step: 1000,
            psnr: 28.3,
            loss: 0.012,
            num_gaussians: 50_000,
            timestamp_secs: 0,
            file_size_bytes: 12 * 1024 * 1024,
        };

        let line = meta.format_summary_line();
        assert!(line.contains("step=1000"), "must include step: {line}");
        assert!(line.contains("psnr=28.3"), "must include psnr: {line}");
        assert!(line.contains("loss=0.0120"), "must include loss: {line}");
        assert!(
            line.contains("gaussians=50000"),
            "must include gaussians: {line}"
        );
        assert!(line.contains("size="), "must include size: {line}");
    }

    #[test]
    fn test_checkpoint_metadata_format_summary_unknown() {
        let meta = CheckpointMetadata {
            path: PathBuf::from("/tmp/step_0.bin"),
            step: 0,
            psnr: -1.0,
            loss: -1.0,
            num_gaussians: 0,
            timestamp_secs: 0,
            file_size_bytes: 1024,
        };
        let line = meta.format_summary_line();
        assert!(
            line.contains("psnr=unknown"),
            "psnr should be unknown: {line}"
        );
        assert!(
            line.contains("loss=unknown"),
            "loss should be unknown: {line}"
        );
    }

    // ------------------------------------------------------------------
    // Scanner
    // ------------------------------------------------------------------

    #[test]
    fn test_scanner_default_extensions() {
        let scanner = CheckpointScanner::new();
        assert!(scanner.extensions.iter().any(|e| e == "json"));
        assert!(scanner.extensions.iter().any(|e| e == "bin"));
        assert!(scanner.extensions.iter().any(|e| e == "safetensors"));
        assert!(scanner.extensions.iter().any(|e| e == "ckpt"));
    }

    #[test]
    fn test_scanner_scan_empty_dir() {
        let dir = std::env::temp_dir().join("oxigaf_resume_test_empty");
        fs::create_dir_all(&dir).expect("create test dir");

        let scanner = CheckpointScanner::new();
        let result = scanner.scan(&dir).expect("scan should succeed");
        assert!(result.is_empty(), "empty dir should yield no checkpoints");

        let _ = fs::remove_dir_all(&dir);
    }

    fn create_checkpoint_file(dir: &Path, name: &str) {
        let path = dir.join(name);
        let mut f = fs::File::create(&path).expect("create file");
        f.write_all(b"{}").expect("write file");
    }

    #[test]
    fn test_scanner_scan_finds_checkpoint_files() {
        let dir = std::env::temp_dir().join("oxigaf_resume_test_finds");
        fs::create_dir_all(&dir).expect("create test dir");

        create_checkpoint_file(&dir, "checkpoint_step_100.json");
        create_checkpoint_file(&dir, "checkpoint_step_200.json");
        create_checkpoint_file(&dir, "not_a_checkpoint.txt"); // wrong extension
        create_checkpoint_file(&dir, "readme.md"); // wrong extension

        let scanner = CheckpointScanner::new();
        let found = scanner.scan(&dir).expect("scan should succeed");
        assert_eq!(
            found.len(),
            2,
            "should find exactly 2 checkpoint files: {:?}",
            found
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_scanner_scan_reads_real_checkpoint_json_header() {
        // Regression test: `scan()` must extract psnr/loss/num_gaussians
        // from an actual `oxigaf_trainer::checkpoint::CheckpointData` JSON
        // file instead of always leaving them at the "unknown" sentinels.
        let dir = std::env::temp_dir().join("oxigaf_resume_test_real_header");
        fs::create_dir_all(&dir).expect("create test dir");

        let payload = serde_json::json!({
            "version": 1,
            "iteration": 250,
            "positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "rotations": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            "scales": [[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]],
            "opacities": [0.5, 0.6],
            "sh_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "sh_degree": 0,
            "face_indices": [0, 1],
            "barycentric": [[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]],
            "local_offsets": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "is_rigid": [true, false],
            "optimizer_groups": [],
            "metrics_history": [
                {"iteration": 100, "psnr": 22.0, "ssim": 0.8, "loss": 0.2},
                {"iteration": 250, "psnr": 29.7, "ssim": 0.92, "loss": 0.031}
            ]
        });

        let path = dir.join("checkpoint_step_250.json");
        fs::write(&path, payload.to_string()).expect("write checkpoint");

        let scanner = CheckpointScanner::new();
        let found = scanner.scan(&dir).expect("scan should succeed");
        assert_eq!(found.len(), 1);
        let ckpt = &found[0];
        assert_eq!(ckpt.step, 250);
        assert_eq!(
            ckpt.num_gaussians, 2,
            "should read positions.len() from the checkpoint header"
        );
        assert!(
            (ckpt.psnr - 29.7).abs() < 1e-4,
            "should read the latest metrics_history psnr, got {}",
            ckpt.psnr
        );
        assert!(
            (ckpt.loss - 0.031).abs() < 1e-5,
            "should read the latest metrics_history loss, got {}",
            ckpt.loss
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_scanner_scan_falls_back_to_filename_psnr_for_non_json_checkpoints() {
        // Non-JSON checkpoint formats (.bin/.safetensors/.ckpt) have no
        // parseable header, so the scanner should fall back to a PSNR value
        // embedded in the filename rather than reporting -1.0 unnecessarily.
        let dir = std::env::temp_dir().join("oxigaf_resume_test_filename_psnr");
        fs::create_dir_all(&dir).expect("create test dir");

        create_checkpoint_file(&dir, "ckpt_1000_psnr_31.2.bin");

        let scanner = CheckpointScanner::new();
        let found = scanner.scan(&dir).expect("scan should succeed");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].step, 1000);
        assert!(
            (found[0].psnr - 31.2).abs() < 1e-4,
            "should fall back to filename-embedded psnr for non-JSON checkpoints, got {}",
            found[0].psnr
        );
        // loss/num_gaussians remain genuinely unknown for this format.
        assert_eq!(found[0].loss, -1.0);
        assert_eq!(found[0].num_gaussians, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_scanner_scan_sorted_by_step() {
        let dir = std::env::temp_dir().join("oxigaf_resume_test_sorted");
        fs::create_dir_all(&dir).expect("create test dir");

        // Create files in non-sorted order
        create_checkpoint_file(&dir, "checkpoint_step_300.bin");
        create_checkpoint_file(&dir, "checkpoint_step_100.bin");
        create_checkpoint_file(&dir, "checkpoint_step_200.bin");

        let scanner = CheckpointScanner::new();
        let found = scanner.scan(&dir).expect("scan should succeed");
        assert_eq!(found.len(), 3);
        assert_eq!(found[0].step, 100);
        assert_eq!(found[1].step, 200);
        assert_eq!(found[2].step, 300);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_scanner_step_range_filter() {
        let dir = std::env::temp_dir().join("oxigaf_resume_test_range");
        fs::create_dir_all(&dir).expect("create test dir");

        create_checkpoint_file(&dir, "step_50.json");
        create_checkpoint_file(&dir, "step_100.json");
        create_checkpoint_file(&dir, "step_200.json");
        create_checkpoint_file(&dir, "step_500.json");

        let scanner = CheckpointScanner::new().with_step_range(100, 300);
        let found = scanner.scan(&dir).expect("scan");
        let steps: Vec<usize> = found.iter().map(|c| c.step).collect();
        assert_eq!(
            steps,
            vec![100, 200],
            "should only include steps 100..=300: {steps:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // Scorer
    // ------------------------------------------------------------------

    #[test]
    fn test_scoring_single_checkpoint() {
        let meta = CheckpointMetadata {
            path: PathBuf::from("/tmp/step_1.json"),
            step: 1,
            psnr: 25.0,
            loss: 0.1,
            num_gaussians: 1000,
            timestamp_secs: 0,
            file_size_bytes: 100,
        };

        let scorer = CheckpointScorer::new();
        let scores = scorer.score(&[meta]);
        assert_eq!(scores.len(), 1);
        // Single checkpoint always gets 1.0
        assert!(
            (scores[0] - 1.0).abs() < 1e-6,
            "single checkpoint score should be 1.0"
        );
    }

    #[test]
    fn test_scoring_psnr_normalized() {
        let ckpts = vec![
            ckpt_with_psnr(100, 20.0),
            ckpt_with_psnr(200, 25.0),
            ckpt_with_psnr(300, 30.0),
        ];

        let scorer = CheckpointScorer::new();
        let scores = scorer.score(&ckpts);
        assert_eq!(scores.len(), 3);

        // The highest-PSNR checkpoint should score highest
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(max_idx, 2, "step=300 (psnr=30.0) should score highest");
    }

    #[test]
    fn test_scoring_all_unknown_psnr() {
        // When all PSNR are -1.0, falls back to step-based proxy
        let ckpts = vec![
            ckpt_with_psnr(100, -1.0),
            ckpt_with_psnr(200, -1.0),
            ckpt_with_psnr(300, -1.0),
        ];

        let scorer = CheckpointScorer::with_weights(ScoringWeights {
            psnr_weight: 1.0,
            loss_stability_weight: 0.0,
            gaussian_stability_weight: 0.0,
        });
        let scores = scorer.score(&ckpts);
        assert_eq!(scores.len(), 3);

        // Later steps should score higher when PSNR unknown
        assert!(
            scores[2] > scores[0],
            "step=300 should score higher than step=100 in step-proxy mode: {scores:?}"
        );
    }

    #[test]
    fn test_scoring_weights_sum_to_one() {
        let w = ScoringWeights::default();
        let sum = w.psnr_weight + w.loss_stability_weight + w.gaussian_stability_weight;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "default weights must sum to 1.0, got {sum}"
        );
    }

    // ------------------------------------------------------------------
    // Analyze functions
    // ------------------------------------------------------------------

    #[test]
    fn test_analyze_no_checkpoints_error() {
        let dir = std::env::temp_dir().join("oxigaf_resume_test_nofiles");
        fs::create_dir_all(&dir).expect("create test dir");

        let result = analyze_checkpoints_default(&dir);
        assert!(
            matches!(result, Err(ResumeError::NoCheckpointsFound)),
            "expected NoCheckpointsFound, got {:?}",
            result
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_analyze_single_checkpoint_confidence() {
        let dir = std::env::temp_dir().join("oxigaf_resume_test_single");
        fs::create_dir_all(&dir).expect("create test dir");
        create_checkpoint_file(&dir, "checkpoint_step_100.json");

        let rec = analyze_checkpoints_default(&dir).expect("analyze should succeed");
        assert_eq!(rec.best_checkpoint.step, 100);
        assert!(
            (rec.confidence - 1.0).abs() < 1e-6,
            "single-file confidence should be 1.0"
        );
        assert_eq!(rec.total_scanned, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_analyze_multiple_checkpoints() {
        let dir = std::env::temp_dir().join("oxigaf_resume_test_multi");
        fs::create_dir_all(&dir).expect("create test dir");

        create_checkpoint_file(&dir, "step_100.json");
        create_checkpoint_file(&dir, "step_200.json");
        create_checkpoint_file(&dir, "step_300.json");

        let rec = analyze_checkpoints_default(&dir).expect("analyze should succeed");
        assert_eq!(rec.total_scanned, 3);
        assert!(!rec.reason.is_empty());
        // Alternatives ≤ 3
        assert!(rec.alternatives.len() <= 3);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_recommendation_format_report() {
        let best = CheckpointMetadata {
            path: PathBuf::from("/tmp/step_5000.json"),
            step: 5000,
            psnr: 28.3,
            loss: 0.012,
            num_gaussians: 50_000,
            timestamp_secs: 0,
            file_size_bytes: 12 * 1024 * 1024,
        };

        let rec = ResumeRecommendation {
            best_checkpoint: best,
            best_score: 0.85,
            confidence: 0.72,
            reason: "Best PSNR (28.3 dB) at step 5000 with stable training".to_string(),
            alternatives: Vec::new(),
            total_scanned: 5,
        };

        let report = rec.format_report();
        assert!(report.contains("5000"), "should mention step 5000");
        assert!(report.contains("28.3"), "should mention PSNR");
        assert!(report.contains('╭'), "should use box-drawing chars");
        assert!(report.contains('╰'), "should close box");
    }

    #[test]
    fn test_recommendation_format_report_multibyte_reason_does_not_panic() {
        // Regression test: `reason` is a public field, so a caller-supplied
        // or localized reason whose 49th *byte* falls inside a multi-byte
        // UTF-8 character must not panic when truncated for display.
        let best = CheckpointMetadata {
            path: PathBuf::from("/tmp/step_5000.json"),
            step: 5000,
            psnr: 28.3,
            loss: 0.012,
            num_gaussians: 50_000,
            timestamp_secs: 0,
            file_size_bytes: 12 * 1024 * 1024,
        };

        let rec = ResumeRecommendation {
            best_checkpoint: best,
            best_score: 0.85,
            confidence: 0.72,
            reason: "損失関数が最も低いチェックポイントであり訓練が安定しているため推奨されます"
                .to_string(),
            alternatives: Vec::new(),
            total_scanned: 5,
        };

        // Must not panic.
        let report = rec.format_report();
        assert!(report.contains("損失関数"), "reason text should appear");
    }

    #[test]
    fn test_confidence_range() {
        let dir = std::env::temp_dir().join("oxigaf_resume_test_confidence");
        fs::create_dir_all(&dir).expect("create test dir");

        // Create many checkpoints with varying scores
        for step in [50, 100, 200, 500, 1000, 2000, 5000] {
            create_checkpoint_file(&dir, &format!("step_{step}.json"));
        }

        let rec = analyze_checkpoints_default(&dir).expect("analyze");
        assert!(
            rec.confidence >= 0.0 && rec.confidence <= 1.0,
            "confidence must be in [0,1], got {}",
            rec.confidence
        );

        let _ = fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // Helpers for tests
    // ------------------------------------------------------------------

    fn ckpt_with_psnr(step: usize, psnr: f32) -> CheckpointMetadata {
        CheckpointMetadata {
            path: PathBuf::from(format!("/tmp/step_{step}.json")),
            step,
            psnr,
            loss: -1.0,
            num_gaussians: 0,
            timestamp_secs: 0,
            file_size_bytes: 1024,
        }
    }
}
