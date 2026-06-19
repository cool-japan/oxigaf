//! Post-hoc model calibration for the OxiGAF training pipeline.
//!
//! Implements three calibration methods:
//! - **Temperature Scaling** (Guo et al. 2017): single-parameter calibration via logit rescaling.
//! - **Platt Scaling**: affine calibration with slope + bias, fitted by gradient descent.
//! - **Isotonic Regression** (PAV algorithm): non-parametric monotone calibration.
//!
//! Also provides ECE / MCE / overconfidence metrics, reliability diagrams, and summary stats.

use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by temperature-scaling / calibration routines.
#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("Empty input: need at least 1 sample")]
    EmptyInput,

    #[error("Length mismatch: logits has {logits}, labels has {labels}")]
    LengthMismatch { logits: usize, labels: usize },

    #[error("Invalid temperature: {t} (must be > 0)")]
    InvalidTemperature { t: f32 },

    #[error("Optimization failed to converge after {iters} iterations")]
    DidNotConverge { iters: usize },

    #[error("Invalid label {label}: must be in [0, 1]")]
    InvalidLabel { label: f32 },
}

// ---------------------------------------------------------------------------
// Config / Result / Stats
// ---------------------------------------------------------------------------

/// Hyperparameters for calibration optimisers.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of bins used in ECE / reliability-diagram computation (default 10).
    pub n_bins: usize,
    /// Maximum number of optimisation iterations (default 1000).
    pub max_iters: usize,
    /// Convergence tolerance for golden-section / gradient descent (default 1e-6).
    pub tolerance: f32,
    /// Learning rate for gradient-based methods such as Platt scaling (default 0.01).
    pub learning_rate: f32,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            n_bins: 10,
            max_iters: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
        }
    }
}

/// Summary of before/after calibration quality.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// ECE before calibration.
    pub pre_ece: f32,
    /// ECE after calibration.
    pub post_ece: f32,
    /// MCE before calibration.
    pub pre_mce: f32,
    /// MCE after calibration.
    pub post_mce: f32,
    /// Name of the calibration method used: `"temperature"`, `"platt"`, or `"isotonic"`.
    pub method: String,
    /// Number of calibration samples.
    pub n_samples: usize,
    /// Actual number of optimisation iterations executed.
    pub iterations_used: usize,
}

/// Aggregate statistics for a set of confidence predictions.
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Mean confidence over all samples.
    pub mean_confidence: f32,
    /// Mean label (i.e. empirical accuracy / positive rate) over all samples.
    pub mean_accuracy: f32,
    /// Standard deviation of confidence values.
    pub confidence_std: f32,
    /// Brier score: MSE between predicted probabilities and binary labels.
    pub brier_score: f32,
    /// Log-loss (binary cross-entropy) with numerical clamping.
    pub log_loss: f32,
}

// ---------------------------------------------------------------------------
// Primitive helpers
// ---------------------------------------------------------------------------

/// Numerically stable sigmoid for a single value.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Validate that `logits` and `labels` have the same non-zero length, and that
/// every label is in \[0, 1\].
fn ts_validate_inputs(logits: &[f32], labels: &[f32]) -> Result<(), CalibrationError> {
    if logits.is_empty() {
        return Err(CalibrationError::EmptyInput);
    }
    if logits.len() != labels.len() {
        return Err(CalibrationError::LengthMismatch {
            logits: logits.len(),
            labels: labels.len(),
        });
    }
    for &l in labels {
        if !(0.0..=1.0).contains(&l) {
            return Err(CalibrationError::InvalidLabel { label: l });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Binary NLL (objective for temperature scaling)
// ---------------------------------------------------------------------------

/// Binary cross-entropy (negative log-likelihood) at a given temperature.
///
/// `nll = -mean( y * log(σ(l/T)) + (1-y) * log(1 - σ(l/T)) )`
///
/// Numerically stable via log-sigmoid identity:
/// `log σ(z) = -log(1 + exp(-z))`; for negative z use `z - log(1+exp(z))`.
pub fn ts_binary_nll(logits: &[f32], labels: &[f32], temperature: f32) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let t = temperature.max(f32::EPSILON);
    let n = logits.len() as f64;
    let mut acc = 0.0_f64;
    for (&logit, &label) in logits.iter().zip(labels.iter()) {
        let z = (logit as f64) / (t as f64);
        // log σ(z) = -softplus(-z) = -(log(1+exp(-z))) — numerically stable form
        let log_sigma = if z >= 0.0 {
            -(-z).exp().ln_1p()
        } else {
            z - z.exp().ln_1p()
        };
        let log_1m_sigma = if z >= 0.0 {
            -z - (-z).exp().ln_1p()
        } else {
            -(z.exp().ln_1p())
        };
        let y = label as f64;
        acc += y * log_sigma + (1.0 - y) * log_1m_sigma;
    }
    (-(acc / n)) as f32
}

// ---------------------------------------------------------------------------
// Golden-section search
// ---------------------------------------------------------------------------

/// Minimise a unimodal function `f` on `[lo, hi]` using golden-section search.
///
/// Uses the golden ratio φ = (3 − √5) / 2 ≈ 0.38197.
/// Terminates when the bracket width falls below `tol` or `max_iters` is reached.
pub fn ts_golden_section_search(
    f: impl Fn(f32) -> f32,
    lo: f32,
    hi: f32,
    tol: f32,
    max_iters: usize,
) -> f32 {
    const PHI: f32 = 0.381_966_01; // (3 - sqrt(5)) / 2

    let mut a = lo;
    let mut b = hi;
    let mut x1 = a + PHI * (b - a);
    let mut x2 = b - PHI * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for _ in 0..max_iters {
        if (b - a).abs() < tol {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + PHI * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - PHI * (b - a);
            f2 = f(x2);
        }
    }
    (a + b) * 0.5
}

// ---------------------------------------------------------------------------
// Temperature Scaler
// ---------------------------------------------------------------------------

/// Post-hoc calibration by a single temperature parameter T > 0.
///
/// Binary form: `p = σ(logit / T)`.
///
/// T is learned by minimising binary cross-entropy on held-out calibration data
/// via golden-section search (exact for convex objectives).
#[derive(Debug, Clone)]
pub struct TemperatureScaler {
    /// Learned temperature value; must be > 0.
    pub temperature: f32,
    /// Whether `fit_binary` has been called successfully.
    pub fitted: bool,
}

impl TemperatureScaler {
    /// Create a new scaler with the given initial temperature.
    ///
    /// Returns [`CalibrationError::InvalidTemperature`] if `initial_temperature <= 0`.
    pub fn new(initial_temperature: f32) -> Result<Self, CalibrationError> {
        if initial_temperature <= 0.0 {
            return Err(CalibrationError::InvalidTemperature {
                t: initial_temperature,
            });
        }
        Ok(Self {
            temperature: initial_temperature,
            fitted: false,
        })
    }

    /// Fit the temperature on binary calibration data.
    ///
    /// Performs golden-section search over T ∈ \[1e-3, 10.0\] minimising binary NLL.
    pub fn fit_binary(
        &mut self,
        logits: &[f32],
        labels: &[f32],
        config: &CalibrationConfig,
    ) -> Result<(), CalibrationError> {
        ts_validate_inputs(logits, labels)?;

        let logits_owned = logits.to_vec();
        let labels_owned = labels.to_vec();

        let t_opt = ts_golden_section_search(
            |t| ts_binary_nll(&logits_owned, &labels_owned, t),
            1e-3,
            10.0,
            config.tolerance,
            config.max_iters,
        );

        if !t_opt.is_finite() || t_opt <= 0.0 {
            return Err(CalibrationError::InvalidTemperature { t: t_opt });
        }
        self.temperature = t_opt;
        self.fitted = true;
        Ok(())
    }

    /// Apply temperature scaling to a single logit: `σ(logit / T)`.
    #[inline]
    pub fn scale(&self, logit: f32) -> f32 {
        sigmoid(logit / self.temperature)
    }

    /// Apply temperature scaling to a batch of logits.
    pub fn scale_batch(&self, logits: &[f32]) -> Vec<f32> {
        logits.iter().map(|&l| self.scale(l)).collect()
    }
}

// ---------------------------------------------------------------------------
// Platt Scaler
// ---------------------------------------------------------------------------

/// Affine calibration: `p = σ(a · logit + b)`.
///
/// Parameters `a` and `b` are fitted via gradient descent on binary NLL.
#[derive(Debug, Clone)]
pub struct PlattScaler {
    /// Slope parameter (initialised to 1.0).
    pub a: f32,
    /// Bias parameter (initialised to 0.0).
    pub b: f32,
    /// Whether `fit` has been called successfully.
    pub fitted: bool,
}

impl PlattScaler {
    /// Create a new Platt scaler with identity initialisation (a=1, b=0).
    pub fn new() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            fitted: false,
        }
    }

    /// Fit slope and bias by gradient descent on binary cross-entropy.
    ///
    /// Uses up to `config.max_iters` gradient steps with `config.learning_rate`.
    pub fn fit(
        &mut self,
        logits: &[f32],
        labels: &[f32],
        config: &CalibrationConfig,
    ) -> Result<(), CalibrationError> {
        ts_validate_inputs(logits, labels)?;

        let n = logits.len() as f32;
        let lr = config.learning_rate;

        for _ in 0..config.max_iters {
            let mut grad_a = 0.0_f32;
            let mut grad_b = 0.0_f32;

            for (&logit, &label) in logits.iter().zip(labels.iter()) {
                // p = σ(a*logit + b); error = p - y
                let z = self.a * logit + self.b;
                let p = sigmoid(z);
                let err = p - label;
                grad_a += err * logit;
                grad_b += err;
            }

            grad_a /= n;
            grad_b /= n;

            self.a -= lr * grad_a;
            self.b -= lr * grad_b;
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict calibrated probability for a single logit: `σ(a·logit + b)`.
    #[inline]
    pub fn predict(&self, logit: f32) -> f32 {
        sigmoid(self.a * logit + self.b)
    }

    /// Predict calibrated probabilities for a batch of logits.
    pub fn predict_batch(&self, logits: &[f32]) -> Vec<f32> {
        logits.iter().map(|&l| self.predict(l)).collect()
    }
}

impl Default for PlattScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PAV (Pool-Adjacent Violators) — core isotonic regression
// ---------------------------------------------------------------------------

/// Core pool-adjacent-violators algorithm for isotonic regression.
///
/// Given a sequence of `values` and associated `weights`, returns a
/// monotonically non-decreasing sequence of the same length computed by
/// iteratively merging adjacent blocks that violate monotonicity.
///
/// Each block's value is its weighted average.
pub fn ts_pav_isotonic(values: &[f32], weights: &[f32]) -> Vec<f32> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }

    // Each block: (weighted_sum, total_weight, block_length)
    struct Block {
        weighted_sum: f64,
        total_weight: f64,
        len: usize,
    }

    let mut blocks: Vec<Block> = Vec::with_capacity(n);

    for i in 0..n {
        let w = weights[i] as f64;
        let v = values[i] as f64;
        blocks.push(Block {
            weighted_sum: w * v,
            total_weight: w,
            len: 1,
        });

        // Pool with preceding block while violating monotonicity
        while blocks.len() >= 2 {
            let last = blocks.len() - 1;
            let prev_avg = blocks[last - 1].weighted_sum / blocks[last - 1].total_weight;
            let cur_avg = blocks[last].weighted_sum / blocks[last].total_weight;
            if cur_avg < prev_avg {
                // Merge
                let merged_ws = blocks[last - 1].weighted_sum + blocks[last].weighted_sum;
                let merged_tw = blocks[last - 1].total_weight + blocks[last].total_weight;
                let merged_len = blocks[last - 1].len + blocks[last].len;
                blocks.pop();
                if let Some(prev) = blocks.last_mut() {
                    prev.weighted_sum = merged_ws;
                    prev.total_weight = merged_tw;
                    prev.len = merged_len;
                }
            } else {
                break;
            }
        }
    }

    // Expand blocks back to per-sample values
    let mut result = Vec::with_capacity(n);
    for block in &blocks {
        let avg = (block.weighted_sum / block.total_weight) as f32;
        for _ in 0..block.len {
            result.push(avg);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Isotonic Calibrator
// ---------------------------------------------------------------------------

/// Non-parametric monotone calibration using pool-adjacent violators (PAV).
///
/// After fitting, the calibrator maps raw scores to calibrated probabilities
/// via piecewise-constant interpolation along sorted breakpoints.
#[derive(Debug, Clone)]
pub struct IsotonicCalibrator {
    /// Sorted input score breakpoints (one per training sample after PAV).
    pub thresholds: Vec<f32>,
    /// Calibrated output values at each breakpoint.
    pub values: Vec<f32>,
    /// Whether `fit` has been called successfully.
    pub fitted: bool,
}

impl IsotonicCalibrator {
    /// Create a new, unfitted isotonic calibrator.
    pub fn new() -> Self {
        Self {
            thresholds: Vec::new(),
            values: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the calibrator using the PAV algorithm.
    ///
    /// Sorts samples by score, runs PAV, then deduplicates to obtain breakpoints.
    pub fn fit(&mut self, scores: &[f32], labels: &[f32]) -> Result<(), CalibrationError> {
        ts_validate_inputs(scores, labels)?;

        let n = scores.len();
        // Sort by score
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_scores: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
        let sorted_labels: Vec<f32> = idx.iter().map(|&i| labels[i]).collect();
        let weights = vec![1.0_f32; n];

        let pav_values = ts_pav_isotonic(&sorted_labels, &weights);

        self.thresholds = sorted_scores;
        self.values = pav_values;
        self.fitted = true;
        Ok(())
    }

    /// Predict calibrated probability for a single score using nearest breakpoint.
    pub fn predict(&self, score: f32) -> f32 {
        if self.thresholds.is_empty() {
            return 0.5;
        }
        // Binary search for the nearest breakpoint
        match self
            .thresholds
            .binary_search_by(|t| t.partial_cmp(&score).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => self.values[idx],
            Err(0) => self.values[0],
            Err(idx) if idx >= self.thresholds.len() => *self.values.last().unwrap_or(&0.5),
            Err(idx) => {
                // Find nearest of idx-1 and idx
                let lo = self.thresholds[idx - 1];
                let hi = self.thresholds[idx];
                if (score - lo).abs() <= (score - hi).abs() {
                    self.values[idx - 1]
                } else {
                    self.values[idx]
                }
            }
        }
    }

    /// Predict calibrated probabilities for a batch of scores.
    pub fn predict_batch(&self, scores: &[f32]) -> Vec<f32> {
        scores.iter().map(|&s| self.predict(s)).collect()
    }
}

impl Default for IsotonicCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ECE / MCE / Overconfidence error
// ---------------------------------------------------------------------------

/// Binned stat result for [`ts_bin_stats`]: `(bin_sum_conf, bin_sum_acc, bin_count)`.
type BinStats = (Vec<f64>, Vec<f64>, Vec<usize>);

/// Compute binned confidence/accuracy pairs for ECE-style metrics.
///
/// Returns `(bin_sum_conf, bin_sum_acc, bin_count)` per bin.
fn ts_bin_stats(
    confidences: &[f32],
    labels: &[f32],
    n_bins: usize,
) -> Result<BinStats, CalibrationError> {
    ts_validate_inputs(confidences, labels)?;
    if n_bins == 0 {
        return Err(CalibrationError::EmptyInput);
    }

    let mut bin_sum_conf = vec![0.0_f64; n_bins];
    let mut bin_sum_acc = vec![0.0_f64; n_bins];
    let mut bin_count = vec![0_usize; n_bins];

    for (&conf, &label) in confidences.iter().zip(labels.iter()) {
        let conf_c = conf.clamp(0.0, 1.0);
        let bin = ((conf_c * n_bins as f32) as usize).min(n_bins - 1);
        bin_sum_conf[bin] += conf_c as f64;
        bin_sum_acc[bin] += label as f64;
        bin_count[bin] += 1;
    }
    Ok((bin_sum_conf, bin_sum_acc, bin_count))
}

/// Expected Calibration Error (ECE).
///
/// `ECE = Σ_b (|acc_b − conf_b| × n_b / N)`
pub fn ts_ece(confidences: &[f32], labels: &[f32], n_bins: usize) -> Result<f32, CalibrationError> {
    let n = confidences.len() as f64;
    let (sum_conf, sum_acc, count) = ts_bin_stats(confidences, labels, n_bins)?;
    let mut ece = 0.0_f64;
    for b in 0..n_bins {
        if count[b] == 0 {
            continue;
        }
        let avg_conf = sum_conf[b] / count[b] as f64;
        let avg_acc = sum_acc[b] / count[b] as f64;
        ece += (avg_acc - avg_conf).abs() * (count[b] as f64 / n);
    }
    Ok(ece as f32)
}

/// Maximum Calibration Error (MCE): maximum per-bin absolute gap.
pub fn ts_mce(confidences: &[f32], labels: &[f32], n_bins: usize) -> Result<f32, CalibrationError> {
    let (sum_conf, sum_acc, count) = ts_bin_stats(confidences, labels, n_bins)?;
    let mut mce = 0.0_f64;
    for b in 0..n_bins {
        if count[b] == 0 {
            continue;
        }
        let avg_conf = sum_conf[b] / count[b] as f64;
        let avg_acc = sum_acc[b] / count[b] as f64;
        let gap = (avg_acc - avg_conf).abs();
        if gap > mce {
            mce = gap;
        }
    }
    Ok(mce as f32)
}

/// Overconfidence error: only penalises bins where `conf_b > acc_b`.
pub fn ts_overconfidence_error(
    confidences: &[f32],
    labels: &[f32],
    n_bins: usize,
) -> Result<f32, CalibrationError> {
    let n = confidences.len() as f64;
    let (sum_conf, sum_acc, count) = ts_bin_stats(confidences, labels, n_bins)?;
    let mut oe = 0.0_f64;
    for b in 0..n_bins {
        if count[b] == 0 {
            continue;
        }
        let avg_conf = sum_conf[b] / count[b] as f64;
        let avg_acc = sum_acc[b] / count[b] as f64;
        if avg_conf > avg_acc {
            oe += (avg_conf - avg_acc) * (count[b] as f64 / n);
        }
    }
    Ok(oe as f32)
}

// ---------------------------------------------------------------------------
// Reliability diagram
// ---------------------------------------------------------------------------

/// Reliability diagram data: per-bin confidence, accuracy, and sample counts.
#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    /// Bin edges: `n_bins + 1` values uniformly spaced in `[0, 1]`.
    pub bin_edges: Vec<f32>,
    /// Mean predicted confidence per bin.
    pub mean_confidence: Vec<f32>,
    /// Mean empirical accuracy per bin.
    pub mean_accuracy: Vec<f32>,
    /// Number of samples per bin.
    pub bin_counts: Vec<usize>,
    /// Expected Calibration Error.
    pub ece: f32,
    /// Maximum Calibration Error.
    pub mce: f32,
}

/// Compute a [`ReliabilityDiagram`] from confidence predictions and binary labels.
pub fn ts_reliability_diagram(
    confidences: &[f32],
    labels: &[f32],
    n_bins: usize,
) -> Result<ReliabilityDiagram, CalibrationError> {
    ts_validate_inputs(confidences, labels)?;
    if n_bins == 0 {
        return Err(CalibrationError::EmptyInput);
    }

    let (sum_conf, sum_acc, count) = ts_bin_stats(confidences, labels, n_bins)?;

    let step = 1.0_f32 / n_bins as f32;
    let bin_edges: Vec<f32> = (0..=n_bins).map(|i| i as f32 * step).collect();

    let mut mean_confidence = Vec::with_capacity(n_bins);
    let mut mean_accuracy = Vec::with_capacity(n_bins);

    for b in 0..n_bins {
        if count[b] == 0 {
            let mid = (b as f32 + 0.5) * step;
            mean_confidence.push(mid);
            mean_accuracy.push(0.0);
        } else {
            mean_confidence.push((sum_conf[b] / count[b] as f64) as f32);
            mean_accuracy.push((sum_acc[b] / count[b] as f64) as f32);
        }
    }

    let ece = ts_ece(confidences, labels, n_bins)?;
    let mce = ts_mce(confidences, labels, n_bins)?;

    Ok(ReliabilityDiagram {
        bin_edges,
        mean_confidence,
        mean_accuracy,
        bin_counts: count,
        ece,
        mce,
    })
}

/// Render an ASCII reliability diagram.
///
/// Rows represent bins from low to high confidence; columns show bar length
/// proportional to calibration gap.
pub fn ts_format_reliability_diagram(diag: &ReliabilityDiagram) -> String {
    let n_bins = diag.mean_confidence.len();
    let mut out = String::new();
    out.push_str("Reliability Diagram\n");
    out.push_str("Bin  | Conf  | Acc   | Count | Gap  | Bar\n");
    out.push_str("-----|-------|-------|-------|------|-----\n");

    for b in 0..n_bins {
        let conf = diag.mean_confidence[b];
        let acc = diag.mean_accuracy[b];
        let gap = (conf - acc).abs();
        let bar_len = (gap * 40.0).round() as usize;
        let bar: String = "#".repeat(bar_len);
        out.push_str(&format!(
            "{:>4} | {:.3} | {:.3} | {:>5} | {:.3} | {}\n",
            b, conf, acc, diag.bin_counts[b], gap, bar
        ));
    }
    out.push_str(&format!("ECE={:.5}  MCE={:.5}\n", diag.ece, diag.mce));
    out
}

// ---------------------------------------------------------------------------
// Brier score / log-loss / stats
// ---------------------------------------------------------------------------

/// Brier score: mean squared error between predicted probabilities and labels.
///
/// Perfect predictions → 0; worst-case binary → 1.
pub fn ts_brier_score(confidences: &[f32], labels: &[f32]) -> Result<f32, CalibrationError> {
    ts_validate_inputs(confidences, labels)?;
    let n = confidences.len() as f64;
    let mut sum = 0.0_f64;
    for (&p, &y) in confidences.iter().zip(labels.iter()) {
        let diff = (p - y) as f64;
        sum += diff * diff;
    }
    Ok((sum / n) as f32)
}

/// Binary log-loss (cross-entropy) with numerical clamping via `eps`.
///
/// `L = -mean( y*log(p+ε) + (1-y)*log(1-p+ε) )`
pub fn ts_log_loss(confidences: &[f32], labels: &[f32], eps: f32) -> Result<f32, CalibrationError> {
    ts_validate_inputs(confidences, labels)?;
    let n = confidences.len() as f64;
    let e = eps as f64;
    let mut sum = 0.0_f64;
    for (&p, &y) in confidences.iter().zip(labels.iter()) {
        let p_c = (p as f64).clamp(e, 1.0 - e);
        let y_d = y as f64;
        sum += y_d * p_c.ln() + (1.0 - y_d) * (1.0 - p_c).ln();
    }
    Ok((-sum / n) as f32)
}

/// Compute aggregate [`CalibrationStats`] from confidence predictions and labels.
pub fn ts_compute_stats(
    confidences: &[f32],
    labels: &[f32],
) -> Result<CalibrationStats, CalibrationError> {
    ts_validate_inputs(confidences, labels)?;

    let n = confidences.len() as f64;

    // Mean confidence
    let mean_c: f64 = confidences.iter().map(|&c| c as f64).sum::<f64>() / n;

    // Mean accuracy (mean label)
    let mean_a: f64 = labels.iter().map(|&l| l as f64).sum::<f64>() / n;

    // Confidence std
    let var_c: f64 = confidences
        .iter()
        .map(|&c| {
            let d = c as f64 - mean_c;
            d * d
        })
        .sum::<f64>()
        / n;
    let std_c = var_c.sqrt() as f32;

    let brier = ts_brier_score(confidences, labels)?;
    let log = ts_log_loss(confidences, labels, 1e-7)?;

    Ok(CalibrationStats {
        mean_confidence: mean_c as f32,
        mean_accuracy: mean_a as f32,
        confidence_std: std_c,
        brier_score: brier,
        log_loss: log,
    })
}

/// Format calibration statistics as a human-readable string.
pub fn ts_format_stats(stats: &CalibrationStats) -> String {
    format!(
        "CalibrationStats {{ mean_conf={:.4}, mean_acc={:.4}, conf_std={:.4}, \
         brier={:.6}, log_loss={:.6} }}",
        stats.mean_confidence,
        stats.mean_accuracy,
        stats.confidence_std,
        stats.brier_score,
        stats.log_loss,
    )
}

/// Format a calibration result as a human-readable string.
pub fn ts_format_result(result: &CalibrationResult) -> String {
    format!(
        "CalibrationResult {{ method={}, n={}, iters={}, \
         pre_ece={:.5}, post_ece={:.5}, pre_mce={:.5}, post_mce={:.5} }}",
        result.method,
        result.n_samples,
        result.iterations_used,
        result.pre_ece,
        result.post_ece,
        result.pre_mce,
        result.post_mce,
    )
}

// ---------------------------------------------------------------------------
// fmt::Display implementations
// ---------------------------------------------------------------------------

impl fmt::Display for CalibrationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", ts_format_stats(self))
    }
}

impl fmt::Display for CalibrationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", ts_format_result(self))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Simple xorshift64 PRNG (per project policy — no `rand` crate).
    fn xorshift64(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        if *state == 0 {
            *state = 1;
        }
        *state
    }

    fn xorshift_f32(state: &mut u64) -> f32 {
        (xorshift64(state) as f32) / (u64::MAX as f32)
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── CalibrationError ──────────────────────────────────────────────────

    #[test]
    fn test_error_empty_input_display() {
        let e = CalibrationError::EmptyInput;
        assert!(e.to_string().contains("Empty"));
    }

    #[test]
    fn test_error_length_mismatch_display() {
        let e = CalibrationError::LengthMismatch {
            logits: 3,
            labels: 5,
        };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("5"));
    }

    #[test]
    fn test_error_invalid_temperature_display() {
        let e = CalibrationError::InvalidTemperature { t: -1.0 };
        assert!(e.to_string().contains("-1"));
    }

    #[test]
    fn test_error_did_not_converge_display() {
        let e = CalibrationError::DidNotConverge { iters: 42 };
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_error_invalid_label_display() {
        let e = CalibrationError::InvalidLabel { label: 1.5 };
        assert!(e.to_string().contains("1.5"));
    }

    // ── validate_inputs ───────────────────────────────────────────────────

    #[test]
    fn test_empty_input_error() {
        assert!(matches!(
            ts_binary_nll(&[], &[], 1.0),
            _ // returns 0.0, not an error — validate the ECE path
        ));
        assert!(matches!(
            ts_ece(&[], &[], 10),
            Err(CalibrationError::EmptyInput)
        ));
    }

    #[test]
    fn test_length_mismatch_error() {
        let logits = vec![1.0_f32, 2.0];
        let labels = vec![0.0_f32];
        assert!(matches!(
            ts_ece(&logits, &labels, 10),
            Err(CalibrationError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn test_invalid_label_error() {
        let logits = vec![1.0_f32];
        let labels = vec![1.5_f32];
        assert!(matches!(
            ts_ece(&logits, &labels, 10),
            Err(CalibrationError::InvalidLabel { .. })
        ));
    }

    // ── TemperatureScaler::new ────────────────────────────────────────────

    #[test]
    fn test_temperature_scaler_new_valid() {
        let ts = TemperatureScaler::new(1.0).expect("should succeed");
        assert!(approx(ts.temperature, 1.0, 1e-9));
        assert!(!ts.fitted);
    }

    #[test]
    fn test_temperature_scaler_new_zero_error() {
        assert!(matches!(
            TemperatureScaler::new(0.0),
            Err(CalibrationError::InvalidTemperature { .. })
        ));
    }

    #[test]
    fn test_temperature_scaler_new_negative_error() {
        assert!(matches!(
            TemperatureScaler::new(-1.0),
            Err(CalibrationError::InvalidTemperature { .. })
        ));
    }

    // ── ts_binary_nll ─────────────────────────────────────────────────────

    #[test]
    fn test_binary_nll_perfect_predictions_near_zero() {
        // logit = 10 → σ(10) ≈ 1; label = 1 → NLL ≈ 0
        let logits = vec![10.0_f32; 20];
        let labels = vec![1.0_f32; 20];
        let nll = ts_binary_nll(&logits, &labels, 1.0);
        assert!(nll < 1e-4, "NLL={nll} should be near 0");
    }

    #[test]
    fn test_binary_nll_negative_logits_zero_label() {
        // logit = -10 → σ(-10) ≈ 0; label = 0 → NLL ≈ 0
        let logits = vec![-10.0_f32; 20];
        let labels = vec![0.0_f32; 20];
        let nll = ts_binary_nll(&logits, &labels, 1.0);
        assert!(nll < 1e-4, "NLL={nll} should be near 0");
    }

    #[test]
    fn test_binary_nll_random_predictions_finite() {
        let mut state = 0xDEAD_BEEF_u64;
        let logits: Vec<f32> = (0..50)
            .map(|_| xorshift_f32(&mut state) * 4.0 - 2.0)
            .collect();
        let labels: Vec<f32> = (0..50)
            .map(|_| {
                if xorshift_f32(&mut state) > 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let nll = ts_binary_nll(&logits, &labels, 1.0);
        assert!(nll.is_finite(), "NLL must be finite, got {nll}");
        assert!(nll > 0.0, "NLL must be positive, got {nll}");
    }

    #[test]
    fn test_binary_nll_empty_returns_zero() {
        let nll = ts_binary_nll(&[], &[], 1.0);
        assert!(approx(nll, 0.0, 1e-9));
    }

    #[test]
    fn test_binary_nll_higher_temp_smooths_nll() {
        // At T >> 1, predictions approach 0.5 → higher NLL for certain data
        let logits = vec![5.0_f32; 10];
        let labels = vec![1.0_f32; 10];
        let nll_t1 = ts_binary_nll(&logits, &labels, 1.0);
        let nll_t100 = ts_binary_nll(&logits, &labels, 100.0);
        assert!(
            nll_t100 > nll_t1,
            "Higher T should increase NLL for confident correct predictions"
        );
    }

    // ── ts_golden_section_search ──────────────────────────────────────────

    #[test]
    fn test_golden_section_x_squared() {
        // Minimum of x² on [-1, 1] is at 0.
        let min = ts_golden_section_search(|x| x * x, -1.0, 1.0, 1e-8, 200);
        assert!(min.abs() < 1e-4, "min of x² should be ~0, got {min}");
    }

    #[test]
    fn test_golden_section_quadratic_with_offset() {
        // Minimum of (x - 0.3)² on [0, 1] is at 0.3.
        let min = ts_golden_section_search(|x| (x - 0.3) * (x - 0.3), 0.0, 1.0, 1e-8, 300);
        assert!(approx(min, 0.3, 1e-3), "expected min near 0.3, got {min}");
    }

    #[test]
    fn test_golden_section_inverted_parabola_boundary() {
        // -(x-2)² on [0,1] is maximised at x=1, i.e. minimum of (x-2)² on [0,1] at 1
        let min = ts_golden_section_search(|x| (x - 2.0) * (x - 2.0), 0.0, 1.0, 1e-8, 300);
        assert!(approx(min, 1.0, 1e-3), "expected 1.0, got {min}");
    }

    // ── TemperatureScaler::fit_binary ─────────────────────────────────────

    #[test]
    fn test_temperature_scaler_fit_balanced_data() {
        // Balanced, roughly calibrated data with moderate logits (not extreme).
        // Use logits where sigmoid(logit) ≈ label, so calibration is already near-optimal.
        // logit=0 → p=0.5 → label=0 or 1 alternating.
        // We use soft logits near 0 so the optimal T is not degenerate.
        let logits = vec![-1.0_f32, -0.5, 0.0, 0.5, 1.0, -1.0, -0.5, 0.0, 0.5, 1.0];
        let labels = vec![0.0_f32, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 1.0];
        let mut ts = TemperatureScaler::new(1.0).expect("ok");
        let cfg = CalibrationConfig::default();
        ts.fit_binary(&logits, &labels, &cfg).expect("fit ok");
        assert!(ts.fitted);
        // T should be positive and finite
        assert!(
            ts.temperature > 0.0,
            "T must be positive, got {}",
            ts.temperature
        );
        assert!(
            ts.temperature.is_finite(),
            "T must be finite, got {}",
            ts.temperature
        );
    }

    #[test]
    fn test_temperature_scaler_fit_overconfident() {
        // Overconfident model: large logits but only ~50% accuracy → T > 1
        let logits = vec![10.0_f32, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0];
        // But actual labels are ~50/50 mixed
        let labels = vec![1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut ts = TemperatureScaler::new(1.0).expect("ok");
        let cfg = CalibrationConfig::default();
        ts.fit_binary(&logits, &labels, &cfg).expect("fit ok");
        assert!(
            ts.temperature > 1.0,
            "Overconfident model should yield T > 1, got {}",
            ts.temperature
        );
    }

    #[test]
    fn test_temperature_scaler_fit_length_mismatch_error() {
        let mut ts = TemperatureScaler::new(1.0).expect("ok");
        let cfg = CalibrationConfig::default();
        assert!(matches!(
            ts.fit_binary(&[1.0, 2.0], &[0.0], &cfg),
            Err(CalibrationError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn test_temperature_scaler_fit_empty_error() {
        let mut ts = TemperatureScaler::new(1.0).expect("ok");
        let cfg = CalibrationConfig::default();
        assert!(matches!(
            ts.fit_binary(&[], &[], &cfg),
            Err(CalibrationError::EmptyInput)
        ));
    }

    // ── TemperatureScaler::scale ──────────────────────────────────────────

    #[test]
    fn test_temperature_scaler_scale_output_in_01() {
        let ts = TemperatureScaler::new(1.5).expect("ok");
        for &logit in &[-10.0_f32, -1.0, 0.0, 1.0, 10.0] {
            let p = ts.scale(logit);
            assert!(p > 0.0 && p < 1.0, "p={p} not in (0,1)");
        }
    }

    #[test]
    fn test_temperature_scaler_scale_batch_length_preserved() {
        let ts = TemperatureScaler::new(1.0).expect("ok");
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let out = ts.scale_batch(&logits);
        assert_eq!(out.len(), logits.len());
    }

    #[test]
    fn test_temperature_scaler_scale_zero_logit_half() {
        // σ(0 / T) = 0.5 for any T
        let ts = TemperatureScaler::new(2.0).expect("ok");
        assert!(approx(ts.scale(0.0), 0.5, 1e-6));
    }

    #[test]
    fn test_temperature_scaler_t1_equals_sigmoid() {
        let ts = TemperatureScaler::new(1.0).expect("ok");
        for &logit in &[-3.0_f32, 0.0, 3.0] {
            assert!(approx(ts.scale(logit), sigmoid(logit), 1e-7));
        }
    }

    #[test]
    fn test_temperature_scaler_large_t_conservative() {
        // Large T → predictions near 0.5
        let ts_large = TemperatureScaler::new(100.0).expect("ok");
        let p = ts_large.scale(5.0);
        assert!(
            approx(p, 0.5, 0.05),
            "Large T should give p near 0.5, got {p}"
        );
    }

    #[test]
    fn test_temperature_scaler_small_t_overconfident() {
        // Small T → predictions near 0 or 1
        let ts_small = TemperatureScaler::new(0.01).expect("ok");
        let p_pos = ts_small.scale(1.0);
        let p_neg = ts_small.scale(-1.0);
        assert!(
            p_pos > 0.99,
            "Small T should give p near 1 for pos logit, got {p_pos}"
        );
        assert!(
            p_neg < 0.01,
            "Small T should give p near 0 for neg logit, got {p_neg}"
        );
    }

    // ── PlattScaler ───────────────────────────────────────────────────────

    #[test]
    fn test_platt_scaler_fit_no_error() {
        let mut ps = PlattScaler::new();
        let logits = vec![1.0_f32, -1.0, 2.0, -2.0];
        let labels = vec![1.0_f32, 0.0, 1.0, 0.0];
        let cfg = CalibrationConfig::default();
        ps.fit(&logits, &labels, &cfg).expect("platt fit ok");
        assert!(ps.fitted);
    }

    #[test]
    fn test_platt_scaler_fit_identity_approx() {
        // Well-calibrated logits → a ≈ 1, b ≈ 0 (approximately)
        let mut ps = PlattScaler::new();
        let logits: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
        let labels: Vec<f32> = logits
            .iter()
            .map(|&l| if l > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let cfg = CalibrationConfig {
            max_iters: 2000,
            learning_rate: 0.01,
            ..Default::default()
        };
        ps.fit(&logits, &labels, &cfg).expect("fit ok");
        // a should be positive
        assert!(ps.a > 0.0, "a should be positive, got {}", ps.a);
    }

    #[test]
    fn test_platt_scaler_predict_output_in_01() {
        let ps = PlattScaler::new();
        for &logit in &[-10.0_f32, 0.0, 10.0] {
            let p = ps.predict(logit);
            assert!(p > 0.0 && p < 1.0, "p={p} not in (0,1)");
        }
    }

    #[test]
    fn test_platt_scaler_predict_batch_length() {
        let ps = PlattScaler::new();
        let logits = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(ps.predict_batch(&logits).len(), 3);
    }

    #[test]
    fn test_platt_scaler_fit_empty_error() {
        let mut ps = PlattScaler::new();
        let cfg = CalibrationConfig::default();
        assert!(matches!(
            ps.fit(&[], &[], &cfg),
            Err(CalibrationError::EmptyInput)
        ));
    }

    #[test]
    fn test_platt_scaler_fit_mismatch_error() {
        let mut ps = PlattScaler::new();
        let cfg = CalibrationConfig::default();
        assert!(matches!(
            ps.fit(&[1.0, 2.0], &[0.0], &cfg),
            Err(CalibrationError::LengthMismatch { .. })
        ));
    }

    // ── ts_pav_isotonic ───────────────────────────────────────────────────

    #[test]
    fn test_pav_isotonic_monotone() {
        let values = vec![0.9_f32, 0.1, 0.8, 0.2, 0.7];
        let weights = vec![1.0_f32; 5];
        let result = ts_pav_isotonic(&values, &weights);
        assert_eq!(result.len(), 5);
        for i in 0..result.len() - 1 {
            assert!(
                result[i] <= result[i + 1] + 1e-6,
                "PAV result must be non-decreasing, got [{i}]={} > [{}]={}",
                result[i],
                i + 1,
                result[i + 1]
            );
        }
    }

    #[test]
    fn test_pav_isotonic_constant_input() {
        let values = vec![0.5_f32; 5];
        let weights = vec![1.0_f32; 5];
        let result = ts_pav_isotonic(&values, &weights);
        for &v in &result {
            assert!(approx(v, 0.5, 1e-6), "Constant input → constant output");
        }
    }

    #[test]
    fn test_pav_isotonic_empty_input() {
        let result = ts_pav_isotonic(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pav_isotonic_already_sorted() {
        // Already monotone input should pass through unchanged
        let values = vec![0.1_f32, 0.3, 0.5, 0.7, 0.9];
        let weights = vec![1.0_f32; 5];
        let result = ts_pav_isotonic(&values, &weights);
        for (r, v) in result.iter().zip(values.iter()) {
            assert!(approx(*r, *v, 1e-5));
        }
    }

    #[test]
    fn test_pav_isotonic_single_element() {
        let result = ts_pav_isotonic(&[0.7_f32], &[1.0]);
        assert_eq!(result.len(), 1);
        assert!(approx(result[0], 0.7, 1e-6));
    }

    // ── IsotonicCalibrator ────────────────────────────────────────────────

    #[test]
    fn test_isotonic_calibrator_fit_no_error() {
        let mut ic = IsotonicCalibrator::new();
        let scores = vec![0.1_f32, 0.5, 0.9];
        let labels = vec![0.0_f32, 1.0, 1.0];
        ic.fit(&scores, &labels).expect("isotonic fit ok");
        assert!(ic.fitted);
    }

    #[test]
    fn test_isotonic_calibrator_predict_in_01() {
        let mut ic = IsotonicCalibrator::new();
        let scores = vec![0.1_f32, 0.3, 0.6, 0.9];
        let labels = vec![0.0_f32, 0.0, 1.0, 1.0];
        ic.fit(&scores, &labels).expect("ok");
        for &s in &[0.0_f32, 0.2, 0.5, 0.8, 1.0] {
            let p = ic.predict(s);
            assert!((0.0..=1.0).contains(&p), "p={p} not in [0,1]");
        }
    }

    #[test]
    fn test_isotonic_calibrator_predict_batch_length() {
        let mut ic = IsotonicCalibrator::new();
        ic.fit(&[0.1_f32, 0.9], &[0.0_f32, 1.0]).expect("ok");
        let out = ic.predict_batch(&[0.0_f32, 0.5, 1.0]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_isotonic_calibrator_fit_empty_error() {
        let mut ic = IsotonicCalibrator::new();
        assert!(matches!(
            ic.fit(&[], &[]),
            Err(CalibrationError::EmptyInput)
        ));
    }

    // ── ECE / MCE / Overconfidence ────────────────────────────────────────

    #[test]
    fn test_ece_perfect_calibration_near_zero() {
        // Each sample's confidence equals its (binary) label exactly
        let n = 100;
        let mut conf = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            // confidence at midpoint of each decile bin
            let c = (i as f32 + 0.5) / n as f32;
            conf.push(c);
            // label = 1 with prob = c → for ECE test use deterministic: label = round(c)
            labels.push(if c >= 0.5 { 1.0_f32 } else { 0.0 });
        }
        // ECE won't be exactly 0 for deterministic labels, but should be small
        let ece = ts_ece(&conf, &labels, 10).expect("ece ok");
        assert!(
            ece < 0.3,
            "ECE={ece} should be relatively small for near-calibrated data"
        );
    }

    #[test]
    fn test_ece_all_confident_but_wrong() {
        // All confident (p=0.99) but all wrong (label=0) → high ECE
        let conf = vec![0.99_f32; 100];
        let labels = vec![0.0_f32; 100];
        let ece = ts_ece(&conf, &labels, 10).expect("ece ok");
        assert!(
            ece > 0.5,
            "ECE={ece} should be high for confidently-wrong predictions"
        );
    }

    #[test]
    fn test_mce_geq_ece() {
        let conf = vec![0.9_f32, 0.8, 0.1, 0.2, 0.6];
        let labels = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0];
        let ece = ts_ece(&conf, &labels, 5).expect("ece ok");
        let mce = ts_mce(&conf, &labels, 5).expect("mce ok");
        assert!(mce >= ece - 1e-6, "MCE={mce} should be >= ECE={ece}");
    }

    #[test]
    fn test_overconfidence_error_nonnegative() {
        let conf = vec![0.9_f32, 0.1, 0.7];
        let labels = vec![0.0_f32, 1.0, 0.5];
        let oe = ts_overconfidence_error(&conf, &labels, 5).expect("oe ok");
        assert!(oe >= 0.0, "Overconfidence error must be >= 0, got {oe}");
    }

    #[test]
    fn test_overconfidence_error_underconfident_model() {
        // If conf < acc everywhere, overconfidence error should be 0
        let conf = vec![0.01_f32; 50];
        let labels = vec![1.0_f32; 50];
        let oe = ts_overconfidence_error(&conf, &labels, 10).expect("oe ok");
        assert!(approx(oe, 0.0, 1e-5), "No overconfidence → OE=0, got {oe}");
    }

    #[test]
    fn test_ece_empty_error() {
        assert!(matches!(
            ts_ece(&[], &[], 10),
            Err(CalibrationError::EmptyInput)
        ));
    }

    #[test]
    fn test_mce_empty_error() {
        assert!(matches!(
            ts_mce(&[], &[], 10),
            Err(CalibrationError::EmptyInput)
        ));
    }

    // ── Reliability diagram ───────────────────────────────────────────────

    #[test]
    fn test_reliability_diagram_bin_edges_count() {
        let conf = vec![0.1_f32, 0.5, 0.9];
        let labels = vec![0.0_f32, 1.0, 1.0];
        let diag = ts_reliability_diagram(&conf, &labels, 10).expect("diag ok");
        assert_eq!(diag.bin_edges.len(), 11, "n_bins+1 edges");
    }

    #[test]
    fn test_reliability_diagram_bin_counts_sum_to_n() {
        let conf = vec![0.1_f32, 0.5, 0.9];
        let labels = vec![0.0_f32, 1.0, 1.0];
        let diag = ts_reliability_diagram(&conf, &labels, 10).expect("diag ok");
        let total: usize = diag.bin_counts.iter().sum();
        assert_eq!(total, conf.len(), "bin counts should sum to N");
    }

    #[test]
    fn test_reliability_diagram_ece_mce_nonnegative() {
        let conf = vec![0.3_f32, 0.7, 0.5];
        let labels = vec![0.0_f32, 1.0, 1.0];
        let diag = ts_reliability_diagram(&conf, &labels, 10).expect("diag ok");
        assert!(diag.ece >= 0.0);
        assert!(diag.mce >= 0.0);
    }

    #[test]
    fn test_reliability_diagram_empty_error() {
        assert!(matches!(
            ts_reliability_diagram(&[], &[], 10),
            Err(CalibrationError::EmptyInput)
        ));
    }

    #[test]
    fn test_format_reliability_diagram_nonempty() {
        let conf = vec![0.1_f32, 0.5, 0.9];
        let labels = vec![0.0_f32, 1.0, 1.0];
        let diag = ts_reliability_diagram(&conf, &labels, 5).expect("diag ok");
        let s = ts_format_reliability_diagram(&diag);
        assert!(!s.is_empty(), "Format should produce non-empty string");
        assert!(s.contains("ECE"), "Should contain ECE label");
    }

    // ── Brier score ───────────────────────────────────────────────────────

    #[test]
    fn test_brier_score_perfect_predictions_zero() {
        let conf = vec![1.0_f32; 10];
        let labels = vec![1.0_f32; 10];
        let bs = ts_brier_score(&conf, &labels).expect("ok");
        assert!(
            approx(bs, 0.0, 1e-7),
            "Perfect predictions → Brier=0, got {bs}"
        );
    }

    #[test]
    fn test_brier_score_worst_predictions() {
        // Predict 1 when label=0 (or 0 when label=1) → Brier = 1
        let conf = vec![1.0_f32; 10];
        let labels = vec![0.0_f32; 10];
        let bs = ts_brier_score(&conf, &labels).expect("ok");
        assert!(
            approx(bs, 1.0, 1e-6),
            "Worst predictions → Brier=1, got {bs}"
        );
    }

    #[test]
    fn test_brier_score_empty_error() {
        assert!(matches!(
            ts_brier_score(&[], &[]),
            Err(CalibrationError::EmptyInput)
        ));
    }

    #[test]
    fn test_brier_score_nonnegative() {
        let mut state = 0xCAFE_u64;
        let conf: Vec<f32> = (0..20).map(|_| xorshift_f32(&mut state)).collect();
        let labels: Vec<f32> = (0..20)
            .map(|_| {
                if xorshift_f32(&mut state) > 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let bs = ts_brier_score(&conf, &labels).expect("ok");
        assert!(bs >= 0.0);
    }

    // ── Log loss ──────────────────────────────────────────────────────────

    #[test]
    fn test_log_loss_perfect_near_zero() {
        let conf = vec![0.9999_f32; 10];
        let labels = vec![1.0_f32; 10];
        let ll = ts_log_loss(&conf, &labels, 1e-7).expect("ok");
        assert!(
            ll < 0.01,
            "Near-perfect predictions → low log-loss, got {ll}"
        );
    }

    #[test]
    fn test_log_loss_worst_large() {
        let conf = vec![1e-7_f32; 10];
        let labels = vec![1.0_f32; 10];
        let ll = ts_log_loss(&conf, &labels, 1e-15).expect("ok");
        assert!(ll > 10.0, "Worst log-loss should be large, got {ll}");
    }

    #[test]
    fn test_log_loss_empty_error() {
        assert!(matches!(
            ts_log_loss(&[], &[], 1e-7),
            Err(CalibrationError::EmptyInput)
        ));
    }

    // ── CalibrationStats ──────────────────────────────────────────────────

    #[test]
    fn test_compute_stats_brier_matches_standalone() {
        let conf = vec![0.2_f32, 0.7, 0.4, 0.9];
        let labels = vec![0.0_f32, 1.0, 0.0, 1.0];
        let stats = ts_compute_stats(&conf, &labels).expect("ok");
        let brier = ts_brier_score(&conf, &labels).expect("ok");
        assert!(approx(stats.brier_score, brier, 1e-6));
    }

    #[test]
    fn test_compute_stats_log_loss_matches_standalone() {
        let conf = vec![0.2_f32, 0.7, 0.4, 0.9];
        let labels = vec![0.0_f32, 1.0, 0.0, 1.0];
        let stats = ts_compute_stats(&conf, &labels).expect("ok");
        let ll = ts_log_loss(&conf, &labels, 1e-7).expect("ok");
        assert!(approx(stats.log_loss, ll, 1e-5));
    }

    #[test]
    fn test_compute_stats_empty_error() {
        assert!(matches!(
            ts_compute_stats(&[], &[]),
            Err(CalibrationError::EmptyInput)
        ));
    }

    #[test]
    fn test_compute_stats_mean_confidence_correct() {
        let conf = vec![0.0_f32, 0.5, 1.0];
        let labels = vec![0.0_f32, 0.0, 1.0];
        let stats = ts_compute_stats(&conf, &labels).expect("ok");
        assert!(approx(stats.mean_confidence, 0.5, 1e-6));
    }

    #[test]
    fn test_compute_stats_confidence_std_nonneg() {
        let conf = vec![0.1_f32, 0.9, 0.5, 0.3];
        let labels = vec![0.0_f32, 1.0, 1.0, 0.0];
        let stats = ts_compute_stats(&conf, &labels).expect("ok");
        assert!(stats.confidence_std >= 0.0);
    }

    // ── Format functions ──────────────────────────────────────────────────

    #[test]
    fn test_format_stats_nonempty() {
        let conf = vec![0.7_f32, 0.3];
        let labels = vec![1.0_f32, 0.0];
        let stats = ts_compute_stats(&conf, &labels).expect("ok");
        let s = ts_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("brier"));
    }

    #[test]
    fn test_format_result_nonempty() {
        let result = CalibrationResult {
            pre_ece: 0.1,
            post_ece: 0.05,
            pre_mce: 0.2,
            post_mce: 0.1,
            method: "temperature".to_string(),
            n_samples: 100,
            iterations_used: 50,
        };
        let s = ts_format_result(&result);
        assert!(!s.is_empty());
        assert!(s.contains("temperature"));
    }

    #[test]
    fn test_format_result_display_trait() {
        let result = CalibrationResult {
            pre_ece: 0.1,
            post_ece: 0.05,
            pre_mce: 0.2,
            post_mce: 0.1,
            method: "platt".to_string(),
            n_samples: 50,
            iterations_used: 1000,
        };
        let s = format!("{}", result);
        assert!(s.contains("platt"));
    }

    // ── Sigmoid helper ────────────────────────────────────────────────────

    #[test]
    fn test_sigmoid_zero_is_half() {
        assert!(approx(sigmoid(0.0), 0.5, 1e-9));
    }

    #[test]
    fn test_sigmoid_large_pos_near_one() {
        assert!(sigmoid(100.0) > 0.999);
    }

    #[test]
    fn test_sigmoid_large_neg_near_zero() {
        assert!(sigmoid(-100.0) < 0.001);
    }

    // ── Platt scaler default ──────────────────────────────────────────────

    #[test]
    fn test_platt_scaler_default_identity() {
        let ps = PlattScaler::default();
        assert!(approx(ps.a, 1.0, 1e-9));
        assert!(approx(ps.b, 0.0, 1e-9));
    }

    // ── CalibrationConfig default ─────────────────────────────────────────

    #[test]
    fn test_calibration_config_default_values() {
        let cfg = CalibrationConfig::default();
        assert_eq!(cfg.n_bins, 10);
        assert_eq!(cfg.max_iters, 1000);
        assert!(approx(cfg.tolerance, 1e-6, 1e-10));
        assert!(approx(cfg.learning_rate, 0.01, 1e-9));
    }
}
