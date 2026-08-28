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

    /// `weights` passed to [`ts_pav_isotonic`] has a different length than `values`.
    #[error("Weights length mismatch: expected {expected} (matching values), got {got}")]
    WeightsLengthMismatch { expected: usize, got: usize },

    /// A weight passed to [`ts_pav_isotonic`] was non-finite or non-positive.
    #[error("Invalid weight {weight} at index {index}: weights must be finite and > 0")]
    InvalidWeight { index: usize, weight: f32 },
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
/// Terminates when the bracket width falls below `tol`, when the bracket can no
/// longer shrink at `f32` resolution (see [`ts_golden_section_search_tracked`]),
/// or when `max_iters` is reached.
pub fn ts_golden_section_search(
    f: impl Fn(f32) -> f32,
    lo: f32,
    hi: f32,
    tol: f32,
    max_iters: usize,
) -> f32 {
    ts_golden_section_search_tracked(f, lo, hi, tol, max_iters).0
}

/// Identical algorithm to [`ts_golden_section_search`], additionally
/// reporting how many iterations actually ran as `.1`.
///
/// [`ts_golden_section_search`] delegates to this function (single source of
/// truth, so the two can never diverge) rather than the reverse, so as to
/// leave that function's existing signature and callers unaffected. The
/// iteration count comes from the real search counting itself rather than
/// separately predicting the bracket's shrink rate, since `f32` precision
/// limits how small `b - a` can get relative to `lo`/`hi`'s magnitude —
/// a width-only prediction can be badly wrong once `tol` approaches that
/// floor, even though it never runs into it.
///
/// # Termination
///
/// Three conditions stop the loop:
///
/// 1. `|b - a| < tol` — the requested tolerance was met.
/// 2. The bracket failed to shrink during an iteration. Each golden-section
///    step multiplies the width by `1 − φ ≈ 0.618`, so in exact arithmetic the
///    width strictly decreases every iteration. It stops decreasing only once
///    `φ · (b − a)` drops below half an ulp of `a`/`b`, at which point the
///    interior points `a + φ(b − a)` and `b − φ(b − a)` round back onto `a`
///    and `b` themselves: the bracket is frozen and every further iteration
///    is a pure no-op that re-evaluates `f` for nothing.
/// 3. `max_iters` is exhausted.
///
/// Condition 2 is what makes a `tol` finer than `f32` can represent near the
/// minimiser (e.g. `tol = 1e-8` on `[0, 1]`, where the spacing around `0.3` is
/// ≈ 2.98e-8) terminate at the resolution floor instead of spinning out the
/// whole `max_iters` budget. It never fires early: as long as the bracket is
/// wide enough for `φ · (b − a)` to be representable, the width strictly
/// decreases and the loop continues.
pub fn ts_golden_section_search_tracked(
    f: impl Fn(f32) -> f32,
    lo: f32,
    hi: f32,
    tol: f32,
    max_iters: usize,
) -> (f32, usize) {
    const PHI: f32 = 0.381_966_01; // (3 - sqrt(5)) / 2

    let mut a = lo;
    let mut b = hi;
    let mut x1 = a + PHI * (b - a);
    let mut x2 = b - PHI * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    let mut width = (b - a).abs();
    let mut iters_run = 0usize;
    for _ in 0..max_iters {
        if width < tol {
            break;
        }
        iters_run += 1;
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
        // Termination condition 2: the bracket hit the `f32` resolution floor
        // and can no longer shrink, so further iterations cannot improve on
        // the midpoint we already have.
        //
        // Phrased as "did not shrink" rather than `new_width >= width` so a
        // NaN width (a NaN `lo`/`hi`) also terminates instead of looping.
        let new_width = (b - a).abs();
        let shrank = new_width < width;
        if !shrank {
            break;
        }
        width = new_width;
    }
    ((a + b) * 0.5, iters_run)
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
    /// Number of golden-section-search iterations the last `fit_binary` call
    /// actually ran (see [`ts_golden_section_search_tracked`]).
    pub iterations_used: usize,
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
            iterations_used: 0,
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

        let (t_opt, iters_used) = ts_golden_section_search_tracked(
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
        self.iterations_used = iters_used;
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
    /// Number of gradient-descent iterations the last `fit` call actually ran.
    pub iterations_used: usize,
    /// Whether the last `fit` call's step size (`‖(Δa, Δb)‖`) dropped below
    /// `config.tolerance` before exhausting `config.max_iters`. `fit` still
    /// returns `Ok(())` either way; callers that need non-convergence to be
    /// fatal should check this flag, or use [`calibrate`] with
    /// [`CalibrationMethod::Platt`], which does.
    pub converged: bool,
}

impl PlattScaler {
    /// Create a new Platt scaler with identity initialisation (a=1, b=0).
    pub fn new() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            fitted: false,
            iterations_used: 0,
            converged: false,
        }
    }

    /// Fit slope and bias by gradient descent on binary cross-entropy.
    ///
    /// Uses up to `config.max_iters` steps at `config.learning_rate`, stopping
    /// early once `‖(Δa, Δb)‖ < config.tolerance` (previously `tolerance` was
    /// ignored and every call ran the full `max_iters` steps).
    /// `self.iterations_used`/`self.converged` report what actually happened.
    pub fn fit(
        &mut self,
        logits: &[f32],
        labels: &[f32],
        config: &CalibrationConfig,
    ) -> Result<(), CalibrationError> {
        ts_validate_inputs(logits, labels)?;

        let n = logits.len() as f32;
        let lr = config.learning_rate;
        let mut iters_run = 0usize;
        let mut converged = false;

        for _ in 0..config.max_iters {
            iters_run += 1;
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

            let delta_a = lr * grad_a;
            let delta_b = lr * grad_b;
            self.a -= delta_a;
            self.b -= delta_b;

            if (delta_a * delta_a + delta_b * delta_b).sqrt() < config.tolerance {
                converged = true;
                break;
            }
        }

        self.iterations_used = iters_run;
        self.converged = converged;
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
/// iteratively merging adjacent blocks that violate monotonicity. Each
/// block's value is its weighted average.
///
/// # Errors
/// [`CalibrationError::WeightsLengthMismatch`] if `weights.len() !=
/// values.len()`; [`CalibrationError::InvalidWeight`] if any weight is
/// non-finite or `<= 0.0` (would divide-by-zero into `NaN`/`Inf`, or invert
/// pooling order if negative).
pub fn ts_pav_isotonic(values: &[f32], weights: &[f32]) -> Result<Vec<f32>, CalibrationError> {
    let n = values.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if weights.len() != n {
        return Err(CalibrationError::WeightsLengthMismatch {
            expected: n,
            got: weights.len(),
        });
    }
    for (i, &w) in weights.iter().enumerate() {
        if !(w.is_finite() && w > 0.0) {
            return Err(CalibrationError::InvalidWeight {
                index: i,
                weight: w,
            });
        }
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
    Ok(result)
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

        let pav_values = ts_pav_isotonic(&sorted_labels, &weights)?;

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
// calibrate() — fit-and-evaluate entry point
// ---------------------------------------------------------------------------

/// Which calibration method [`calibrate`] should fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Single-parameter temperature scaling (Guo et al. 2017).
    Temperature,
    /// Two-parameter affine (slope + bias) Platt scaling.
    Platt,
    /// Non-parametric monotone isotonic regression (PAV).
    Isotonic,
}

/// Fit a calibrator on `(logits, labels)` and report a full before/after
/// [`CalibrationResult`]: pre-fit ECE/MCE on the raw sigmoid confidences,
/// post-fit ECE/MCE on the calibrated ones, and how many optimisation
/// iterations the fit actually used (golden-section iterations for
/// [`CalibrationMethod::Temperature`], gradient-descent iterations for
/// [`CalibrationMethod::Platt`], always `1` for
/// [`CalibrationMethod::Isotonic`] since PAV is an exact single pass).
///
/// # Errors
/// Propagates [`CalibrationError`] from input validation or from
/// golden-section search producing a non-finite temperature. Returns
/// [`CalibrationError::DidNotConverge`] for [`CalibrationMethod::Platt`] if
/// gradient descent exhausts `config.max_iters` without its step size
/// dropping below `config.tolerance` — see [`PlattScaler`]'s `converged`
/// field for a non-fatal way to fit Platt scaling even when it may not
/// converge.
pub fn calibrate(
    logits: &[f32],
    labels: &[f32],
    method: CalibrationMethod,
    config: &CalibrationConfig,
) -> Result<CalibrationResult, CalibrationError> {
    ts_validate_inputs(logits, labels)?;

    // Pre-calibration confidences: raw logits through a plain sigmoid (T=1).
    let pre_confidences: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let pre_ece = ts_ece(&pre_confidences, labels, config.n_bins)?;
    let pre_mce = ts_mce(&pre_confidences, labels, config.n_bins)?;

    let (post_confidences, method_name, iterations_used): (Vec<f32>, &str, usize) = match method {
        CalibrationMethod::Temperature => {
            let mut scaler = TemperatureScaler::new(1.0)?;
            scaler.fit_binary(logits, labels, config)?;
            (
                scaler.scale_batch(logits),
                "temperature",
                scaler.iterations_used,
            )
        }
        CalibrationMethod::Platt => {
            let mut scaler = PlattScaler::new();
            scaler.fit(logits, labels, config)?;
            if !scaler.converged {
                return Err(CalibrationError::DidNotConverge {
                    iters: scaler.iterations_used,
                });
            }
            (
                scaler.predict_batch(logits),
                "platt",
                scaler.iterations_used,
            )
        }
        CalibrationMethod::Isotonic => {
            // PAV only needs monotonic order, which raw logits already
            // provide (sigmoid is monotonic, so sorting by logit or by
            // sigmoid(logit) yields identical PAV groupings) — fit and
            // predict directly in logit-space.
            let mut calibrator = IsotonicCalibrator::new();
            calibrator.fit(logits, labels)?;
            (calibrator.predict_batch(logits), "isotonic", 1)
        }
    };

    let post_ece = ts_ece(&post_confidences, labels, config.n_bins)?;
    let post_mce = ts_mce(&post_confidences, labels, config.n_bins)?;

    Ok(CalibrationResult {
        pre_ece,
        post_ece,
        pre_mce,
        post_mce,
        method: method_name.to_string(),
        n_samples: logits.len(),
        iterations_used,
    })
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
mod tests;
