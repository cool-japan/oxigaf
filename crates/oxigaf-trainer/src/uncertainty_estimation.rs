//! Uncertainty estimation for 3DGS avatar rendering quality.
//!
//! Provides Monte Carlo dropout-based uncertainty, ensemble variance,
//! calibration tools, and per-Gaussian confidence scoring.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the uncertainty estimation subsystem.
#[derive(Debug, Error)]
pub enum UncertaintyError {
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("Empty predictions")]
    EmptyPredictions,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Insufficient samples: need at least {needed}, got {got}")]
    InsufficientSamples { needed: usize, got: usize },

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Empty bins")]
    EmptyBins,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for uncertainty estimation.
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Number of MC samples (default 20).
    pub num_samples: usize,
    /// Dropout probability in [0, 1) (default 0.1).
    pub dropout_rate: f32,
    /// Number of ensemble members (default 5).
    pub ensemble_size: usize,
    /// Threshold for high-confidence classification (default 0.8).
    pub confidence_threshold: f32,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            num_samples: 20,
            dropout_rate: 0.1,
            ensemble_size: 5,
            confidence_threshold: 0.8,
        }
    }
}

impl UncertaintyConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), UncertaintyError> {
        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err(UncertaintyError::InvalidConfig(format!(
                "dropout_rate must be in [0, 1), got {}",
                self.dropout_rate
            )));
        }
        if self.num_samples < 2 {
            return Err(UncertaintyError::InvalidConfig(format!(
                "num_samples must be >= 2, got {}",
                self.num_samples
            )));
        }
        if self.ensemble_size < 1 {
            return Err(UncertaintyError::InvalidConfig(
                "ensemble_size must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Uncertainty decomposition into aleatoric and epistemic components.
#[derive(Debug, Clone)]
pub struct UncertaintyDecomposition {
    /// Total variance.
    pub total: f32,
    /// Data uncertainty (irreducible).
    pub aleatoric: f32,
    /// Model uncertainty (reducible with more data).
    pub epistemic: f32,
}

/// Per-prediction uncertainty estimate.
#[derive(Debug, Clone)]
pub struct PredictionUncertainty {
    /// Mean prediction.
    pub mean: f32,
    /// Variance of predictions.
    pub variance: f32,
    /// Standard deviation.
    pub std: f32,
    /// Predictive entropy.
    pub entropy: f32,
    /// BALD approximation (mutual information).
    pub mutual_information: f32,
    /// Confidence: 1 - normalized_variance.
    pub confidence: f32,
}

/// Calibration result for a model.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Expected Calibration Error.
    pub ece: f32,
    /// Maximum Calibration Error.
    pub mce: f32,
    /// Fraction of predictions that are overconfident.
    pub overconfident_fraction: f32,
    /// Fraction of predictions that are underconfident.
    pub underconfident_fraction: f32,
    /// Reliability diagram: (confidence, accuracy) per bin.
    pub reliability_diagram: Vec<(f32, f32)>,
}

/// Per-pixel confidence map for a rendered image.
#[derive(Debug, Clone)]
pub struct ConfidenceMap {
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Per-pixel confidence values in [0, 1].
    pub data: Vec<f32>,
}

impl ConfidenceMap {
    /// Create a new confidence map filled with `fill`.
    pub fn new(width: usize, height: usize, fill: f32) -> Self {
        let n = width * height;
        Self {
            width,
            height,
            data: vec![fill; n],
        }
    }

    /// Build a confidence map from a variance map.
    ///
    /// Confidence is computed as `exp(-variance)` clamped to [0, 1].
    pub fn from_variance_map(
        variance: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Self, UncertaintyError> {
        let expected = width * height;
        if variance.len() != expected {
            return Err(UncertaintyError::DimensionMismatch {
                expected,
                actual: variance.len(),
            });
        }
        let data = variance
            .iter()
            .map(|&v| (-v).exp().clamp(0.0, 1.0))
            .collect();
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Mean confidence over all pixels.
    pub fn mean_confidence(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.data.iter().sum();
        sum / self.data.len() as f32
    }

    /// Fraction of pixels with confidence below `threshold`.
    pub fn low_confidence_fraction(&self, threshold: f32) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let count = self.data.iter().filter(|&&c| c < threshold).count();
        count as f32 / self.data.len() as f32
    }

    /// Encode confidence map as an RGB heatmap using the jet colormap.
    ///
    /// Returns a `width * height * 3` byte buffer (R, G, B order).
    pub fn to_heatmap_rgb(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.data.len() * 3);
        for &c in &self.data {
            let (r, g, b) = jet_rgb(c);
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// Auxiliary structures
// ---------------------------------------------------------------------------

/// Aggregate uncertainty statistics for a spatial region.
#[derive(Debug, Clone)]
pub struct RegionUncertainty {
    /// Region identifier.
    pub region_id: usize,
    /// Mean variance within the region.
    pub mean_variance: f32,
    /// Maximum variance within the region.
    pub max_variance: f32,
    /// Fraction of region pixels with variance above threshold.
    pub high_uncertainty_fraction: f32,
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Compute MC dropout statistics across samples.
///
/// Returns `(means, variances)` where each vector has length equal to the
/// inner prediction dimension.
pub fn mc_dropout_stats(samples: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<f32>), UncertaintyError> {
    if samples.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    if samples.len() < 2 {
        return Err(UncertaintyError::InsufficientSamples {
            needed: 2,
            got: samples.len(),
        });
    }
    let dim = samples[0].len();
    for s in samples.iter() {
        if s.len() != dim {
            return Err(UncertaintyError::DimensionMismatch {
                expected: dim,
                actual: s.len(),
            });
        }
    }
    let n = samples.len() as f32;
    let mut means = vec![0.0f32; dim];
    for s in samples {
        for (m, &v) in means.iter_mut().zip(s.iter()) {
            *m += v;
        }
    }
    for m in &mut means {
        *m /= n;
    }
    let mut variances = vec![0.0f32; dim];
    for s in samples {
        for (var, (&v, &mu)) in variances.iter_mut().zip(s.iter().zip(means.iter())) {
            let diff = v - mu;
            *var += diff * diff;
        }
    }
    for var in &mut variances {
        *var /= n;
    }
    Ok((means, variances))
}

/// Apply an inverted-dropout mask to a parameter vector.
///
/// Uses an xorshift64 PRNG; each element is zeroed with probability
/// `dropout_rate` and every surviving element is rescaled by
/// `1 / (1 - dropout_rate)` so that the expectation of the masked vector
/// equals the input. That rescale is what makes repeated masks feed
/// [`mc_dropout_stats`] an unbiased Monte Carlo dropout estimator: without it
/// the means come out low by a factor of `1 - dropout_rate` and the variances
/// by `(1 - dropout_rate)^2`. The seed guard ensures the PRNG state is
/// non-zero.
///
/// # Errors
///
/// Returns [`UncertaintyError::InvalidConfig`] unless `dropout_rate` lies in
/// `[0, 1)`.
pub fn apply_dropout_mask(
    params: &[f32],
    dropout_rate: f32,
    seed: u64,
) -> Result<Vec<f32>, UncertaintyError> {
    if !(0.0..1.0).contains(&dropout_rate) {
        return Err(UncertaintyError::InvalidConfig(format!(
            "dropout_rate must be in [0, 1), got {}",
            dropout_rate
        )));
    }
    // Guaranteed positive by the range check above; guarded anyway so the
    // inverted-dropout scale can never become infinite.
    let keep_prob = 1.0 - dropout_rate;
    if keep_prob <= 0.0 {
        return Err(UncertaintyError::NumericalError(format!(
            "keep probability 1 - dropout_rate must be > 0, got {}",
            keep_prob
        )));
    }
    let scale = 1.0 / keep_prob;
    let mut state = seed.max(1);
    let threshold = (dropout_rate as f64 * u64::MAX as f64) as u64;
    let mut out = Vec::with_capacity(params.len());
    for &p in params {
        state = xorshift64(state);
        if state < threshold {
            out.push(0.0f32);
        } else {
            out.push(p * scale);
        }
    }
    Ok(out)
}

/// Compute ensemble variance across multiple model predictions.
///
/// Returns `(means, variances)` per position.
pub fn ensemble_variance(
    predictions: &[Vec<f32>],
) -> Result<(Vec<f32>, Vec<f32>), UncertaintyError> {
    if predictions.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    let dim = predictions[0].len();
    for p in predictions.iter() {
        if p.len() != dim {
            return Err(UncertaintyError::DimensionMismatch {
                expected: dim,
                actual: p.len(),
            });
        }
    }
    let n = predictions.len() as f32;
    let mut means = vec![0.0f32; dim];
    for p in predictions {
        for (m, &v) in means.iter_mut().zip(p.iter()) {
            *m += v;
        }
    }
    for m in &mut means {
        *m /= n;
    }
    let mut variances = vec![0.0f32; dim];
    for p in predictions {
        for (var, (&v, &mu)) in variances.iter_mut().zip(p.iter().zip(means.iter())) {
            let diff = v - mu;
            *var += diff * diff;
        }
    }
    for var in &mut variances {
        *var /= n;
    }
    Ok((means, variances))
}

/// BALD score (Bayesian Active Learning by Disagreement).
///
/// Each row in `predictions` is a softmax probability vector over classes.
/// MI = H\[E\[p\]\] - E\[H\[p\]\].
pub fn bald_score(predictions: &[Vec<f32>]) -> Result<f32, UncertaintyError> {
    if predictions.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    if predictions.len() < 2 {
        return Err(UncertaintyError::InsufficientSamples {
            needed: 2,
            got: predictions.len(),
        });
    }
    let n_classes = predictions[0].len();
    for p in predictions {
        if p.len() != n_classes {
            return Err(UncertaintyError::DimensionMismatch {
                expected: n_classes,
                actual: p.len(),
            });
        }
    }
    let n = predictions.len() as f32;
    // Mean distribution
    let mut mean_probs = vec![0.0f32; n_classes];
    for p in predictions {
        for (m, &v) in mean_probs.iter_mut().zip(p.iter()) {
            *m += v;
        }
    }
    for m in &mut mean_probs {
        *m /= n;
    }
    // H[E[p]]
    let h_mean = entropy_of_probs(&mean_probs);
    // E[H[p]]
    let mean_h: f32 = predictions.iter().map(|p| entropy_of_probs(p)).sum::<f32>() / n;
    Ok((h_mean - mean_h).max(0.0))
}

/// Shannon entropy of a probability distribution.
///
/// Convention: `0 * ln(0) = 0`.
pub fn prediction_entropy(probs: &[f32]) -> Result<f32, UncertaintyError> {
    if probs.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    Ok(entropy_of_probs(probs))
}

/// Compute calibration metrics from per-prediction (confidence, accuracy) pairs.
///
/// `confidences[i]` is the max softmax probability; `accuracies[i]` is 1 if
/// correct, 0 otherwise. `bins` is the number of equal-width calibration bins.
pub fn compute_calibration(
    confidences: &[f32],
    accuracies: &[f32],
    bins: usize,
) -> Result<CalibrationResult, UncertaintyError> {
    if confidences.is_empty() || accuracies.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    if confidences.len() != accuracies.len() {
        return Err(UncertaintyError::DimensionMismatch {
            expected: confidences.len(),
            actual: accuracies.len(),
        });
    }
    if bins == 0 {
        return Err(UncertaintyError::EmptyBins);
    }
    let n = confidences.len() as f32;
    // Bin boundaries: [0, 1/bins), [1/bins, 2/bins), ...
    let mut bin_conf_sum = vec![0.0f32; bins];
    let mut bin_acc_sum = vec![0.0f32; bins];
    let mut bin_count = vec![0usize; bins];
    for (&c, &a) in confidences.iter().zip(accuracies.iter()) {
        let idx = ((c * bins as f32) as usize).min(bins - 1);
        bin_conf_sum[idx] += c;
        bin_acc_sum[idx] += a;
        bin_count[idx] += 1;
    }
    let mut ece = 0.0f32;
    let mut mce = 0.0f32;
    let mut reliability_diagram = Vec::with_capacity(bins);
    let mut overconfident = 0usize;
    let mut underconfident = 0usize;
    for b in 0..bins {
        if bin_count[b] == 0 {
            continue;
        }
        let cnt = bin_count[b] as f32;
        let avg_conf = bin_conf_sum[b] / cnt;
        let avg_acc = bin_acc_sum[b] / cnt;
        let gap = (avg_conf - avg_acc).abs();
        ece += (cnt / n) * gap;
        if gap > mce {
            mce = gap;
        }
        reliability_diagram.push((avg_conf, avg_acc));
        if avg_conf > avg_acc {
            overconfident += bin_count[b];
        } else if avg_acc > avg_conf {
            underconfident += bin_count[b];
        }
    }
    let total = confidences.len() as f32;
    Ok(CalibrationResult {
        ece,
        mce,
        overconfident_fraction: overconfident as f32 / total,
        underconfident_fraction: underconfident as f32 / total,
        reliability_diagram,
    })
}

/// Scale logits by temperature before applying softmax.
///
/// `temperature > 1` produces softer (more uniform) distributions.
pub fn temperature_scale(logits: &[f32], temperature: f32) -> Result<Vec<f32>, UncertaintyError> {
    if temperature <= 0.0 {
        return Err(UncertaintyError::InvalidConfig(format!(
            "temperature must be > 0, got {}",
            temperature
        )));
    }
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    stable_softmax(&scaled)
}

/// Numerically stable softmax: subtract max before exp.
pub fn stable_softmax(logits: &[f32]) -> Result<Vec<f32>, UncertaintyError> {
    if logits.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        return Err(UncertaintyError::NumericalError(
            "softmax denominator is zero or non-finite".to_string(),
        ));
    }
    Ok(exps.iter().map(|&e| e / sum).collect())
}

/// Per-Gaussian positional uncertainty from multiple renders.
///
/// `positions` is laid out as `[n_gaussians * n_renders * 3]` — all renders
/// concatenated. Returns variance per Gaussian (scalar, averaged over xyz).
pub fn per_gaussian_position_uncertainty(
    positions: &[f32],
    n_gaussians: usize,
    n_renders: usize,
) -> Result<Vec<f32>, UncertaintyError> {
    let expected = n_gaussians * n_renders * 3;
    if positions.len() != expected {
        return Err(UncertaintyError::DimensionMismatch {
            expected,
            actual: positions.len(),
        });
    }
    if n_gaussians == 0 {
        return Ok(vec![]);
    }
    if n_renders < 1 {
        return Err(UncertaintyError::InsufficientSamples {
            needed: 1,
            got: n_renders,
        });
    }
    let mut variances = vec![0.0f32; n_gaussians];
    for (g, variance) in variances.iter_mut().enumerate() {
        // Collect n_renders position samples (xyz) for this Gaussian.
        // Layout: for each render r, for each Gaussian g, xyz stored at
        // positions[r * n_gaussians * 3 + g * 3 .. +3]
        // BUT the spec says "all renders concatenated", so layout is
        // [g0_r0_xyz, g1_r0_xyz, ..., gN_r0_xyz, g0_r1_xyz, ...]
        // i.e. positions[r * n_gaussians * 3 + g * 3]
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for r in 0..n_renders {
            let base = r * n_gaussians * 3 + g * 3;
            sum_x += positions[base];
            sum_y += positions[base + 1];
            sum_z += positions[base + 2];
        }
        let n = n_renders as f32;
        let mx = sum_x / n;
        let my = sum_y / n;
        let mz = sum_z / n;
        let mut var_x = 0.0f32;
        let mut var_y = 0.0f32;
        let mut var_z = 0.0f32;
        for r in 0..n_renders {
            let base = r * n_gaussians * 3 + g * 3;
            let dx = positions[base] - mx;
            let dy = positions[base + 1] - my;
            let dz = positions[base + 2] - mz;
            var_x += dx * dx;
            var_y += dy * dy;
            var_z += dz * dz;
        }
        *variance = (var_x + var_y + var_z) / (3.0 * n);
    }
    Ok(variances)
}

/// Convert per-pixel variance to confidence: `exp(-variance)`.
pub fn variance_to_confidence(variance_map: &[f32]) -> Vec<f32> {
    variance_map.iter().map(|&v| (-v).exp()).collect()
}

/// Decompose the uncertainty of *point* predictions into aleatoric and
/// epistemic components.
///
/// Each row of `samples` is one stochastic forward pass (an MC-dropout pass or
/// an ensemble member) and `model_mean` is the reference prediction the spread
/// is measured against — usually the mean of `samples`, though a deterministic
/// forward pass works equally well.
///
/// Point predictions carry no estimate of observation noise, so the law of
/// total variance degenerates: the per-pass predictive variances are
/// identically zero, hence so is the data term, and all the spread is model
/// uncertainty. Concretely, for every output dimension `d`:
///
/// - `total[d]` — second moment of the passes about `model_mean[d]`, i.e.
///   `E_t[(samples[t][d] - model_mean[d])^2]`.
/// - `aleatoric[d]` — exactly `0.0`; observation noise is not identifiable
///   from point predictions alone and is therefore not invented here.
/// - `epistemic[d]` — equal to `total[d]`. It reduces to `Var_t(samples[t][d])`
///   when `model_mean` is the mean of `samples`, and otherwise also carries the
///   squared bias between that mean and `model_mean`.
///
/// Use [`decompose_uncertainty_with_variances`] when the model has a variance
/// head: feeding it the per-pass predictive variances is the only way to
/// obtain a genuine aleatoric/epistemic split.
pub fn decompose_uncertainty(
    samples: &[Vec<f32>],
    model_mean: &[f32],
) -> Result<Vec<UncertaintyDecomposition>, UncertaintyError> {
    if samples.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    let dim = model_mean.len();
    for s in samples {
        if s.len() != dim {
            return Err(UncertaintyError::DimensionMismatch {
                expected: dim,
                actual: s.len(),
            });
        }
    }
    let n = samples.len() as f32;
    // Total variance: E[(x - mu)^2] across all samples.
    let mut total_var = vec![0.0f32; dim];
    for s in samples {
        for (tv, (&v, &mu)) in total_var.iter_mut().zip(s.iter().zip(model_mean.iter())) {
            let diff = v - mu;
            *tv += diff * diff;
        }
    }
    for tv in &mut total_var {
        *tv /= n;
    }
    // Law of total variance with per-pass predictive variances identically
    // zero: aleatoric = E_t[0] = 0, so the whole second moment about
    // `model_mean` is model (epistemic) uncertainty. No split is fabricated;
    // callers with a variance head should use
    // `decompose_uncertainty_with_variances` instead.
    let result = total_var
        .into_iter()
        .map(|total| UncertaintyDecomposition {
            total,
            aleatoric: 0.0,
            epistemic: total,
        })
        .collect();
    Ok(result)
}

/// Decompose predictive uncertainty via the law of total variance.
///
/// `means[t]` and `variances[t]` are the predictive mean and the predicted
/// (observation-noise) variance of stochastic forward pass `t` — one
/// MC-dropout pass or ensemble member of a model with a variance head.
/// `model_mean` is the reference prediction the per-pass means are measured
/// against, normally the mean of `means`.
///
/// For every output dimension `d`:
///
/// - `aleatoric[d] = E_t[variances[t][d]]` — the data noise the model itself
///   predicts, i.e. `E_θ[Var(y | θ)]`. Irreducible: more data will not shrink
///   it.
/// - `epistemic[d] = E_t[(means[t][d] - model_mean[d])^2]` — the second moment
///   of the per-pass means about `model_mean[d]`. It equals `Var_θ(E[y | θ])`
///   when `model_mean` is the mean of `means`, and otherwise also carries the
///   squared bias between the two. Reducible: this is the term that flags
///   where more data or more capacity would help.
/// - `total[d] = aleatoric[d] + epistemic[d]`.
///
/// # Errors
///
/// Returns [`UncertaintyError::EmptyPredictions`] when `means` is empty,
/// [`UncertaintyError::DimensionMismatch`] when `variances` does not pair up
/// with `means` or a row length differs from `model_mean.len()`, and
/// [`UncertaintyError::NumericalError`] for a negative or non-finite predicted
/// variance — a broken variance head is reported rather than silently clamped.
pub fn decompose_uncertainty_with_variances(
    means: &[Vec<f32>],
    variances: &[Vec<f32>],
    model_mean: &[f32],
) -> Result<Vec<UncertaintyDecomposition>, UncertaintyError> {
    if means.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    if variances.len() != means.len() {
        return Err(UncertaintyError::DimensionMismatch {
            expected: means.len(),
            actual: variances.len(),
        });
    }
    let dim = model_mean.len();
    for (m, v) in means.iter().zip(variances.iter()) {
        if m.len() != dim {
            return Err(UncertaintyError::DimensionMismatch {
                expected: dim,
                actual: m.len(),
            });
        }
        if v.len() != dim {
            return Err(UncertaintyError::DimensionMismatch {
                expected: dim,
                actual: v.len(),
            });
        }
        for &sigma_sq in v.iter() {
            if !sigma_sq.is_finite() || sigma_sq < 0.0 {
                return Err(UncertaintyError::NumericalError(format!(
                    "predicted variance must be finite and non-negative, got {}",
                    sigma_sq
                )));
            }
        }
    }
    let n = means.len() as f32;
    // Aleatoric: E_theta[Var(y | theta)] — mean of the predicted variances.
    let mut aleatoric_sum = vec![0.0f32; dim];
    for v in variances {
        for (acc, &sigma_sq) in aleatoric_sum.iter_mut().zip(v.iter()) {
            *acc += sigma_sq;
        }
    }
    // Epistemic: spread of the per-pass means about the reference prediction.
    let mut epistemic_sum = vec![0.0f32; dim];
    for m in means {
        for (acc, (&mu_t, &mu)) in epistemic_sum
            .iter_mut()
            .zip(m.iter().zip(model_mean.iter()))
        {
            let diff = mu_t - mu;
            *acc += diff * diff;
        }
    }
    let result = aleatoric_sum
        .into_iter()
        .zip(epistemic_sum)
        .map(|(a_sum, e_sum)| {
            let aleatoric = a_sum / n;
            let epistemic = e_sum / n;
            UncertaintyDecomposition {
                total: aleatoric + epistemic,
                aleatoric,
                epistemic,
            }
        })
        .collect();
    Ok(result)
}

/// Return indices of predictions where `variance > threshold`.
pub fn high_uncertainty_indices(variances: &[f32], threshold: f32) -> Vec<usize> {
    variances
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v > threshold { Some(i) } else { None })
        .collect()
}

/// Uncertainty-weighted loss: weight each sample by `1 / (1 + variance)`.
pub fn uncertainty_weighted_loss(
    losses: &[f32],
    variances: &[f32],
) -> Result<f32, UncertaintyError> {
    if losses.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    if losses.len() != variances.len() {
        return Err(UncertaintyError::DimensionMismatch {
            expected: losses.len(),
            actual: variances.len(),
        });
    }
    let mut weighted_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for (&l, &v) in losses.iter().zip(variances.iter()) {
        let w = 1.0 / (1.0 + v);
        weighted_sum += w * l;
        weight_sum += w;
    }
    if weight_sum == 0.0 {
        return Err(UncertaintyError::NumericalError(
            "weight sum is zero".to_string(),
        ));
    }
    Ok(weighted_sum / weight_sum)
}

/// Convenience alias for the reliability diagram points.
pub fn reliability_diagram_points(calibration: &CalibrationResult) -> Vec<(f32, f32)> {
    calibration.reliability_diagram.clone()
}

/// Aggregate uncertainty statistics per spatial region.
pub fn aggregate_region_uncertainty(
    variances: &[f32],
    region_masks: &[Vec<bool>],
    threshold: f32,
) -> Result<Vec<RegionUncertainty>, UncertaintyError> {
    let n = variances.len();
    let mut result = Vec::with_capacity(region_masks.len());
    for (region_id, mask) in region_masks.iter().enumerate() {
        if mask.len() != n {
            return Err(UncertaintyError::DimensionMismatch {
                expected: n,
                actual: mask.len(),
            });
        }
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();
        if indices.is_empty() {
            result.push(RegionUncertainty {
                region_id,
                mean_variance: 0.0,
                max_variance: 0.0,
                high_uncertainty_fraction: 0.0,
            });
            continue;
        }
        let cnt = indices.len() as f32;
        let sum_var: f32 = indices.iter().map(|&i| variances[i]).sum();
        let max_var: f32 = indices
            .iter()
            .map(|&i| variances[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let high_count = indices
            .iter()
            .filter(|&&i| variances[i] > threshold)
            .count();
        result.push(RegionUncertainty {
            region_id,
            mean_variance: sum_var / cnt,
            max_variance: max_var,
            high_uncertainty_fraction: high_count as f32 / cnt,
        });
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// xorshift64 PRNG step.
#[inline]
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// Shannon entropy of a probability vector (convention: 0*ln(0) = 0).
#[inline]
fn entropy_of_probs(probs: &[f32]) -> f32 {
    probs
        .iter()
        .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum()
}

/// Jet colormap: maps value in [0, 1] → (R, G, B) as u8.
fn jet_rgb(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    // Jet: blue → cyan → green → yellow → red
    let r = (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // --- UncertaintyConfig ---

    #[test]
    fn test_config_default_valid() {
        let cfg = UncertaintyConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_dropout_rate_one_is_error() {
        let cfg = UncertaintyConfig {
            dropout_rate: 1.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_dropout_rate_negative_is_error() {
        let cfg = UncertaintyConfig {
            dropout_rate: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_num_samples_one_is_error() {
        let cfg = UncertaintyConfig {
            num_samples: 1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_ensemble_size_zero_is_error() {
        let cfg = UncertaintyConfig {
            ensemble_size: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // --- ConfidenceMap ---

    #[test]
    fn test_confidence_map_new() {
        let cm = ConfidenceMap::new(4, 4, 0.7);
        assert_eq!(cm.data.len(), 16);
        assert!(cm.data.iter().all(|&v| approx(v, 0.7, 1e-6)));
    }

    #[test]
    fn test_confidence_map_from_variance_zero() {
        let variance = vec![0.0f32; 6];
        let cm =
            ConfidenceMap::from_variance_map(&variance, 3, 2).expect("from_variance_map failed");
        assert!(cm.data.iter().all(|&v| approx(v, 1.0, 1e-6)));
    }

    #[test]
    fn test_confidence_map_from_variance_size_mismatch() {
        let variance = vec![0.0f32; 5];
        assert!(ConfidenceMap::from_variance_map(&variance, 3, 2).is_err());
    }

    #[test]
    fn test_confidence_map_mean_confidence() {
        let data = vec![0.5f32, 0.5f32];
        let cm = ConfidenceMap {
            width: 2,
            height: 1,
            data,
        };
        assert!(approx(cm.mean_confidence(), 0.5, 1e-6));
    }

    #[test]
    fn test_confidence_map_low_confidence_fraction() {
        let data = vec![0.3f32, 0.9f32, 0.1f32, 0.8f32];
        let cm = ConfidenceMap {
            width: 4,
            height: 1,
            data,
        };
        // Below 0.5: indices 0 and 2 → 2/4 = 0.5
        assert!(approx(cm.low_confidence_fraction(0.5), 0.5, 1e-6));
    }

    #[test]
    fn test_confidence_map_to_heatmap_rgb_length() {
        let cm = ConfidenceMap::new(3, 2, 0.5);
        let rgb = cm.to_heatmap_rgb();
        assert_eq!(rgb.len(), 3 * 2 * 3);
    }

    // --- mc_dropout_stats ---

    #[test]
    fn test_mc_dropout_stats_identical_samples() {
        let sample = vec![1.0f32, 2.0, 3.0];
        let samples = vec![sample.clone(), sample.clone(), sample.clone()];
        let (means, vars) = mc_dropout_stats(&samples).expect("mc_dropout_stats failed");
        assert!(approx(means[0], 1.0, 1e-5));
        assert!(vars.iter().all(|&v| approx(v, 0.0, 1e-5)));
    }

    #[test]
    fn test_mc_dropout_stats_two_different() {
        let s1 = vec![0.0f32, 0.0];
        let s2 = vec![2.0f32, 2.0];
        let (means, vars) = mc_dropout_stats(&[s1, s2]).expect("mc_dropout_stats failed");
        assert!(approx(means[0], 1.0, 1e-5));
        // var = ((0-1)^2 + (2-1)^2) / 2 = 1
        assert!(approx(vars[0], 1.0, 1e-5));
    }

    #[test]
    fn test_mc_dropout_stats_empty_error() {
        assert!(mc_dropout_stats(&[]).is_err());
    }

    #[test]
    fn test_mc_dropout_stats_single_sample_error() {
        assert!(mc_dropout_stats(&[vec![1.0f32]]).is_err());
    }

    // --- apply_dropout_mask ---

    #[test]
    fn test_apply_dropout_mask_rate_zero_identical() {
        let params = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = apply_dropout_mask(&params, 0.0, 42).expect("dropout failed");
        assert_eq!(out, params);
    }

    #[test]
    fn test_apply_dropout_mask_rate_one_error() {
        let params = vec![1.0f32];
        assert!(apply_dropout_mask(&params, 1.0, 42).is_err());
    }

    #[test]
    fn test_apply_dropout_mask_same_length() {
        let params = vec![1.0f32; 100];
        let out = apply_dropout_mask(&params, 0.5, 7).expect("dropout failed");
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_apply_dropout_mask_rescales_survivors() {
        // Regression: inverted dropout must scale survivors by 1/(1-p), so the
        // expectation of the masked vector matches the input. Without the
        // rescale the mean collapses to (1-p) = 0.5.
        let params = vec![1.0f32; 10_000];
        let out = apply_dropout_mask(&params, 0.5, 2024).expect("dropout failed");
        // Structural: every element is either dropped or exactly 1/(1-0.5) = 2.
        assert!(out.iter().all(|&v| v == 0.0 || v == 2.0));
        assert!(out.contains(&0.0), "expected some drops");
        assert!(out.contains(&2.0), "expected some survivors");
        // Statistical: mean preserved (0.5 away from the un-rescaled value).
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(approx(mean, 1.0, 0.1), "expected mean ~1.0, got {}", mean);
    }

    #[test]
    fn test_apply_dropout_mask_scale_matches_rate() {
        let params = vec![4.0f32; 256];
        let out = apply_dropout_mask(&params, 0.2, 99).expect("dropout failed");
        // Survivors: 4 * 1/(1 - 0.2) = 5.
        assert!(out.iter().all(|&v| v == 0.0 || approx(v, 5.0, 1e-4)));
    }

    #[test]
    fn test_apply_dropout_mask_high_rate_mostly_zero() {
        let params = vec![1.0f32; 1000];
        // dropout_rate of 0.99 — nearly all zeroed
        let out = apply_dropout_mask(&params, 0.99, 13).expect("dropout failed");
        let zero_count = out.iter().filter(|&&v| v == 0.0).count();
        // Expect at least 90% zeroed
        assert!(zero_count > 900, "Expected >90% zeros, got {}", zero_count);
    }

    // --- ensemble_variance ---

    #[test]
    fn test_ensemble_variance_single_member() {
        let preds = vec![vec![1.0f32, 2.0, 3.0]];
        let (means, vars) = ensemble_variance(&preds).expect("ensemble failed");
        assert!(approx(means[0], 1.0, 1e-6));
        assert!(vars.iter().all(|&v| approx(v, 0.0, 1e-6)));
    }

    #[test]
    fn test_ensemble_variance_two_identical() {
        let p = vec![5.0f32, 6.0];
        let preds = vec![p.clone(), p.clone()];
        let (_means, vars) = ensemble_variance(&preds).expect("ensemble failed");
        assert!(vars.iter().all(|&v| approx(v, 0.0, 1e-6)));
    }

    #[test]
    fn test_ensemble_variance_two_different() {
        let preds = vec![vec![0.0f32], vec![4.0f32]];
        let (means, vars) = ensemble_variance(&preds).expect("ensemble failed");
        assert!(approx(means[0], 2.0, 1e-5));
        // var = ((0-2)^2 + (4-2)^2) / 2 = 4
        assert!(approx(vars[0], 4.0, 1e-5));
    }

    // --- bald_score ---

    #[test]
    fn test_bald_score_uniform_predictions_zero_mi() {
        // All samples identical uniform → MI = 0
        let p = vec![0.25f32; 4];
        let preds = vec![p.clone(), p.clone(), p.clone()];
        let mi = bald_score(&preds).expect("bald failed");
        assert!(approx(mi, 0.0, 1e-5));
    }

    #[test]
    fn test_bald_score_diverse_predictions_positive_mi() {
        // One sample certain about class 0, another certain about class 1
        let s1 = vec![0.99f32, 0.01];
        let s2 = vec![0.01f32, 0.99];
        let mi = bald_score(&[s1, s2]).expect("bald failed");
        assert!(mi > 0.0, "Expected positive MI, got {}", mi);
    }

    // --- prediction_entropy ---

    #[test]
    fn test_prediction_entropy_uniform() {
        let n = 4;
        let p = vec![0.25f32; n];
        let h = prediction_entropy(&p).expect("entropy failed");
        let expected = (n as f32).ln();
        assert!(approx(h, expected, 1e-5));
    }

    #[test]
    fn test_prediction_entropy_certain() {
        let p = vec![1.0f32, 0.0, 0.0];
        let h = prediction_entropy(&p).expect("entropy failed");
        assert!(approx(h, 0.0, 1e-6));
    }

    #[test]
    fn test_prediction_entropy_empty_error() {
        assert!(prediction_entropy(&[]).is_err());
    }

    // --- compute_calibration ---

    #[test]
    fn test_compute_calibration_perfect() {
        // Perfect calibration: confidence equals accuracy at each bin
        let confidences = vec![0.1f32, 0.3, 0.5, 0.7, 0.9];
        let accuracies = vec![0.1f32, 0.3, 0.5, 0.7, 0.9];
        let cal = compute_calibration(&confidences, &accuracies, 10).expect("calibration failed");
        assert!(approx(cal.ece, 0.0, 1e-5));
    }

    #[test]
    fn test_compute_calibration_empty_error() {
        assert!(compute_calibration(&[], &[], 10).is_err());
    }

    #[test]
    fn test_compute_calibration_mismatched_lengths_error() {
        assert!(compute_calibration(&[0.5f32], &[0.5f32, 0.5f32], 10).is_err());
    }

    #[test]
    fn test_compute_calibration_zero_bins_error() {
        assert!(compute_calibration(&[0.5f32], &[1.0f32], 0).is_err());
    }

    // --- temperature_scale ---

    #[test]
    fn test_temperature_scale_t1_same_as_softmax() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let ts = temperature_scale(&logits, 1.0).expect("temp_scale failed");
        let sm = stable_softmax(&logits).expect("softmax failed");
        for (a, b) in ts.iter().zip(sm.iter()) {
            assert!(approx(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_temperature_scale_high_temp_approaches_uniform() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let ts = temperature_scale(&logits, 1000.0).expect("temp_scale failed");
        let expected = 1.0 / 3.0;
        for &v in &ts {
            assert!(approx(v, expected, 0.01));
        }
    }

    #[test]
    fn test_temperature_scale_zero_temp_error() {
        assert!(temperature_scale(&[1.0f32], 0.0).is_err());
    }

    // --- stable_softmax ---

    #[test]
    fn test_stable_softmax_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, -1.0];
        let sm = stable_softmax(&logits).expect("softmax failed");
        let sum: f32 = sm.iter().sum();
        assert!(approx(sum, 1.0, 1e-6));
    }

    #[test]
    fn test_stable_softmax_max_element_highest() {
        let logits = vec![1.0f32, 5.0, 2.0];
        let sm = stable_softmax(&logits).expect("softmax failed");
        let max_idx = sm
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(max_idx, 1);
    }

    #[test]
    fn test_stable_softmax_single_element() {
        let sm = stable_softmax(&[3.0f32]).expect("softmax failed");
        assert!(approx(sm[0], 1.0, 1e-6));
    }

    #[test]
    fn test_stable_softmax_empty_error() {
        assert!(stable_softmax(&[]).is_err());
    }

    // --- per_gaussian_position_uncertainty ---

    #[test]
    fn test_per_gaussian_uncertainty_single_render() {
        // 2 Gaussians, 1 render → zero variance
        let positions = vec![
            1.0f32, 0.0, 0.0, // g0 r0
            0.0f32, 1.0, 0.0, // g1 r0
        ];
        let vars = per_gaussian_position_uncertainty(&positions, 2, 1)
            .expect("per_gaussian_uncertainty failed");
        assert_eq!(vars.len(), 2);
        assert!(vars.iter().all(|&v| approx(v, 0.0, 1e-6)));
    }

    #[test]
    fn test_per_gaussian_uncertainty_two_renders() {
        // 1 Gaussian, 2 renders at different positions
        // Layout: [g0_r0_xyz, g0_r1_xyz]
        let positions = vec![
            0.0f32, 0.0, 0.0, // g0 r0
            2.0f32, 0.0, 0.0, // g0 r1
        ];
        let vars = per_gaussian_position_uncertainty(&positions, 1, 2)
            .expect("per_gaussian_uncertainty failed");
        assert_eq!(vars.len(), 1);
        // mean_x = 1, var_x = ((0-1)^2 + (2-1)^2)/2 = 1
        // var_y = var_z = 0; total = (1 + 0 + 0) / (3*2) = 1/6
        assert!(vars[0] > 0.0, "Expected positive variance");
    }

    #[test]
    fn test_per_gaussian_uncertainty_dim_mismatch() {
        assert!(per_gaussian_position_uncertainty(&[0.0f32; 5], 2, 1).is_err());
    }

    // --- variance_to_confidence ---

    #[test]
    fn test_variance_to_confidence_zero() {
        let conf = variance_to_confidence(&[0.0f32]);
        assert!(approx(conf[0], 1.0, 1e-6));
    }

    #[test]
    fn test_variance_to_confidence_large() {
        let conf = variance_to_confidence(&[100.0f32]);
        assert!(conf[0] < 0.001);
    }

    // --- decompose_uncertainty ---

    #[test]
    fn test_decompose_uncertainty_total_approx() {
        let samples = vec![vec![0.0f32, 0.0], vec![2.0f32, 2.0], vec![1.0f32, 1.0]];
        let model_mean = vec![1.0f32, 1.0];
        let decomp = decompose_uncertainty(&samples, &model_mean).expect("decompose failed");
        for d in &decomp {
            // total >= aleatoric + epistemic (may have floating point rounding)
            assert!(approx(d.total, d.aleatoric + d.epistemic, 1e-5));
        }
    }

    #[test]
    fn test_decompose_uncertainty_empty_error() {
        assert!(decompose_uncertainty(&[], &[1.0f32]).is_err());
    }

    #[test]
    fn test_decompose_uncertainty_point_samples_are_fully_epistemic() {
        // Regression: point predictions carry no observation-noise estimate, so
        // no aleatoric share may be invented (the earlier implementation
        // reported a fixed ~0.64 * total via a mean-absolute-deviation proxy).
        let samples = vec![vec![0.0f32], vec![2.0f32], vec![1.0f32]];
        let model_mean = vec![1.0f32];
        let decomp = decompose_uncertainty(&samples, &model_mean).expect("decompose failed");
        assert_eq!(decomp.len(), 1);
        // total = ((0-1)^2 + (2-1)^2 + (1-1)^2) / 3 = 2/3
        assert!(approx(decomp[0].total, 2.0 / 3.0, 1e-5));
        assert_eq!(decomp[0].aleatoric, 0.0);
        assert!(approx(decomp[0].epistemic, decomp[0].total, 1e-6));
    }

    // --- decompose_uncertainty_with_variances ---

    #[test]
    fn test_decompose_with_variances_pure_epistemic() {
        // Disagreeing passes with zero predicted noise → all epistemic.
        let means = vec![vec![0.0f32], vec![2.0f32]];
        let variances = vec![vec![0.0f32], vec![0.0f32]];
        let model_mean = vec![1.0f32];
        let decomp = decompose_uncertainty_with_variances(&means, &variances, &model_mean)
            .expect("decompose failed");
        assert_eq!(decomp[0].aleatoric, 0.0);
        assert!(approx(decomp[0].epistemic, 1.0, 1e-5));
        assert!(approx(decomp[0].total, 1.0, 1e-5));
    }

    #[test]
    fn test_decompose_with_variances_pure_aleatoric() {
        // Identical passes → no model disagreement; all uncertainty is data noise.
        let means = vec![vec![1.0f32], vec![1.0f32], vec![1.0f32]];
        let variances = vec![vec![0.25f32], vec![0.75f32], vec![0.5f32]];
        let model_mean = vec![1.0f32];
        let decomp = decompose_uncertainty_with_variances(&means, &variances, &model_mean)
            .expect("decompose failed");
        assert!(approx(decomp[0].aleatoric, 0.5, 1e-5));
        assert_eq!(decomp[0].epistemic, 0.0);
        assert!(approx(decomp[0].total, 0.5, 1e-5));
    }

    #[test]
    fn test_decompose_with_variances_law_of_total_variance() {
        let means = vec![vec![0.0f32, 1.0], vec![2.0f32, 1.0]];
        let variances = vec![vec![0.1f32, 0.4], vec![0.3f32, 0.6]];
        let model_mean = vec![1.0f32, 1.0];
        let decomp = decompose_uncertainty_with_variances(&means, &variances, &model_mean)
            .expect("decompose failed");
        assert_eq!(decomp.len(), 2);
        // dim 0: aleatoric = (0.1 + 0.3)/2 = 0.2, epistemic = ((0-1)^2 + (2-1)^2)/2 = 1
        assert!(approx(decomp[0].aleatoric, 0.2, 1e-5));
        assert!(approx(decomp[0].epistemic, 1.0, 1e-5));
        // dim 1: aleatoric = (0.4 + 0.6)/2 = 0.5, epistemic = 0
        assert!(approx(decomp[1].aleatoric, 0.5, 1e-5));
        assert!(approx(decomp[1].epistemic, 0.0, 1e-6));
        for d in &decomp {
            assert!(approx(d.total, d.aleatoric + d.epistemic, 1e-6));
        }
    }

    #[test]
    fn test_decompose_with_variances_empty_error() {
        assert!(decompose_uncertainty_with_variances(&[], &[], &[1.0f32]).is_err());
    }

    #[test]
    fn test_decompose_with_variances_count_mismatch_error() {
        let means = vec![vec![1.0f32], vec![2.0f32]];
        let variances = vec![vec![0.1f32]];
        assert!(decompose_uncertainty_with_variances(&means, &variances, &[1.0f32]).is_err());
    }

    #[test]
    fn test_decompose_with_variances_row_dim_mismatch_error() {
        let means = vec![vec![1.0f32, 2.0]];
        let variances = vec![vec![0.1f32, 0.2]];
        // model_mean has dim 1, rows have dim 2
        assert!(decompose_uncertainty_with_variances(&means, &variances, &[1.0f32]).is_err());
    }

    #[test]
    fn test_decompose_with_variances_negative_variance_error() {
        let means = vec![vec![1.0f32]];
        let variances = vec![vec![-0.5f32]];
        assert!(decompose_uncertainty_with_variances(&means, &variances, &[1.0f32]).is_err());
    }

    #[test]
    fn test_decompose_with_variances_non_finite_variance_error() {
        let means = vec![vec![1.0f32]];
        let variances = vec![vec![f32::NAN]];
        assert!(decompose_uncertainty_with_variances(&means, &variances, &[1.0f32]).is_err());
    }

    // --- high_uncertainty_indices ---

    #[test]
    fn test_high_uncertainty_indices_threshold_zero_all() {
        let vars = vec![0.1f32, 0.2, 0.3];
        let idxs = high_uncertainty_indices(&vars, 0.0);
        assert_eq!(idxs, vec![0, 1, 2]);
    }

    #[test]
    fn test_high_uncertainty_indices_threshold_large_none() {
        let vars = vec![0.1f32, 0.2, 0.3];
        let idxs = high_uncertainty_indices(&vars, 1.0);
        assert!(idxs.is_empty());
    }

    #[test]
    fn test_high_uncertainty_indices_partial() {
        let vars = vec![0.1f32, 0.9, 0.3, 0.8];
        let idxs = high_uncertainty_indices(&vars, 0.5);
        assert_eq!(idxs, vec![1, 3]);
    }

    // --- uncertainty_weighted_loss ---

    #[test]
    fn test_uncertainty_weighted_loss_zero_variance() {
        let losses = vec![1.0f32, 2.0, 3.0];
        let vars = vec![0.0f32; 3];
        let wl = uncertainty_weighted_loss(&losses, &vars).expect("weighted_loss failed");
        // All weights equal 1, so wl = (1+2+3)/3 = 2
        assert!(approx(wl, 2.0, 1e-5));
    }

    #[test]
    fn test_uncertainty_weighted_loss_high_variance_low_weight() {
        // One high-loss sample has huge variance → its weight is tiny
        let losses = vec![0.0f32, 100.0];
        let vars = vec![0.0f32, 1e6];
        let wl = uncertainty_weighted_loss(&losses, &vars).expect("weighted_loss failed");
        // The second sample's weight ≈ 1e-6, first ≈ 1, so result ≈ 0
        assert!(wl < 0.01);
    }

    #[test]
    fn test_uncertainty_weighted_loss_empty_error() {
        assert!(uncertainty_weighted_loss(&[], &[]).is_err());
    }

    // --- aggregate_region_uncertainty ---

    #[test]
    fn test_aggregate_region_uncertainty_empty_region() {
        let variances = vec![0.5f32, 1.0, 0.2];
        let masks = vec![vec![false, false, false]];
        let result =
            aggregate_region_uncertainty(&variances, &masks, 0.3).expect("aggregate failed");
        assert_eq!(result.len(), 1);
        assert!(approx(result[0].mean_variance, 0.0, 1e-6));
    }

    #[test]
    fn test_aggregate_region_uncertainty_single_point() {
        let variances = vec![0.8f32, 0.1];
        let masks = vec![vec![true, false]];
        let result =
            aggregate_region_uncertainty(&variances, &masks, 0.5).expect("aggregate failed");
        assert_eq!(result.len(), 1);
        assert!(approx(result[0].mean_variance, 0.8, 1e-6));
        assert!(approx(result[0].max_variance, 0.8, 1e-6));
        assert!(approx(result[0].high_uncertainty_fraction, 1.0, 1e-6));
    }

    #[test]
    fn test_aggregate_region_uncertainty_mask_mismatch_error() {
        let variances = vec![0.5f32; 3];
        let masks = vec![vec![true, false]]; // wrong length
        assert!(aggregate_region_uncertainty(&variances, &masks, 0.3).is_err());
    }
}
