//! Noise prediction analysis tools for diffusion model training diagnostics.
//!
//! Provides utilities for comparing predicted noise vs. true noise, computing
//! signal-to-noise ratio (SNR) maps, analysing prediction errors spatially and
//! temporally, and detecting common failure modes such as over-smoothing or
//! colour-shift artefacts.
//!
//! # Diffusion model background
//!
//! During training, a diffusion model is presented with a noisy image
//!
//! ```text
//! x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
//! ```
//!
//! and is asked to predict the noise `epsilon_pred`.  The true noise `epsilon`
//! is known, so we can compute rich diagnostic statistics that go beyond a
//! simple scalar loss.
//!
//! Images are flat `Vec<f32>` in HWC order (height × width × channels).
//! Values are expected to lie in `[-1, 1]` or `[0, 1]`, but the functions here
//! make no hard assumption about that range.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during noise prediction analysis.
#[derive(Debug, Error, PartialEq)]
pub enum NoiseAnalysisError {
    /// The supplied image buffer is empty.
    #[error("Empty image buffer")]
    EmptyBuffer,

    /// The predicted-noise buffer and true-noise buffer have different lengths.
    #[error("Buffer length mismatch: predicted has {pred_len}, true has {true_len}")]
    LengthMismatch { pred_len: usize, true_len: usize },

    /// The buffer length is not divisible by the number of channels.
    #[error("Invalid image dimensions: {len} values is not divisible by {channels} channels")]
    InvalidDimensions { len: usize, channels: usize },

    /// The timestep index is out of the valid range.
    #[error("Timestep {t} out of range [0, {max_t}]")]
    InvalidTimestep { t: usize, max_t: usize },

    /// Not enough history entries to satisfy the window size.
    #[error("Insufficient history: need {needed} steps, have {got}")]
    InsufficientHistory { needed: usize, got: usize },
}

// ---------------------------------------------------------------------------
// Core statistics
// ---------------------------------------------------------------------------

/// Basic statistics describing how well a noise prediction matches the truth.
#[derive(Debug, Clone)]
pub struct PredictionStats {
    /// Mean squared error between `epsilon_pred` and `epsilon`.
    pub mse: f32,
    /// Mean absolute error between `epsilon_pred` and `epsilon`.
    pub mae: f32,
    /// Root mean squared error (sqrt of `mse`).
    pub rmse: f32,
    /// Maximum absolute per-element error.
    pub max_error: f32,
    /// Cosine similarity between `epsilon_pred` and `epsilon` (in `[-1, 1]`).
    pub cosine_sim: f32,
    /// Number of elements analysed.
    pub n_values: usize,
}

/// SNR (signal-to-noise ratio) information for a single diffusion timestep.
#[derive(Debug, Clone)]
pub struct TimestepSnr {
    /// The timestep index.
    pub timestep: usize,
    /// Cumulative noise schedule value ᾱ_t.
    pub alpha_bar: f32,
    /// Linear SNR = ᾱ_t / (1 − ᾱ_t).
    pub snr: f32,
    /// SNR in decibels: 10 · log₁₀(snr).
    pub snr_db: f32,
    /// Min-SNR loss weight: `snr_clamp / max(snr, ε)` (clamped at `snr_clamp`).
    pub loss_weight: f32,
}

/// Per-pixel mean absolute error map.
#[derive(Debug, Clone)]
pub struct SpatialErrorMap {
    /// Per-pixel MAE (averaged across channels), length = `height × width`.
    pub errors: Vec<f32>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Number of colour channels.
    pub channels: usize,
    /// Spatial mean of `errors`.
    pub mean_error: f32,
    /// Maximum value in `errors`.
    pub max_error: f32,
    /// Population standard deviation of `errors`.
    pub error_std: f32,
}

impl SpatialErrorMap {
    /// Fraction of pixels (in `[0, 1]`) whose per-pixel error strictly
    /// exceeds `threshold`. Returns `0.0` for an empty map.
    ///
    /// Typically called with [`NoiseAnalysisConfig::error_threshold`] to
    /// answer "what fraction of the image is badly predicted?", a question
    /// `mean_error` alone cannot answer (a few very bad pixels and many
    /// mediocre ones can produce the same mean as many uniformly-bad pixels).
    pub fn fraction_above(&self, threshold: f32) -> f32 {
        if self.errors.is_empty() {
            return 0.0;
        }
        let count = self.errors.iter().filter(|&&e| e > threshold).count();
        count as f32 / self.errors.len() as f32
    }
}

/// Configuration knobs for the analysis pipeline.
#[derive(Debug, Clone)]
pub struct NoiseAnalysisConfig {
    /// Number of colour channels (default 3).
    pub channels: usize,
    /// Total number of training timesteps (default 1000).
    pub max_timesteps: usize,
    /// Clamping value used for the Min-SNR loss weight (default 5.0).
    pub snr_clamp: f32,
    /// Threshold above which a per-pixel error is considered "bad" (default 0.1).
    pub error_threshold: f32,
}

impl Default for NoiseAnalysisConfig {
    fn default() -> Self {
        Self {
            channels: 3,
            max_timesteps: 1000,
            snr_clamp: 5.0,
            error_threshold: 0.1,
        }
    }
}

/// Full analysis report for a single noise prediction.
#[derive(Debug, Clone)]
pub struct NoisePredictionReport {
    /// Timestep at which the prediction was made.
    pub timestep: usize,
    /// Scalar prediction statistics.
    pub stats: PredictionStats,
    /// SNR information for this timestep.
    pub snr_info: TimestepSnr,
    /// Spatial error map.
    pub spatial_error: SpatialErrorMap,
    /// Detected failure mode, if any.
    pub failure_mode: Option<FailureMode>,
    /// Fraction of pixels (in `[0, 1]`) whose per-pixel error exceeds
    /// [`NoiseAnalysisConfig::error_threshold`]. See
    /// [`SpatialErrorMap::fraction_above`].
    pub fraction_pixels_above_threshold: f32,
}

/// Common failure modes detected in diffusion noise predictions.
#[derive(Debug, Clone, PartialEq)]
pub enum FailureMode {
    /// Predicted noise has lower variance than the true noise (blurry predictions).
    Oversmoothing,
    /// Prediction error is concentrated at high spatial frequencies (ringing).
    HighFreqArtifact,
    /// Systematic per-channel mean bias between prediction and truth.
    ColorShift,
    /// MSE exceeds 2 × the expected value, indicating training divergence.
    Diverged,
}

/// Running history of prediction quality across training steps.
#[derive(Debug, Clone)]
pub struct PredictionHistory {
    /// Training step indices.
    pub steps: Vec<usize>,
    /// MSE recorded at each training step.
    pub mse_history: Vec<f32>,
    /// Cosine similarity recorded at each training step.
    pub cosine_history: Vec<f32>,
    /// Diffusion timestep used at each training step.
    pub timestep_history: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Helper functions (also part of the public API)
// ---------------------------------------------------------------------------

/// Compute the population standard deviation of a slice.
///
/// Returns `0.0` for a slice with fewer than two elements.
pub fn vec_std(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let mean = v.iter().sum::<f32>() / n as f32;
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    variance.sqrt()
}

/// Compute the per-channel mean of a flat HWC buffer.
///
/// Returns a `Vec` of length `channels`.  An empty buffer or `channels == 0`
/// returns an empty `Vec`.
pub fn channel_means(v: &[f32], channels: usize) -> Vec<f32> {
    if v.is_empty() || channels == 0 {
        return Vec::new();
    }
    let mut sums = vec![0.0_f32; channels];
    let mut counts = vec![0usize; channels];
    for (i, &val) in v.iter().enumerate() {
        let c = i % channels;
        sums[c] += val;
        counts[c] += 1;
    }
    sums.iter()
        .zip(counts.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f32 } else { 0.0 })
        .collect()
}

// ---------------------------------------------------------------------------
// Core analysis functions
// ---------------------------------------------------------------------------

/// Compute scalar prediction statistics comparing `pred` to `true_noise`.
///
/// Both slices must have equal, non-zero length.
pub fn compute_prediction_stats(
    pred: &[f32],
    true_noise: &[f32],
) -> Result<PredictionStats, NoiseAnalysisError> {
    let n = pred.len();
    if n == 0 {
        return Err(NoiseAnalysisError::EmptyBuffer);
    }
    if n != true_noise.len() {
        return Err(NoiseAnalysisError::LengthMismatch {
            pred_len: n,
            true_len: true_noise.len(),
        });
    }

    let mut sum_sq = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    let mut max_err = 0.0_f32;
    let mut dot = 0.0_f32;
    let mut norm_pred_sq = 0.0_f32;
    let mut norm_true_sq = 0.0_f32;

    for (&p, &t) in pred.iter().zip(true_noise.iter()) {
        let diff = p - t;
        sum_sq += diff * diff;
        sum_abs += diff.abs();
        if diff.abs() > max_err {
            max_err = diff.abs();
        }
        dot += p * t;
        norm_pred_sq += p * p;
        norm_true_sq += t * t;
    }

    let nf = n as f32;
    let mse = sum_sq / nf;
    let mae = sum_abs / nf;
    let rmse = mse.sqrt();

    let norm_pred = norm_pred_sq.sqrt();
    let norm_true = norm_true_sq.sqrt();
    let cosine_sim = if norm_pred < 1e-8 || norm_true < 1e-8 {
        0.0
    } else {
        (dot / (norm_pred * norm_true)).clamp(-1.0, 1.0)
    };

    Ok(PredictionStats {
        mse,
        mae,
        rmse,
        max_error: max_err,
        cosine_sim,
        n_values: n,
    })
}

/// Compute the cumulative noise-schedule value ᾱ_t.
///
/// Supported schedule strings:
/// - `"linear"`: fast quadratic approximation `(1 - t/T)²`
/// - `"cosine"` (default): `cos²(((t/T + 0.008) / 1.008) · π/2)`
///
/// Any unknown schedule string falls back to cosine.
pub fn compute_alpha_bar(timestep: usize, max_timesteps: usize, schedule: &str) -> f32 {
    let max_t = max_timesteps.max(1);
    let t = timestep.min(max_t) as f32;
    let big_t = max_t as f32;

    match schedule {
        "linear" => {
            let frac = 1.0 - t / big_t;
            (frac * frac).clamp(0.0, 1.0)
        }
        _ => {
            // cosine schedule (Nichol & Dhariwal 2021)
            let frac = (t / big_t + 0.008) / 1.008;
            let angle = frac * std::f32::consts::PI * 0.5;
            angle.cos().powi(2).clamp(0.0, 1.0)
        }
    }
}

/// Compute SNR information for a specific timestep.
///
/// `snr_clamp` sets the Min-SNR loss-weight denominator (see
/// [`NoiseAnalysisConfig::snr_clamp`]; the conventional default is `5.0`).
pub fn compute_timestep_snr(
    timestep: usize,
    max_timesteps: usize,
    schedule: &str,
    snr_clamp: f32,
) -> Result<TimestepSnr, NoiseAnalysisError> {
    if max_timesteps == 0 || timestep > max_timesteps {
        return Err(NoiseAnalysisError::InvalidTimestep {
            t: timestep,
            max_t: max_timesteps,
        });
    }

    let alpha_bar = compute_alpha_bar(timestep, max_timesteps, schedule);
    let snr = alpha_bar / (1.0 - alpha_bar).max(1e-8);

    let snr_db = if snr <= 0.0 {
        f32::NEG_INFINITY
    } else {
        10.0 * snr.log10()
    };

    // Min-SNR weighting: clamp(snr, snr_clamp) / snr
    // Equivalently: snr_clamp / max(snr, epsilon) when snr >= snr_clamp, else 1.0
    let loss_weight = (snr_clamp / snr.max(1e-8)).min(1.0);

    Ok(TimestepSnr {
        timestep,
        alpha_bar,
        snr,
        snr_db,
        loss_weight,
    })
}

/// Compute a per-pixel MAE spatial error map.
///
/// `pred` and `true_noise` must be flat HWC buffers of length
/// `height × width × channels`.
pub fn compute_spatial_error_map(
    pred: &[f32],
    true_noise: &[f32],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<SpatialErrorMap, NoiseAnalysisError> {
    let expected_len = height * width * channels;
    if expected_len == 0 || pred.is_empty() {
        return Err(NoiseAnalysisError::EmptyBuffer);
    }
    if pred.len() != true_noise.len() {
        return Err(NoiseAnalysisError::LengthMismatch {
            pred_len: pred.len(),
            true_len: true_noise.len(),
        });
    }
    // Require an exact match against the declared width/height/channels
    // rather than just divisibility by `channels`: a buffer shorter than
    // `expected_len` (but still a multiple of `channels`) would otherwise
    // silently read as having trailing pixels with zero error — a
    // perfect-looking but wrong report — instead of being rejected.
    if pred.len() != expected_len {
        return Err(NoiseAnalysisError::InvalidDimensions {
            len: pred.len(),
            channels,
        });
    }

    let n_pixels = height * width;
    let mut errors = Vec::with_capacity(n_pixels);

    for p in 0..n_pixels {
        let base = p * channels;
        let mut abs_sum = 0.0_f32;
        for c in 0..channels {
            // Safe: `pred.len() == true_noise.len() == expected_len ==
            // n_pixels * channels`, and `base + c < n_pixels * channels`
            // for all `p < n_pixels`, `c < channels`.
            abs_sum += (pred[base + c] - true_noise[base + c]).abs();
        }
        errors.push(abs_sum / channels as f32);
    }

    let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
    let max_error = errors
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(0.0);
    let error_std = vec_std(&errors);

    Ok(SpatialErrorMap {
        errors,
        width,
        height,
        channels,
        mean_error,
        max_error,
        error_std,
    })
}

/// Ratio of "energy a 3-tap box blur removes" to "total error energy" above
/// which [`detect_failure_mode`] reports [`FailureMode::HighFreqArtifact`].
///
/// Calibrated empirically (see the comment at the call site) so that
/// genuinely random, i.i.d. per-pixel prediction error — which is *not*
/// itself evidence of ringing — stays comfortably below this line even in
/// its observed worst case, while a structured/checkerboard ringing pattern
/// scores well above it.
const HIGH_FREQ_ENERGY_RATIO_THRESHOLD: f32 = 0.7;

/// Minimum pixel count (`width * height`) required before
/// [`detect_failure_mode`] attempts [`FailureMode::HighFreqArtifact`]
/// detection. Below this, the high-frequency-energy ratio for pure i.i.d.
/// noise has enough sample variance to occasionally exceed
/// [`HIGH_FREQ_ENERGY_RATIO_THRESHOLD`] by chance (observed up to ~0.87 at
/// 10 pixels), so the check is skipped rather than risk a false positive on
/// small images.
const HIGH_FREQ_MIN_PIXELS: usize = 64;

/// Detect whether the noise prediction exhibits a common failure mode.
///
/// Priority order: `Diverged` → `Oversmoothing` → `HighFreqArtifact` →
/// `ColorShift` → `None`.
///
/// `channels` is used for `ColorShift` and `HighFreqArtifact` detection;
/// `ColorShift` defaults to 3 channels if 0. `width`/`height` are the spatial
/// dimensions of `pred`/`true_noise` (needed only for `HighFreqArtifact`,
/// which requires 2D structure); that check is silently skipped if
/// `pred.len()` or `true_noise.len()` differs from `width * height *
/// channels`, or if there are fewer than `HIGH_FREQ_MIN_PIXELS` pixels.
/// This function never panics, regardless of how `pred`/`true_noise`/
/// `width`/`height`/`channels` relate to each other.
pub fn detect_failure_mode(
    pred: &[f32],
    true_noise: &[f32],
    stats: &PredictionStats,
    channels: usize,
    width: usize,
    height: usize,
) -> Option<FailureMode> {
    // 1. Diverged: MSE > 2.0
    if stats.mse > 2.0 {
        return Some(FailureMode::Diverged);
    }

    // 2. Oversmoothing: std(pred) < 0.5 × std(true)
    let std_pred = vec_std(pred);
    let std_true = vec_std(true_noise);
    if std_pred < 0.5 * std_true {
        return Some(FailureMode::Oversmoothing);
    }

    // 3. HighFreqArtifact: build a per-pixel error map and compare its
    // energy against a box-blurred version of itself (reusing the
    // separable box blur from `latent_blend::lb_smooth_mask`). Ringing
    // concentrates error energy at high spatial frequencies, so blurring
    // cancels most of it out; a smooth or uniform error field survives
    // blurring almost unchanged. Only runs when the buffers cleanly
    // reshape into `width * height` pixels of `channels` values each, and
    // when there are enough pixels for the ratio below to be a reliable
    // statistic (see the threshold constants' doc for the calibration).
    let n_pixels = width * height;
    if channels > 0
        && n_pixels >= HIGH_FREQ_MIN_PIXELS
        && pred.len() == n_pixels * channels
        && true_noise.len() == n_pixels * channels
    {
        let mut error_map = Vec::with_capacity(n_pixels);
        for p in 0..n_pixels {
            let base = p * channels;
            let mut abs_sum = 0.0_f32;
            for c in 0..channels {
                abs_sum += (pred[base + c] - true_noise[base + c]).abs();
            }
            error_map.push(abs_sum / channels as f32);
        }
        if let Ok(blurred) = crate::latent_blend::lb_smooth_mask(&error_map, height, width, 1) {
            let error_energy = error_map.iter().sum::<f32>() / n_pixels as f32;
            let high_freq_energy = error_map
                .iter()
                .zip(blurred.iter())
                .map(|(&e, &b)| (e - b).abs())
                .sum::<f32>()
                / n_pixels as f32;
            // The energy a 3-tap box blur removes from *genuinely random,
            // i.i.d.* per-pixel error (healthy prediction noise) already
            // averages ~47% of the total error energy at 1 channel and
            // ~27% at 3 channels (verified empirically: a linear
            // combination of independent samples has std sqrt(2/3) times
            // the source std, so even pure noise "loses" a large,
            // non-trivial fraction of its energy to blurring — this is not
            // by itself evidence of structure). Genuine ringing/checkerboard
            // patterns score far higher (~1.3, since they are a
            // near-deterministic alternation rather than independent
            // samples). The threshold and minimum pixel count below are
            // chosen with a comfortable margin above the empirically
            // observed noise ceiling (max ~0.59 at n=64 over 200 trials, up
            // to ~0.87 for tiny n<20 samples, which is why a minimum pixel
            // count is enforced) while still firing clearly on structured
            // ringing.
            if error_energy > 1e-6
                && high_freq_energy > HIGH_FREQ_ENERGY_RATIO_THRESHOLD * error_energy
            {
                return Some(FailureMode::HighFreqArtifact);
            }
        }
    }

    // 4. ColorShift: per-channel mean bias > 0.1
    let ch = channels.max(1);
    let pred_means = channel_means(pred, ch);
    let true_means = channel_means(true_noise, ch);
    for (pm, tm) in pred_means.iter().zip(true_means.iter()) {
        if (pm - tm).abs() > 0.1 {
            return Some(FailureMode::ColorShift);
        }
    }

    None
}

/// Run the full noise prediction analysis pipeline.
///
/// Validates inputs, computes all statistics, and assembles a
/// [`NoisePredictionReport`].
pub fn analyze_noise_prediction(
    pred: &[f32],
    true_noise: &[f32],
    timestep: usize,
    width: usize,
    height: usize,
    config: &NoiseAnalysisConfig,
) -> Result<NoisePredictionReport, NoiseAnalysisError> {
    if pred.is_empty() {
        return Err(NoiseAnalysisError::EmptyBuffer);
    }
    if pred.len() != true_noise.len() {
        return Err(NoiseAnalysisError::LengthMismatch {
            pred_len: pred.len(),
            true_len: true_noise.len(),
        });
    }

    let stats = compute_prediction_stats(pred, true_noise)?;
    let snr_info =
        compute_timestep_snr(timestep, config.max_timesteps, "cosine", config.snr_clamp)?;
    let spatial_error =
        compute_spatial_error_map(pred, true_noise, width, height, config.channels)?;
    let failure_mode =
        detect_failure_mode(pred, true_noise, &stats, config.channels, width, height);
    let fraction_pixels_above_threshold = spatial_error.fraction_above(config.error_threshold);

    Ok(NoisePredictionReport {
        timestep,
        stats,
        snr_info,
        spatial_error,
        failure_mode,
        fraction_pixels_above_threshold,
    })
}

/// Compute an SNR-weighted noise prediction loss.
///
/// `loss = mse × loss_weight` where `loss_weight` comes from [`compute_timestep_snr`]
/// using the given `snr_clamp` (see [`NoiseAnalysisConfig::snr_clamp`]).
pub fn weighted_noise_loss(
    pred: &[f32],
    true_noise: &[f32],
    timestep: usize,
    max_timesteps: usize,
    schedule: &str,
    snr_clamp: f32,
) -> Result<f32, NoiseAnalysisError> {
    let stats = compute_prediction_stats(pred, true_noise)?;
    let snr = compute_timestep_snr(timestep, max_timesteps, schedule, snr_clamp)?;
    Ok(stats.mse * snr.loss_weight)
}

/// Return an interpolated percentile value from a [`SpatialErrorMap`].
///
/// `percentile` must be in `[0.0, 100.0]`.  Values outside this range are
/// clamped before use.
pub fn error_map_percentile(error_map: &SpatialErrorMap, percentile: f32) -> f32 {
    if error_map.errors.is_empty() {
        return 0.0;
    }

    let mut sorted = error_map.errors.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p = percentile.clamp(0.0, 100.0);
    let n = sorted.len();

    if n == 1 {
        return sorted[0];
    }

    // Linear interpolation between adjacent sorted values.
    let idx_f = p / 100.0 * (n - 1) as f32;
    let lo = idx_f.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx_f - lo as f32;

    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Compute the expected noise variance (1 − ᾱ_t) at each timestep.
///
/// The returned `Vec` has length `max_timesteps`, indexed by timestep.
pub fn compute_noise_floor(max_timesteps: usize, schedule: &str) -> Vec<f32> {
    (0..max_timesteps)
        .map(|t| 1.0 - compute_alpha_bar(t, max_timesteps, schedule))
        .collect()
}

/// Compare two predictions against the same ground-truth noise, returning
/// `(stats_a, stats_b)`.
pub fn compare_predictions(
    pred_a: &[f32],
    pred_b: &[f32],
    true_noise: &[f32],
) -> Result<(PredictionStats, PredictionStats), NoiseAnalysisError> {
    let stats_a = compute_prediction_stats(pred_a, true_noise)?;
    let stats_b = compute_prediction_stats(pred_b, true_noise)?;
    Ok((stats_a, stats_b))
}

/// Produce a concise human-readable summary of a [`NoisePredictionReport`].
pub fn format_noise_report(report: &NoisePredictionReport) -> String {
    let failure_str = match &report.failure_mode {
        None => "None".to_string(),
        Some(fm) => format!("{:?}", fm),
    };
    format!(
        "Step t{timestep}: MSE={mse:.4}, CosSim={cos:.4}, Failure={failure}",
        timestep = report.timestep,
        mse = report.stats.mse,
        cos = report.stats.cosine_sim,
        failure = failure_str,
    )
}

// ---------------------------------------------------------------------------
// PredictionHistory
// ---------------------------------------------------------------------------

impl PredictionHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            mse_history: Vec::new(),
            cosine_history: Vec::new(),
            timestep_history: Vec::new(),
        }
    }

    /// Append a new observation to the history.
    pub fn record(&mut self, training_step: usize, mse: f32, cosine_sim: f32, timestep: usize) {
        self.steps.push(training_step);
        self.mse_history.push(mse);
        self.cosine_history.push(cosine_sim);
        self.timestep_history.push(timestep);
    }

    /// Compute the linear-regression slope of MSE over the last `window` steps.
    ///
    /// A negative slope indicates the loss is improving (decreasing).
    /// Returns [`NoiseAnalysisError::InsufficientHistory`] if there are fewer
    /// than `window` recorded steps.
    pub fn recent_trend(&self, window: usize) -> Result<f32, NoiseAnalysisError> {
        let n = self.mse_history.len();
        if n < window || window == 0 {
            return Err(NoiseAnalysisError::InsufficientHistory {
                needed: window,
                got: n,
            });
        }
        let slice = &self.mse_history[n - window..];
        let slope = linear_regression_slope(slice);
        Ok(slope)
    }

    /// Return the minimum MSE ever recorded, or `None` if history is empty.
    pub fn best_mse(&self) -> Option<f32> {
        self.mse_history.iter().cloned().reduce(f32::min)
    }

    /// Return `true` if the MSE trend over the last `window` steps is negative
    /// (i.e. the model is improving).
    pub fn is_improving(&self, window: usize) -> Result<bool, NoiseAnalysisError> {
        let slope = self.recent_trend(window)?;
        Ok(slope < 0.0)
    }
}

impl Default for PredictionHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute the ordinary-least-squares slope of a sequence of y-values,
/// using equally-spaced integer x-values (0, 1, 2, …).
///
/// Returns `0.0` for fewer than two points.
fn linear_regression_slope(y: &[f32]) -> f32 {
    let n = y.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    // x̄ for x = 0..n-1
    let x_mean = (nf - 1.0) * 0.5;
    let y_mean = y.iter().sum::<f32>() / nf;

    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for (i, &yi) in y.iter().enumerate() {
        let xi = i as f32 - x_mean;
        num += xi * (yi - y_mean);
        den += xi * xi;
    }
    if den.abs() < 1e-12 {
        0.0
    } else {
        num / den
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    // Helper: make a flat HWC buffer of uniform value
    fn uniform_buf(len: usize, val: f32) -> Vec<f32> {
        vec![val; len]
    }

    // Helper: simple linearly-spaced buffer
    fn linspace_buf(len: usize, start: f32, end: f32) -> Vec<f32> {
        if len == 0 {
            return Vec::new();
        }
        if len == 1 {
            return vec![start];
        }
        (0..len)
            .map(|i| start + (end - start) * i as f32 / (len - 1) as f32)
            .collect()
    }

    // -----------------------------------------------------------------------
    // compute_prediction_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_perfect_prediction() {
        let v = linspace_buf(12, -1.0, 1.0);
        let stats = compute_prediction_stats(&v, &v).expect("perfect pred");
        assert!(stats.mse < EPSILON);
        assert!(stats.mae < EPSILON);
        assert!(stats.rmse < EPSILON);
        assert!(stats.max_error < EPSILON);
        assert!((stats.cosine_sim - 1.0).abs() < EPSILON);
        assert_eq!(stats.n_values, 12);
    }

    #[test]
    fn test_stats_zero_prediction() {
        let pred = uniform_buf(6, 0.0);
        let truth = uniform_buf(6, 1.0);
        let stats = compute_prediction_stats(&pred, &truth).expect("zero pred");
        assert!((stats.mse - 1.0).abs() < EPSILON);
        assert!((stats.mae - 1.0).abs() < EPSILON);
        assert!((stats.cosine_sim).abs() < EPSILON);
    }

    #[test]
    fn test_stats_opposite_prediction() {
        let truth = uniform_buf(4, 1.0);
        let pred = uniform_buf(4, -1.0);
        let stats = compute_prediction_stats(&pred, &truth).expect("opposite");
        // Each element differs by 2
        assert!((stats.mse - 4.0).abs() < EPSILON);
        assert!((stats.mae - 2.0).abs() < EPSILON);
        assert!((stats.max_error - 2.0).abs() < EPSILON);
        assert!((stats.cosine_sim + 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_stats_empty_buffer_error() {
        let r = compute_prediction_stats(&[], &[]);
        assert!(matches!(r, Err(NoiseAnalysisError::EmptyBuffer)));
    }

    #[test]
    fn test_stats_length_mismatch_error() {
        let pred = vec![0.0_f32; 4];
        let truth = vec![0.0_f32; 6];
        let r = compute_prediction_stats(&pred, &truth);
        assert!(matches!(r, Err(NoiseAnalysisError::LengthMismatch { .. })));
    }

    #[test]
    fn test_stats_cosine_zero_pred_returns_zero() {
        // If both pred and truth are all zeros, cosine should be 0.0 (not NaN)
        let r = compute_prediction_stats(&[0.0; 8], &[0.0; 8]).expect("zero-zero");
        assert!((r.cosine_sim).abs() < EPSILON);
    }

    #[test]
    fn test_stats_cosine_only_true_zero() {
        let pred = vec![1.0_f32; 4];
        let truth = vec![0.0_f32; 4];
        let stats = compute_prediction_stats(&pred, &truth).expect("only truth zero");
        assert!((stats.cosine_sim).abs() < EPSILON);
    }

    #[test]
    fn test_stats_n_values() {
        let v = vec![0.5_f32; 30];
        let stats = compute_prediction_stats(&v, &v).expect("same");
        assert_eq!(stats.n_values, 30);
    }

    // -----------------------------------------------------------------------
    // compute_alpha_bar
    // -----------------------------------------------------------------------

    #[test]
    fn test_alpha_bar_cosine_t0_near_one() {
        let ab = compute_alpha_bar(0, 1000, "cosine");
        assert!(ab > 0.99, "alpha_bar at t=0 should be ~1.0, got {ab}");
    }

    #[test]
    fn test_alpha_bar_cosine_tmax_near_zero() {
        let ab = compute_alpha_bar(1000, 1000, "cosine");
        assert!(ab < 0.01, "alpha_bar at t=T should be ~0.0, got {ab}");
    }

    #[test]
    fn test_alpha_bar_linear_t0_near_one() {
        let ab = compute_alpha_bar(0, 1000, "linear");
        assert!(
            (ab - 1.0).abs() < EPSILON,
            "linear alpha_bar at t=0 should be 1.0, got {ab}"
        );
    }

    #[test]
    fn test_alpha_bar_linear_tmax_near_zero() {
        let ab = compute_alpha_bar(1000, 1000, "linear");
        assert!(
            ab < EPSILON,
            "linear alpha_bar at t=T should be ~0.0, got {ab}"
        );
    }

    #[test]
    fn test_alpha_bar_cosine_monotone_decreasing() {
        let t = 1000_usize;
        let prev = compute_alpha_bar(499, t, "cosine");
        let next = compute_alpha_bar(500, t, "cosine");
        assert!(next < prev, "alpha_bar must decrease with timestep");
    }

    #[test]
    fn test_alpha_bar_linear_monotone_decreasing() {
        let t = 1000_usize;
        let prev = compute_alpha_bar(499, t, "linear");
        let next = compute_alpha_bar(500, t, "linear");
        assert!(next < prev, "alpha_bar must decrease with timestep");
    }

    #[test]
    fn test_alpha_bar_unknown_schedule_falls_back_to_cosine() {
        let ab_unk = compute_alpha_bar(500, 1000, "unknown_schedule");
        let ab_cos = compute_alpha_bar(500, 1000, "cosine");
        assert!((ab_unk - ab_cos).abs() < EPSILON);
    }

    #[test]
    fn test_alpha_bar_in_zero_one() {
        for t in [0, 100, 500, 999, 1000] {
            let ab = compute_alpha_bar(t, 1000, "cosine");
            assert!((0.0..=1.0).contains(&ab), "out of [0,1] at t={t}");
        }
    }

    // -----------------------------------------------------------------------
    // compute_timestep_snr
    // -----------------------------------------------------------------------

    #[test]
    fn test_snr_t0_high() {
        let snr = compute_timestep_snr(0, 1000, "cosine", 5.0).expect("t=0 SNR");
        assert!(
            snr.snr > 100.0,
            "SNR at t=0 should be large, got {}",
            snr.snr
        );
    }

    #[test]
    fn test_snr_tmax_near_zero() {
        let snr = compute_timestep_snr(999, 1000, "cosine", 5.0).expect("t=999 SNR");
        assert!(
            snr.snr < 1.0,
            "SNR near t=T should be small, got {}",
            snr.snr
        );
    }

    #[test]
    fn test_snr_invalid_timestep_error() {
        let r = compute_timestep_snr(1001, 1000, "cosine", 5.0);
        assert!(matches!(r, Err(NoiseAnalysisError::InvalidTimestep { .. })));
    }

    #[test]
    fn test_snr_loss_weight_clamped_at_one() {
        // At low t (high SNR), loss_weight should equal the clamped value / snr,
        // which must be ≤ 1.0.
        for t in [0, 100, 500, 999] {
            let snr = compute_timestep_snr(t, 1000, "cosine", 5.0).expect("snr");
            assert!(
                snr.loss_weight <= 1.0 + EPSILON,
                "loss_weight must be ≤ 1 at t={t}, got {}",
                snr.loss_weight
            );
            assert!(snr.loss_weight >= 0.0, "loss_weight must be ≥ 0");
        }
    }

    #[test]
    fn test_snr_db_finite_for_mid_timestep() {
        let snr = compute_timestep_snr(500, 1000, "cosine", 5.0).expect("t=500 SNR");
        assert!(snr.snr_db.is_finite(), "snr_db should be finite at t=500");
    }

    #[test]
    fn test_snr_alpha_bar_matches_direct() {
        let ab_direct = compute_alpha_bar(300, 1000, "linear");
        let snr = compute_timestep_snr(300, 1000, "linear", 5.0).expect("snr linear");
        assert!((snr.alpha_bar - ab_direct).abs() < EPSILON);
    }

    // Regression test: `snr_clamp` was previously hardcoded to 5.0 inside
    // `compute_timestep_snr` and the parameter had no effect on the result.
    #[test]
    fn test_snr_clamp_is_wired_through() {
        // t=0 has very high SNR, so loss_weight == snr_clamp / snr for any
        // clamp well below that SNR value, making the two results diverge
        // proportionally to the clamp used.
        let snr_small_clamp = compute_timestep_snr(0, 1000, "cosine", 1.0).expect("clamp 1.0");
        let snr_large_clamp = compute_timestep_snr(0, 1000, "cosine", 20.0).expect("clamp 20.0");
        assert!(
            (snr_large_clamp.loss_weight - 20.0 * snr_small_clamp.loss_weight).abs() < 1e-3,
            "loss_weight should scale linearly with snr_clamp when unclamped at 1.0, got {} vs {}",
            snr_small_clamp.loss_weight,
            snr_large_clamp.loss_weight
        );
        assert!(
            (snr_small_clamp.loss_weight - snr_large_clamp.loss_weight).abs() > 1e-6,
            "different snr_clamp values must produce different loss_weight"
        );
    }

    // -----------------------------------------------------------------------
    // vec_std
    // -----------------------------------------------------------------------

    #[test]
    fn test_vec_std_constant_is_zero() {
        assert!((vec_std(&[3.0, 3.0, 3.0, 3.0])).abs() < EPSILON);
    }

    #[test]
    fn test_vec_std_known_values() {
        // Population std of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        let v = [2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = vec_std(&v);
        assert!((s - 2.0).abs() < 1e-4, "expected std ≈ 2.0, got {s}");
    }

    #[test]
    fn test_vec_std_single_element() {
        assert!((vec_std(&[42.0])).abs() < EPSILON);
    }

    #[test]
    fn test_vec_std_empty() {
        assert!((vec_std(&[])).abs() < EPSILON);
    }

    // -----------------------------------------------------------------------
    // channel_means
    // -----------------------------------------------------------------------

    #[test]
    fn test_channel_means_basic() {
        // 2 pixels, 3 channels: [1,2,3, 4,5,6]
        // ch0 mean=(1+4)/2=2.5, ch1=(2+5)/2=3.5, ch2=(3+6)/2=4.5
        let buf = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let means = channel_means(&buf, 3);
        assert_eq!(means.len(), 3);
        assert!((means[0] - 2.5).abs() < EPSILON);
        assert!((means[1] - 3.5).abs() < EPSILON);
        assert!((means[2] - 4.5).abs() < EPSILON);
    }

    #[test]
    fn test_channel_means_single_channel() {
        let buf = [1.0_f32, 2.0, 3.0, 4.0];
        let means = channel_means(&buf, 1);
        assert_eq!(means.len(), 1);
        assert!((means[0] - 2.5).abs() < EPSILON);
    }

    #[test]
    fn test_channel_means_empty_returns_empty() {
        assert!(channel_means(&[], 3).is_empty());
    }

    #[test]
    fn test_channel_means_zero_channels_returns_empty() {
        assert!(channel_means(&[1.0, 2.0], 0).is_empty());
    }

    // -----------------------------------------------------------------------
    // compute_spatial_error_map
    // -----------------------------------------------------------------------

    #[test]
    fn test_spatial_error_uniform_perfect() {
        // Perfect prediction → all errors = 0
        let buf = uniform_buf(4 * 4 * 3, 0.5);
        let map = compute_spatial_error_map(&buf, &buf, 4, 4, 3).expect("spatial");
        assert!(map.mean_error < EPSILON);
        assert!(map.max_error < EPSILON);
        assert_eq!(map.errors.len(), 16);
    }

    #[test]
    fn test_spatial_error_uniform_known() {
        // pred is all 1.0, true is all 0.0 → per-pixel MAE = 1.0
        let pred = uniform_buf(2 * 2 * 2, 1.0);
        let truth = uniform_buf(2 * 2 * 2, 0.0);
        let map = compute_spatial_error_map(&pred, &truth, 2, 2, 2).expect("spatial known");
        assert_eq!(map.errors.len(), 4);
        for &e in &map.errors {
            assert!(
                (e - 1.0).abs() < EPSILON,
                "per-pixel error should be 1.0, got {e}"
            );
        }
        assert!((map.mean_error - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_spatial_error_empty_buffer_error() {
        let r = compute_spatial_error_map(&[], &[], 0, 0, 3);
        assert!(matches!(r, Err(NoiseAnalysisError::EmptyBuffer)));
    }

    #[test]
    fn test_spatial_error_length_mismatch() {
        let pred = vec![0.0_f32; 12];
        let truth = vec![0.0_f32; 9];
        let r = compute_spatial_error_map(&pred, &truth, 2, 2, 3);
        assert!(matches!(r, Err(NoiseAnalysisError::LengthMismatch { .. })));
    }

    #[test]
    fn test_spatial_error_map_dimensions() {
        let pred = uniform_buf(3 * 5 * 3, 0.2);
        let truth = uniform_buf(3 * 5 * 3, 0.5);
        let map = compute_spatial_error_map(&pred, &truth, 5, 3, 3).expect("dims");
        assert_eq!(map.width, 5);
        assert_eq!(map.height, 3);
        assert_eq!(map.channels, 3);
        assert_eq!(map.errors.len(), 15);
    }

    // Regression test: `NoiseAnalysisConfig::error_threshold` was previously
    // never read anywhere in the crate.
    #[test]
    fn test_fraction_above_counts_pixels_over_threshold() {
        let map = SpatialErrorMap {
            errors: vec![0.0, 0.05, 0.2, 0.5, 0.9],
            width: 5,
            height: 1,
            channels: 1,
            mean_error: 0.33,
            max_error: 0.9,
            error_std: 0.0,
        };
        // 0.2, 0.5, 0.9 exceed 0.1 -> 3/5.
        assert!((map.fraction_above(0.1) - 0.6).abs() < EPSILON);
        // Nothing exceeds 10.0.
        assert!((map.fraction_above(10.0)).abs() < EPSILON);
        // Everything exceeds a negative threshold.
        assert!((map.fraction_above(-1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_fraction_above_empty_map_is_zero() {
        let map = SpatialErrorMap {
            errors: Vec::new(),
            width: 0,
            height: 0,
            channels: 1,
            mean_error: 0.0,
            max_error: 0.0,
            error_std: 0.0,
        };
        assert!((map.fraction_above(0.0)).abs() < EPSILON);
    }

    // Regression test: a buffer shorter than `width * height * channels`
    // (but still evenly divisible by `channels`) must be rejected, not
    // silently zero-padded into a perfect-looking report.
    #[test]
    fn test_spatial_error_map_rejects_buffer_shorter_than_declared_dims() {
        // width=4, height=4, channels=3 declares 48 elements; supply only 24
        // (still a multiple of 3, so the old divisibility-only check would
        // have accepted it and silently treated the missing half as
        // zero-error pixels).
        let pred = uniform_buf(24, 1.0);
        let truth = uniform_buf(24, 0.0);
        let r = compute_spatial_error_map(&pred, &truth, 4, 4, 3);
        assert!(matches!(
            r,
            Err(NoiseAnalysisError::InvalidDimensions {
                len: 24,
                channels: 3
            })
        ));
    }

    // -----------------------------------------------------------------------
    // detect_failure_mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_diverged() {
        let pred = uniform_buf(4, 0.0);
        let truth = uniform_buf(4, 2.0); // diff=2, mse=4 > 2.0
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        let fm = detect_failure_mode(&pred, &truth, &stats, 3, 1, 1);
        assert_eq!(fm, Some(FailureMode::Diverged));
    }

    #[test]
    fn test_detect_oversmoothing() {
        // Prediction is very smooth (near-zero std); truth has large spread
        let pred = uniform_buf(100, 0.01); // near constant
        let truth = linspace_buf(100, -1.0, 1.0); // large spread
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        let fm = detect_failure_mode(&pred, &truth, &stats, 3, 1, 1);
        // mse will be around 0.33 (< 2.0) so won't diverge; should detect Oversmoothing
        assert_eq!(fm, Some(FailureMode::Oversmoothing));
    }

    #[test]
    fn test_detect_color_shift() {
        // pred and truth with large channel bias but similar variance
        let pred: Vec<f32> = (0..90)
            .map(|i| if i % 3 == 0 { 0.5 } else { 0.0 })
            .collect();
        let truth: Vec<f32> = (0..90).map(|_| 0.0_f32).collect();
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        // 90 values / 3 channels = 30 pixels, below HIGH_FREQ_MIN_PIXELS
        // (64), so HighFreqArtifact detection is skipped here and this
        // falls straight through to the expected ColorShift.
        let fm = detect_failure_mode(&pred, &truth, &stats, 3, 30, 1);
        assert_eq!(fm, Some(FailureMode::ColorShift));
    }

    #[test]
    fn test_detect_no_failure() {
        let v = linspace_buf(30, -0.3, 0.3);
        let stats = compute_prediction_stats(&v, &v).expect("perfect");
        let fm = detect_failure_mode(&v, &v, &stats, 3, 10, 1);
        assert_eq!(fm, None);
    }

    #[test]
    fn test_detect_diverged_takes_priority() {
        // Both MSE > 2.0 AND smoothing; Diverged should win
        let pred = uniform_buf(10, 0.0);
        let truth = uniform_buf(10, 3.0); // mse = 9.0
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        let fm = detect_failure_mode(&pred, &truth, &stats, 3, 1, 1);
        assert_eq!(fm, Some(FailureMode::Diverged));
    }

    // Regression tests for the previously-unreachable HighFreqArtifact variant.

    /// Deterministic LCG (glibc-style constants), matched against an
    /// independent Python reference implementation used to calibrate
    /// [`HIGH_FREQ_ENERGY_RATIO_THRESHOLD`] / [`HIGH_FREQ_MIN_PIXELS`].
    fn lcg_next(state: u64) -> u64 {
        (1103515245u64.wrapping_mul(state).wrapping_add(12345)) & 0x7FFF_FFFF
    }

    /// Generate a `pred` buffer of i.i.d.-ish zero-mean noise (a 4-uniform
    /// sum, approximating a bounded Gaussian via a crude CLT) with the given
    /// `scale`, alongside an all-zero `true_noise` of the same shape — i.e.
    /// a synthetic "healthy" noise-prediction residual with no structure.
    fn gen_noise_pred(n_pixels: usize, channels: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        let mut pred = Vec::with_capacity(n_pixels * channels);
        for _ in 0..(n_pixels * channels) {
            let mut u = 0.0_f32;
            for _ in 0..4 {
                state = lcg_next(state);
                u += state as f32 / 0x7FFF_FFFF as f32;
            }
            u -= 2.0;
            pred.push(u * scale);
        }
        pred
    }

    #[test]
    fn test_detect_high_freq_artifact() {
        // Checkerboard-pattern error (alternating between 1 and 0 every
        // pixel) is almost entirely high-frequency energy: a box blur
        // collapses most of its magnitude, so the "energy removed by
        // blurring" is large relative to the raw error magnitude.
        let n = 80; // >= HIGH_FREQ_MIN_PIXELS
        let pred: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let truth = vec![0.0_f32; n];
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        let fm = detect_failure_mode(&pred, &truth, &stats, 1, n, 1);
        assert_eq!(fm, Some(FailureMode::HighFreqArtifact));
    }

    #[test]
    fn test_detect_high_freq_artifact_not_triggered_by_smooth_gradient() {
        // A smoothly-varying (linear) error survives box-blurring almost
        // unchanged in its interior, so it must not be misclassified as
        // ringing.
        let n = 80; // >= HIGH_FREQ_MIN_PIXELS
        let pred = linspace_buf(n, 0.0, 1.0);
        let truth = vec![0.0_f32; n];
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        let fm = detect_failure_mode(&pred, &truth, &stats, 1, n, 1);
        assert_ne!(fm, Some(FailureMode::HighFreqArtifact));
    }

    // Regression tests for the false-positive risk flagged during review:
    // genuinely random (i.i.d.) prediction noise is NOT itself evidence of
    // ringing, even though a box blur removes a large fraction of its
    // energy too. These use a fixed deterministic PRNG sequence (values
    // cross-checked against an independent reference implementation) rather
    // than `rand`, so the result is reproducible.
    #[test]
    fn test_detect_high_freq_artifact_not_triggered_by_random_noise_1channel() {
        let n = 100; // >= HIGH_FREQ_MIN_PIXELS
        let pred = gen_noise_pred(n, 1, 42, 0.3);
        let truth = vec![0.0_f32; n];
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        let fm = detect_failure_mode(&pred, &truth, &stats, 1, n, 1);
        assert_ne!(
            fm,
            Some(FailureMode::HighFreqArtifact),
            "pure i.i.d. noise (1 channel) must not be misclassified as ringing"
        );
    }

    #[test]
    fn test_detect_high_freq_artifact_not_triggered_by_random_noise_3channel() {
        let n_pixels = 100; // >= HIGH_FREQ_MIN_PIXELS
        let pred = gen_noise_pred(n_pixels, 3, 42, 0.3);
        let truth = vec![0.0_f32; n_pixels * 3];
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        let fm = detect_failure_mode(&pred, &truth, &stats, 3, n_pixels, 1);
        assert_ne!(
            fm,
            Some(FailureMode::HighFreqArtifact),
            "pure i.i.d. noise (3 channels) must not be misclassified as ringing"
        );
    }

    #[test]
    fn test_detect_high_freq_artifact_skipped_below_min_pixels() {
        // A checkerboard pattern that WOULD trigger at n >= 64 must be
        // skipped below HIGH_FREQ_MIN_PIXELS, since the ratio statistic is
        // unreliable at very small pixel counts.
        let n = 20;
        let pred: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let truth = vec![0.0_f32; n];
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        let fm = detect_failure_mode(&pred, &truth, &stats, 1, n, 1);
        assert_ne!(fm, Some(FailureMode::HighFreqArtifact));
    }

    #[test]
    fn test_detect_high_freq_artifact_skipped_on_shape_mismatch() {
        // width * height * channels != pred.len(): the HighFreqArtifact
        // check must be silently skipped (not panic, not error) rather than
        // attempting an out-of-bounds reshape.
        let pred = uniform_buf(7, 1.0);
        let truth = uniform_buf(7, 0.0);
        let stats = compute_prediction_stats(&pred, &truth).expect("stats");
        // 7 does not factor as width * height * channels for (3, 3, 1).
        let fm = detect_failure_mode(&pred, &truth, &stats, 1, 3, 3);
        // MSE = 1.0 (< 2.0, no Diverged); std_pred == 0 == std_true (no
        // Oversmoothing); shape mismatch skips HighFreqArtifact; per-channel
        // bias of 1.0 (> 0.1) still correctly falls through to ColorShift.
        assert_eq!(fm, Some(FailureMode::ColorShift));
    }

    // -----------------------------------------------------------------------
    // analyze_noise_prediction
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyze_smoke_test() {
        let pred = linspace_buf(3 * 4 * 4, -0.5, 0.5);
        let truth = linspace_buf(3 * 4 * 4, -0.4, 0.4);
        let cfg = NoiseAnalysisConfig::default();
        let report =
            analyze_noise_prediction(&pred, &truth, 500, 4, 4, &cfg).expect("analyze smoke test");
        assert_eq!(report.timestep, 500);
        assert!(report.stats.mse.is_finite());
        assert!(report.snr_info.snr.is_finite());
        assert_eq!(report.spatial_error.errors.len(), 16);
        assert!((0.0..=1.0).contains(&report.fraction_pixels_above_threshold));
    }

    // Regression test: `analyze_noise_prediction` now threads
    // `config.error_threshold` into the report end-to-end.
    #[test]
    fn test_analyze_fraction_pixels_above_threshold_matches_config() {
        let pred = linspace_buf(3 * 4 * 4, -0.5, 0.5);
        let truth = linspace_buf(3 * 4 * 4, -0.4, 0.4);
        let cfg = NoiseAnalysisConfig::default();
        let report = analyze_noise_prediction(&pred, &truth, 500, 4, 4, &cfg).expect("report");
        let expected = report.spatial_error.fraction_above(cfg.error_threshold);
        assert!((report.fraction_pixels_above_threshold - expected).abs() < EPSILON);
    }

    #[test]
    fn test_analyze_length_mismatch_error() {
        let pred = vec![0.0_f32; 12];
        let truth = vec![0.0_f32; 6];
        let cfg = NoiseAnalysisConfig::default();
        let r = analyze_noise_prediction(&pred, &truth, 0, 4, 1, &cfg);
        assert!(matches!(r, Err(NoiseAnalysisError::LengthMismatch { .. })));
    }

    #[test]
    fn test_analyze_empty_buffer_error() {
        let cfg = NoiseAnalysisConfig::default();
        let r = analyze_noise_prediction(&[], &[], 0, 0, 0, &cfg);
        assert!(matches!(r, Err(NoiseAnalysisError::EmptyBuffer)));
    }

    // -----------------------------------------------------------------------
    // weighted_noise_loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_weighted_loss_finite() {
        let pred = linspace_buf(12, -0.5, 0.5);
        let truth = linspace_buf(12, -0.4, 0.6);
        let w = weighted_noise_loss(&pred, &truth, 500, 1000, "cosine", 5.0).expect("wloss");
        assert!(w.is_finite());
        assert!(w >= 0.0);
    }

    #[test]
    fn test_weighted_loss_perfect_is_zero() {
        let v = linspace_buf(12, -1.0, 1.0);
        let w = weighted_noise_loss(&v, &v, 500, 1000, "cosine", 5.0).expect("perfect wloss");
        assert!(w < EPSILON);
    }

    #[test]
    fn test_weighted_loss_invalid_timestep() {
        let v = vec![0.0_f32; 12];
        let r = weighted_noise_loss(&v, &v, 2000, 1000, "cosine", 5.0);
        assert!(matches!(r, Err(NoiseAnalysisError::InvalidTimestep { .. })));
    }

    #[test]
    fn test_weighted_loss_snr_clamp_changes_result() {
        // t=0 (very high SNR) so loss_weight is unclamped and scales with
        // snr_clamp; a non-trivial mse makes the resulting loss differ too.
        let pred = linspace_buf(12, -0.5, 0.5);
        let truth = linspace_buf(12, -0.4, 0.6);
        let w_small = weighted_noise_loss(&pred, &truth, 0, 1000, "cosine", 1.0).expect("w1");
        let w_large = weighted_noise_loss(&pred, &truth, 0, 1000, "cosine", 20.0).expect("w20");
        assert!(
            (w_large - 20.0 * w_small).abs() < 1e-3,
            "loss should scale with snr_clamp, got {w_small} vs {w_large}"
        );
    }

    // -----------------------------------------------------------------------
    // error_map_percentile
    // -----------------------------------------------------------------------

    #[test]
    fn test_percentile_p0_is_min() {
        let pred = linspace_buf(3 * 4 * 4, 0.0, 1.0);
        let truth = uniform_buf(3 * 4 * 4, 0.0);
        let map = compute_spatial_error_map(&pred, &truth, 4, 4, 3).expect("map");
        let p0 = error_map_percentile(&map, 0.0);
        let min = map.errors.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!((p0 - min).abs() < EPSILON);
    }

    #[test]
    fn test_percentile_p100_is_max() {
        let pred = linspace_buf(3 * 4 * 4, 0.0, 1.0);
        let truth = uniform_buf(3 * 4 * 4, 0.5);
        let map = compute_spatial_error_map(&pred, &truth, 4, 4, 3).expect("map");
        let p100 = error_map_percentile(&map, 100.0);
        let max = map.errors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((p100 - max).abs() < EPSILON);
    }

    #[test]
    fn test_percentile_p50_between_min_max() {
        let pred = linspace_buf(3 * 4 * 4, 0.0, 1.0);
        let truth = uniform_buf(3 * 4 * 4, 0.0);
        let map = compute_spatial_error_map(&pred, &truth, 4, 4, 3).expect("map");
        let p50 = error_map_percentile(&map, 50.0);
        let min = map.errors.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = map.errors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(p50 >= min && p50 <= max);
    }

    #[test]
    fn test_percentile_empty_map_returns_zero() {
        let map = SpatialErrorMap {
            errors: Vec::new(),
            width: 0,
            height: 0,
            channels: 3,
            mean_error: 0.0,
            max_error: 0.0,
            error_std: 0.0,
        };
        assert!((error_map_percentile(&map, 50.0)).abs() < EPSILON);
    }

    #[test]
    fn test_percentile_clamps_out_of_range() {
        let pred = linspace_buf(3 * 4, 0.0, 1.0);
        let truth = uniform_buf(3 * 4, 0.0);
        let map = compute_spatial_error_map(&pred, &truth, 4, 1, 3).expect("map");
        let below = error_map_percentile(&map, -10.0);
        let above = error_map_percentile(&map, 110.0);
        let p0 = error_map_percentile(&map, 0.0);
        let p100 = error_map_percentile(&map, 100.0);
        assert!((below - p0).abs() < EPSILON);
        assert!((above - p100).abs() < EPSILON);
    }

    // -----------------------------------------------------------------------
    // PredictionHistory
    // -----------------------------------------------------------------------

    #[test]
    fn test_history_new_is_empty() {
        let h = PredictionHistory::new();
        assert!(h.steps.is_empty());
        assert!(h.mse_history.is_empty());
        assert!(h.best_mse().is_none());
    }

    #[test]
    fn test_history_record() {
        let mut h = PredictionHistory::new();
        h.record(0, 0.5, 0.7, 500);
        h.record(1, 0.4, 0.75, 400);
        assert_eq!(h.steps.len(), 2);
        assert_eq!(h.mse_history, vec![0.5, 0.4]);
    }

    #[test]
    fn test_history_best_mse() {
        let mut h = PredictionHistory::new();
        h.record(0, 0.8, 0.5, 900);
        h.record(1, 0.3, 0.8, 700);
        h.record(2, 0.6, 0.7, 500);
        let best = h.best_mse().expect("best mse");
        assert!((best - 0.3).abs() < EPSILON);
    }

    #[test]
    fn test_history_recent_trend_improving() {
        let mut h = PredictionHistory::new();
        for i in 0..10 {
            h.record(i, 1.0 - i as f32 * 0.05, 0.5 + i as f32 * 0.01, 500);
        }
        let trend = h.recent_trend(5).expect("trend");
        assert!(
            trend < 0.0,
            "trend should be negative (improving), got {trend}"
        );
    }

    #[test]
    fn test_history_recent_trend_worsening() {
        let mut h = PredictionHistory::new();
        for i in 0..10 {
            h.record(i, i as f32 * 0.1, 0.9 - i as f32 * 0.05, 500);
        }
        let trend = h.recent_trend(5).expect("trend");
        assert!(
            trend > 0.0,
            "trend should be positive (worsening), got {trend}"
        );
    }

    #[test]
    fn test_history_trend_insufficient_history() {
        let mut h = PredictionHistory::new();
        h.record(0, 0.5, 0.7, 500);
        let r = h.recent_trend(5);
        assert!(matches!(
            r,
            Err(NoiseAnalysisError::InsufficientHistory { .. })
        ));
    }

    #[test]
    fn test_history_is_improving() {
        let mut h = PredictionHistory::new();
        for i in 0..6 {
            h.record(i, 1.0 - i as f32 * 0.1, 0.5, 500);
        }
        let imp = h.is_improving(4).expect("is_improving");
        assert!(imp, "should be improving");
    }

    #[test]
    fn test_history_is_not_improving() {
        let mut h = PredictionHistory::new();
        for i in 0..6 {
            h.record(i, i as f32 * 0.1, 0.5, 500);
        }
        let imp = h.is_improving(4).expect("is_improving");
        assert!(!imp, "should not be improving");
    }

    // -----------------------------------------------------------------------
    // compute_noise_floor
    // -----------------------------------------------------------------------

    #[test]
    fn test_noise_floor_length() {
        let floor = compute_noise_floor(1000, "cosine");
        assert_eq!(floor.len(), 1000);
    }

    #[test]
    fn test_noise_floor_starts_near_zero() {
        let floor = compute_noise_floor(1000, "cosine");
        assert!(floor[0] < 0.01, "noise floor at t=0 should be near 0");
    }

    #[test]
    fn test_noise_floor_ends_near_one() {
        let floor = compute_noise_floor(1000, "cosine");
        assert!(floor[999] > 0.99, "noise floor at t=T should be near 1.0");
    }

    #[test]
    fn test_noise_floor_monotone_increasing() {
        let floor = compute_noise_floor(100, "cosine");
        for w in floor.windows(2) {
            assert!(w[1] >= w[0], "noise floor must be non-decreasing");
        }
    }

    #[test]
    fn test_noise_floor_in_zero_one() {
        let floor = compute_noise_floor(100, "linear");
        for (i, &v) in floor.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "out of [0,1] at i={i}: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // compare_predictions
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_predictions_winner() {
        let truth = linspace_buf(12, -1.0, 1.0);
        let good = truth.clone();
        let bad: Vec<f32> = truth.iter().map(|x| x + 0.5).collect();
        let (stats_good, stats_bad) = compare_predictions(&good, &bad, &truth).expect("compare");
        assert!(
            stats_good.mse < stats_bad.mse,
            "good pred should have lower MSE"
        );
    }

    #[test]
    fn test_compare_predictions_both_finite() {
        let truth = uniform_buf(6, 0.3);
        let pred_a = uniform_buf(6, 0.2);
        let pred_b = uniform_buf(6, 0.4);
        let (a, b) = compare_predictions(&pred_a, &pred_b, &truth).expect("compare both");
        assert!(a.mse.is_finite() && b.mse.is_finite());
    }

    #[test]
    fn test_compare_predictions_length_mismatch() {
        let truth = vec![0.0_f32; 6];
        let pred_a = vec![0.0_f32; 6];
        let pred_b = vec![0.0_f32; 4]; // mismatch
        let r = compare_predictions(&pred_a, &pred_b, &truth);
        assert!(matches!(r, Err(NoiseAnalysisError::LengthMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // format_noise_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_noise_report_non_empty() {
        let pred = linspace_buf(3 * 4 * 4, -0.5, 0.5);
        let truth = linspace_buf(3 * 4 * 4, -0.4, 0.4);
        let cfg = NoiseAnalysisConfig::default();
        let report = analyze_noise_prediction(&pred, &truth, 300, 4, 4, &cfg).expect("report");
        let s = format_noise_report(&report);
        assert!(!s.is_empty());
        assert!(s.contains("300"), "should contain timestep");
        assert!(s.contains("MSE="), "should contain MSE label");
    }

    #[test]
    fn test_format_noise_report_failure_mode_shown() {
        // Build a report with Diverged failure mode manually
        let pred = uniform_buf(12, 0.0);
        let truth = uniform_buf(12, 2.0);
        let cfg = NoiseAnalysisConfig::default();
        let report = analyze_noise_prediction(&pred, &truth, 100, 2, 2, &cfg).expect("report");
        let s = format_noise_report(&report);
        assert!(s.contains("Diverged"), "should show failure mode");
    }

    // -----------------------------------------------------------------------
    // NoiseAnalysisError variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_empty_buffer_display() {
        let e = NoiseAnalysisError::EmptyBuffer;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_length_mismatch_display() {
        let e = NoiseAnalysisError::LengthMismatch {
            pred_len: 4,
            true_len: 6,
        };
        let s = e.to_string();
        assert!(s.contains("4") && s.contains("6"));
    }

    #[test]
    fn test_error_invalid_dimensions_display() {
        let e = NoiseAnalysisError::InvalidDimensions {
            len: 7,
            channels: 3,
        };
        let s = e.to_string();
        assert!(s.contains("7") && s.contains("3"));
    }

    #[test]
    fn test_error_invalid_timestep_display() {
        let e = NoiseAnalysisError::InvalidTimestep {
            t: 1500,
            max_t: 1000,
        };
        let s = e.to_string();
        assert!(s.contains("1500") && s.contains("1000"));
    }

    #[test]
    fn test_error_insufficient_history_display() {
        let e = NoiseAnalysisError::InsufficientHistory { needed: 10, got: 3 };
        let s = e.to_string();
        assert!(s.contains("10") && s.contains("3"));
    }

    // -----------------------------------------------------------------------
    // FailureMode variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_failure_mode_variants_debug() {
        let modes = [
            FailureMode::Oversmoothing,
            FailureMode::HighFreqArtifact,
            FailureMode::ColorShift,
            FailureMode::Diverged,
        ];
        for m in &modes {
            let s = format!("{:?}", m);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_failure_mode_clone_eq() {
        let a = FailureMode::Oversmoothing;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // linear_regression_slope (internal, tested via recent_trend)
    // -----------------------------------------------------------------------

    #[test]
    fn test_regression_constant_slope_is_zero() {
        let mut h = PredictionHistory::new();
        for i in 0..10 {
            h.record(i, 0.5, 0.7, 500);
        }
        let trend = h.recent_trend(10).expect("trend");
        assert!(trend.abs() < 1e-4, "constant MSE → slope ~0, got {trend}");
    }

    #[test]
    fn test_noise_analysis_config_default() {
        let cfg = NoiseAnalysisConfig::default();
        assert_eq!(cfg.channels, 3);
        assert_eq!(cfg.max_timesteps, 1000);
        assert!((cfg.snr_clamp - 5.0).abs() < EPSILON);
        assert!((cfg.error_threshold - 0.1).abs() < EPSILON);
    }
}
