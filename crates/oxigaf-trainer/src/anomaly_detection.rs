//! Anomaly detection for 3D Gaussian Splatting training loops.
//!
//! Monitors the training process for signs of instability or poor convergence,
//! providing structured diagnostics and recovery suggestions. Automatically
//! identifies and classifies: NaN/Inf values, exploding/vanishing gradients,
//! mode collapse, loss spikes, opacity collapse, and position drift.
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::anomaly_detection::{AnomalyDetector, AnomalyDetectorConfig, AnomalyThresholds};
//!
//! let config = AnomalyDetectorConfig::default();
//! let mut detector = AnomalyDetector::new(config);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by anomaly detection operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum AnomalyDetectionError {
    /// Input slice is empty when data is required.
    #[error("empty input")]
    EmptyInput,
    /// A threshold parameter is invalid.
    #[error("invalid threshold: {0}")]
    InvalidThreshold(String),
    /// A configuration parameter is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// Not enough history for the requested operation.
    #[error("history too short: need {needed}, have {available}")]
    HistoryTooShort { needed: usize, available: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalySeverity
// ─────────────────────────────────────────────────────────────────────────────

/// Severity level of a detected anomaly, ordered from least to most severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    /// Informational — notable but not critical.
    Info,
    /// Potential issue — may indicate a problem.
    Warning,
    /// Likely bad — training likely broken.
    Critical,
    /// Training definitely broken.
    Fatal,
}

impl AnomalySeverity {
    /// Return a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Critical => "CRITICAL",
            Self::Fatal => "FATAL",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyKind
// ─────────────────────────────────────────────────────────────────────────────

/// Specific type of anomaly detected during training.
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyKind {
    /// NaN values found in a parameter array.
    NanValues { n_nan: usize, location: String },
    /// Infinity values found in a parameter array.
    InfValues { n_inf: usize, location: String },
    /// Gradient norm exceeds explosion threshold.
    ExplodingGradients { norm: f32, threshold: f32 },
    /// Gradient norm is below vanishing threshold.
    VanishingGradients { norm: f32, threshold: f32 },
    /// Loss has spiked relative to running mean.
    LossSpike {
        current: f32,
        expected: f32,
        ratio: f32,
    },
    /// Loss has been consistently increasing.
    LossDivergence { steps_increasing: usize },
    /// Mean Gaussian opacity has collapsed below threshold.
    OpacityCollapse { mean_opacity: f32, threshold: f32 },
    /// Gaussian scale values have exploded.
    ScaleExplosion { max_scale: f32, threshold: f32 },
    /// Gaussians have drifted too far from reference positions.
    PositionDrift { max_drift: f32, threshold: f32 },
    /// All Gaussians have nearly identical opacity (mode collapse).
    ModeCollapse { opacity_std: f32, threshold: f32 },
    /// NaN or Inf found in gradient values.
    GradientNanInf { location: String },
    /// Convergence rate is below expectation.
    SlowConvergence {
        improvement_rate: f32,
        expected: f32,
    },
}

impl AnomalyKind {
    /// Return the default severity level of this anomaly kind.
    pub fn default_severity(&self) -> AnomalySeverity {
        match self {
            Self::NanValues { .. } => AnomalySeverity::Fatal,
            Self::InfValues { .. } => AnomalySeverity::Fatal,
            Self::GradientNanInf { .. } => AnomalySeverity::Fatal,
            Self::ExplodingGradients { .. } => AnomalySeverity::Critical,
            Self::LossDivergence { .. } => AnomalySeverity::Critical,
            Self::OpacityCollapse { .. } => AnomalySeverity::Critical,
            Self::ScaleExplosion { .. } => AnomalySeverity::Critical,
            Self::PositionDrift { .. } => AnomalySeverity::Warning,
            Self::ModeCollapse { .. } => AnomalySeverity::Warning,
            Self::LossSpike { .. } => AnomalySeverity::Warning,
            Self::VanishingGradients { .. } => AnomalySeverity::Warning,
            Self::SlowConvergence { .. } => AnomalySeverity::Info,
        }
    }

    /// Return a short human-readable description of this anomaly.
    pub fn description(&self) -> String {
        match self {
            Self::NanValues { n_nan, location } => {
                format!("NaN values ({}) in '{}'", n_nan, location)
            }
            Self::InfValues { n_inf, location } => {
                format!("Inf values ({}) in '{}'", n_inf, location)
            }
            Self::ExplodingGradients { norm, threshold } => {
                format!(
                    "Exploding gradients: norm {:.4e} > threshold {:.4e}",
                    norm, threshold
                )
            }
            Self::VanishingGradients { norm, threshold } => {
                format!(
                    "Vanishing gradients: norm {:.4e} < threshold {:.4e}",
                    norm, threshold
                )
            }
            Self::LossSpike {
                current,
                expected,
                ratio,
            } => {
                format!(
                    "Loss spike: current {:.4e}, expected {:.4e}, ratio {:.2}x",
                    current, expected, ratio
                )
            }
            Self::LossDivergence { steps_increasing } => {
                format!(
                    "Loss divergence: {} consecutive increasing steps",
                    steps_increasing
                )
            }
            Self::OpacityCollapse {
                mean_opacity,
                threshold,
            } => {
                format!(
                    "Opacity collapse: mean {:.4e} < threshold {:.4e}",
                    mean_opacity, threshold
                )
            }
            Self::ScaleExplosion {
                max_scale,
                threshold,
            } => {
                format!(
                    "Scale explosion: max_scale {:.4e} > threshold {:.4e}",
                    max_scale, threshold
                )
            }
            Self::PositionDrift {
                max_drift,
                threshold,
            } => {
                format!(
                    "Position drift: max_drift {:.4e} > threshold {:.4e}",
                    max_drift, threshold
                )
            }
            Self::ModeCollapse {
                opacity_std,
                threshold,
            } => {
                format!(
                    "Mode collapse: opacity_std {:.4e} < threshold {:.4e}",
                    opacity_std, threshold
                )
            }
            Self::GradientNanInf { location } => {
                format!("NaN/Inf in gradients at '{}'", location)
            }
            Self::SlowConvergence {
                improvement_rate,
                expected,
            } => {
                format!(
                    "Slow convergence: improvement_rate {:.4e}, expected {:.4e}",
                    improvement_rate, expected
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyEvent
// ─────────────────────────────────────────────────────────────────────────────

/// A detected anomaly with context.
#[derive(Debug, Clone)]
pub struct AnomalyEvent {
    /// The type of anomaly detected.
    pub kind: AnomalyKind,
    /// Severity level for this event.
    pub severity: AnomalySeverity,
    /// Training step when the anomaly was detected.
    pub step: usize,
    /// Human-readable message.
    pub message: String,
}

impl AnomalyEvent {
    /// Create a new event. Severity and message are derived from `kind`.
    pub fn new(kind: AnomalyKind, step: usize) -> Self {
        let severity = kind.default_severity();
        let message = format!(
            "[Step {}] [{}] {}",
            step,
            severity.label(),
            kind.description()
        );
        Self {
            kind,
            severity,
            step,
            message,
        }
    }

    /// Returns true if severity is Fatal.
    pub fn is_fatal(&self) -> bool {
        self.severity == AnomalySeverity::Fatal
    }

    /// Returns true if severity is Critical or Fatal.
    pub fn is_critical_or_above(&self) -> bool {
        self.severity >= AnomalySeverity::Critical
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyThresholds
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration thresholds for all anomaly detection checks.
#[derive(Debug, Clone)]
pub struct AnomalyThresholds {
    /// Gradient L2 norm above this triggers ExplodingGradients.
    pub max_gradient_norm: f32,
    /// Gradient L2 norm below this triggers VanishingGradients.
    pub min_gradient_norm: f32,
    /// Loss spike: current / mean > ratio triggers LossSpike.
    pub loss_spike_ratio: f32,
    /// Consecutive loss-increasing steps to flag LossDivergence.
    pub loss_divergence_steps: usize,
    /// Mean opacity below this triggers OpacityCollapse.
    pub min_mean_opacity: f32,
    /// Max log-space scale (real scale = exp(log_scale)) above this triggers ScaleExplosion.
    pub max_gaussian_scale: f32,
    /// Max Gaussian drift from reference positions above this triggers PositionDrift.
    pub max_position_drift: f32,
    /// Opacity std below this triggers ModeCollapse.
    pub min_opacity_std: f32,
    /// Window size (steps) for convergence rate calculation.
    pub slow_convergence_window: usize,
    /// Minimum PSNR improvement per step to avoid SlowConvergence.
    pub slow_convergence_min_rate: f32,
}

impl Default for AnomalyThresholds {
    fn default() -> Self {
        Self {
            max_gradient_norm: 1000.0,
            min_gradient_norm: 1e-10,
            loss_spike_ratio: 10.0,
            loss_divergence_steps: 50,
            min_mean_opacity: 0.001,
            max_gaussian_scale: 10.0,
            max_position_drift: 5.0,
            min_opacity_std: 1e-6,
            slow_convergence_window: 1000,
            slow_convergence_min_rate: 1e-4,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the mean and standard deviation of a slice.
/// Returns `(0.0, 0.0)` for an empty slice.
pub fn anom_mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().copied().sum::<f32>() / n;
    let variance = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    (mean, variance.sqrt())
}

/// Count NaN and Inf values in a slice.
/// Returns `(n_nan, n_inf)`.
pub fn anom_count_nonfinite(values: &[f32]) -> (usize, usize) {
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    for &v in values {
        if v.is_nan() {
            n_nan += 1;
        } else if v.is_infinite() {
            n_inf += 1;
        }
    }
    (n_nan, n_inf)
}

/// Check whether the last `n` values in a slice form a strictly monotone increasing sequence.
/// Returns `false` if `n == 0` or `n > values.len()`.
pub fn anom_is_monotone_increasing(values: &[f32], n: usize) -> bool {
    if n == 0 || n > values.len() {
        return false;
    }
    let tail = &values[values.len() - n..];
    tail.windows(2).all(|w| w[1] > w[0])
}

/// Compute the L2 norm of a slice. Returns 0.0 for empty input.
pub fn anom_l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Compute the maximum absolute value in a slice. Returns 0.0 for empty input.
pub fn anom_max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()))
}

/// Compute the maximum per-element L2 distance between two slices of length N×3.
/// Returns an error if the lengths differ.
pub fn anom_max_pairwise_dist(a: &[f32], b: &[f32]) -> Result<f32, AnomalyDetectionError> {
    if a.len() != b.len() {
        return Err(AnomalyDetectionError::InvalidConfig(format!(
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    // Treat as N×3 vectors; compute per-point distance.
    // If not divisible by 3, fall back to element-wise comparison.
    let stride = if a.len().is_multiple_of(3) { 3 } else { 1 };
    let n_points = a.len() / stride;
    let mut max_dist = 0.0f32;
    for i in 0..n_points {
        let mut dist_sq = 0.0f32;
        for j in 0..stride {
            let d = a[i * stride + j] - b[i * stride + j];
            dist_sq += d * d;
        }
        let dist = dist_sq.sqrt();
        if dist > max_dist {
            max_dist = dist;
        }
    }
    Ok(max_dist)
}

// ─────────────────────────────────────────────────────────────────────────────
// Core detection functions
// ─────────────────────────────────────────────────────────────────────────────

/// Check for NaN/Inf values in a parameter array.
/// Returns events for each type of non-finite value found.
pub fn anom_check_numerical(values: &[f32], location: &str) -> Vec<AnomalyEvent> {
    let (n_nan, n_inf) = anom_count_nonfinite(values);
    let mut events = Vec::new();
    let step = 0; // standalone function uses step 0
    if n_nan > 0 {
        events.push(AnomalyEvent::new(
            AnomalyKind::NanValues {
                n_nan,
                location: location.to_string(),
            },
            step,
        ));
    }
    if n_inf > 0 {
        events.push(AnomalyEvent::new(
            AnomalyKind::InfValues {
                n_inf,
                location: location.to_string(),
            },
            step,
        ));
    }
    events
}

/// Check gradient norm for explosion or vanishing.
pub fn anom_check_gradient_norm(
    gradient_norm: f32,
    step: usize,
    thresholds: &AnomalyThresholds,
) -> Vec<AnomalyEvent> {
    let mut events = Vec::new();
    if !gradient_norm.is_finite() {
        return events;
    }
    if gradient_norm > thresholds.max_gradient_norm {
        events.push(AnomalyEvent::new(
            AnomalyKind::ExplodingGradients {
                norm: gradient_norm,
                threshold: thresholds.max_gradient_norm,
            },
            step,
        ));
    } else if gradient_norm < thresholds.min_gradient_norm {
        events.push(AnomalyEvent::new(
            AnomalyKind::VanishingGradients {
                norm: gradient_norm,
                threshold: thresholds.min_gradient_norm,
            },
            step,
        ));
    }
    events
}

/// Check for gradient NaN/Inf.
pub fn anom_check_gradient_numerical(
    gradients: &[f32],
    location: &str,
    step: usize,
) -> Vec<AnomalyEvent> {
    let (n_nan, n_inf) = anom_count_nonfinite(gradients);
    if n_nan > 0 || n_inf > 0 {
        vec![AnomalyEvent::new(
            AnomalyKind::GradientNanInf {
                location: location.to_string(),
            },
            step,
        )]
    } else {
        Vec::new()
    }
}

/// Check loss for spikes relative to running mean.
pub fn anom_check_loss_spike(
    current_loss: f32,
    loss_history: &[f32],
    step: usize,
    thresholds: &AnomalyThresholds,
) -> Vec<AnomalyEvent> {
    if loss_history.len() < 2 {
        return Vec::new();
    }
    if !current_loss.is_finite() {
        return Vec::new();
    }
    let (mean, _) = anom_mean_std(loss_history);
    if mean <= 0.0 {
        return Vec::new();
    }
    let ratio = current_loss / mean;
    if ratio > thresholds.loss_spike_ratio {
        vec![AnomalyEvent::new(
            AnomalyKind::LossSpike {
                current: current_loss,
                expected: mean,
                ratio,
            },
            step,
        )]
    } else {
        Vec::new()
    }
}

/// Check for consecutive loss increase (divergence).
pub fn anom_check_loss_divergence(
    loss_history: &[f32],
    step: usize,
    thresholds: &AnomalyThresholds,
) -> Vec<AnomalyEvent> {
    let n = thresholds.loss_divergence_steps + 1; // need N+1 values for N consecutive increases
    if loss_history.len() < n {
        return Vec::new();
    }
    if anom_is_monotone_increasing(loss_history, n) {
        vec![AnomalyEvent::new(
            AnomalyKind::LossDivergence {
                steps_increasing: n - 1,
            },
            step,
        )]
    } else {
        Vec::new()
    }
}

/// Check Gaussian opacity for collapse (mean opacity too low).
pub fn anom_check_opacity_collapse(
    opacities: &[f32],
    step: usize,
    thresholds: &AnomalyThresholds,
) -> Vec<AnomalyEvent> {
    if opacities.is_empty() {
        return Vec::new();
    }
    let (mean, _) = anom_mean_std(opacities);
    if mean < thresholds.min_mean_opacity {
        vec![AnomalyEvent::new(
            AnomalyKind::OpacityCollapse {
                mean_opacity: mean,
                threshold: thresholds.min_mean_opacity,
            },
            step,
        )]
    } else {
        Vec::new()
    }
}

/// Check for mode collapse: all Gaussians have nearly identical opacity (low std).
pub fn anom_check_mode_collapse(
    opacities: &[f32],
    step: usize,
    thresholds: &AnomalyThresholds,
) -> Vec<AnomalyEvent> {
    if opacities.len() < 2 {
        return Vec::new();
    }
    let (_, std) = anom_mean_std(opacities);
    if std < thresholds.min_opacity_std {
        vec![AnomalyEvent::new(
            AnomalyKind::ModeCollapse {
                opacity_std: std,
                threshold: thresholds.min_opacity_std,
            },
            step,
        )]
    } else {
        Vec::new()
    }
}

/// Check Gaussian log-space scales for explosion.
pub fn anom_check_scale_explosion(
    log_scales: &[f32],
    step: usize,
    thresholds: &AnomalyThresholds,
) -> Vec<AnomalyEvent> {
    if log_scales.is_empty() {
        return Vec::new();
    }
    let max_scale = log_scales.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_scale > thresholds.max_gaussian_scale {
        vec![AnomalyEvent::new(
            AnomalyKind::ScaleExplosion {
                max_scale,
                threshold: thresholds.max_gaussian_scale,
            },
            step,
        )]
    } else {
        Vec::new()
    }
}

/// Check position drift from reference positions (e.g., FLAME mesh binding).
/// `current_positions` and `reference_positions` are N×3 flat arrays.
pub fn anom_check_position_drift(
    current_positions: &[f32],
    reference_positions: &[f32],
    step: usize,
    thresholds: &AnomalyThresholds,
) -> Vec<AnomalyEvent> {
    let max_drift = match anom_max_pairwise_dist(current_positions, reference_positions) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    if max_drift > thresholds.max_position_drift {
        vec![AnomalyEvent::new(
            AnomalyKind::PositionDrift {
                max_drift,
                threshold: thresholds.max_position_drift,
            },
            step,
        )]
    } else {
        Vec::new()
    }
}

/// Check PSNR convergence rate over recent steps.
/// The history is expected in chronological order (most recent last).
pub fn anom_check_convergence(
    psnr_history: &[f32],
    step: usize,
    thresholds: &AnomalyThresholds,
) -> Vec<AnomalyEvent> {
    let window = thresholds.slow_convergence_window;
    if psnr_history.len() < window {
        // Not enough data: silently return no events (not an error condition)
        return Vec::new();
    }
    let tail = &psnr_history[psnr_history.len() - window..];
    let first = tail[0];
    let last = tail[tail.len() - 1];
    let improvement = last - first;
    let improvement_rate = improvement / window as f32;
    if improvement_rate < thresholds.slow_convergence_min_rate {
        vec![AnomalyEvent::new(
            AnomalyKind::SlowConvergence {
                improvement_rate,
                expected: thresholds.slow_convergence_min_rate,
            },
            step,
        )]
    } else {
        Vec::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyDetectorConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the stateful `AnomalyDetector`.
#[derive(Debug, Clone)]
pub struct AnomalyDetectorConfig {
    /// Detection thresholds for all checks.
    pub thresholds: AnomalyThresholds,
    /// Only run checks every `check_interval` steps (to reduce overhead).
    pub check_interval: usize,
    /// Maximum number of events stored in history.
    pub max_history: usize,
    /// If true, `should_pause()` returns true when any fatal event has occurred.
    pub auto_pause_on_fatal: bool,
    /// Enable gradient-related checks.
    pub enable_gradient_checks: bool,
    /// Enable scene/Gaussian health checks.
    pub enable_scene_checks: bool,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            thresholds: AnomalyThresholds::default(),
            check_interval: 10,
            max_history: 1000,
            auto_pause_on_fatal: true,
            enable_gradient_checks: true,
            enable_scene_checks: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful anomaly detector that accumulates events and history across training steps.
pub struct AnomalyDetector {
    config: AnomalyDetectorConfig,
    events: Vec<AnomalyEvent>,
    loss_history: Vec<f32>,
    psnr_history: Vec<f32>,
    step: usize,
    n_fatal: usize,
    n_critical: usize,
    n_warning: usize,
}

impl AnomalyDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
            loss_history: Vec::new(),
            psnr_history: Vec::new(),
            step: 0,
            n_fatal: 0,
            n_critical: 0,
            n_warning: 0,
        }
    }

    /// Run all enabled checks for the current step.
    /// Returns any new anomaly events detected.
    ///
    /// - `gradient_norm`: optional pre-computed gradient L2 norm
    /// - `loss`: current training loss
    /// - `psnr`: optional current PSNR metric
    /// - `opacities`: optional sigmoid-applied Gaussian opacities in [0, 1]
    /// - `log_scales`: optional log-space Gaussian scales
    /// - `positions`: optional tuple of (current_positions N×3, reference_positions N×3)
    pub fn check_step(
        &mut self,
        gradient_norm: Option<f32>,
        loss: f32,
        psnr: Option<f32>,
        opacities: Option<&[f32]>,
        log_scales: Option<&[f32]>,
        positions: Option<(&[f32], &[f32])>,
    ) -> Vec<AnomalyEvent> {
        // Only run checks at the configured interval.
        let should_check = self.step.is_multiple_of(self.config.check_interval);

        // Always update histories.
        self.loss_history.push(loss);
        if self.loss_history.len() > 200 {
            let drain_to = self.loss_history.len() - 200;
            self.loss_history.drain(0..drain_to);
        }
        if let Some(p) = psnr {
            self.psnr_history.push(p);
            if self.psnr_history.len() > 200 {
                let drain_to = self.psnr_history.len() - 200;
                self.psnr_history.drain(0..drain_to);
            }
        }

        if !should_check {
            return Vec::new();
        }

        let step = self.step;
        let thresholds = &self.config.thresholds;
        let mut new_events: Vec<AnomalyEvent> = Vec::new();

        // --- Gradient checks ---
        if self.config.enable_gradient_checks {
            if let Some(norm) = gradient_norm {
                new_events.extend(anom_check_gradient_norm(norm, step, thresholds));
            }
        }

        // --- Loss checks ---
        // Check loss for NaN/Inf using numerical check.
        new_events.extend(anom_check_numerical(&[loss], "loss"));
        // Loss spike relative to history (need at least 2 samples before current).
        if self.loss_history.len() > 1 {
            let history = &self.loss_history[..self.loss_history.len() - 1];
            new_events.extend(anom_check_loss_spike(loss, history, step, thresholds));
        }
        // Loss divergence.
        new_events.extend(anom_check_loss_divergence(
            &self.loss_history,
            step,
            thresholds,
        ));

        // --- Scene / Gaussian checks ---
        if self.config.enable_scene_checks {
            if let Some(ops) = opacities {
                new_events.extend(anom_check_opacity_collapse(ops, step, thresholds));
                new_events.extend(anom_check_mode_collapse(ops, step, thresholds));
            }
            if let Some(scales) = log_scales {
                new_events.extend(anom_check_scale_explosion(scales, step, thresholds));
            }
            if let Some((curr, refs)) = positions {
                new_events.extend(anom_check_position_drift(curr, refs, step, thresholds));
            }
        }

        // --- Convergence check ---
        new_events.extend(anom_check_convergence(&self.psnr_history, step, thresholds));

        // Update severity counters.
        for event in &new_events {
            match event.severity {
                AnomalySeverity::Fatal => self.n_fatal += 1,
                AnomalySeverity::Critical => self.n_critical += 1,
                AnomalySeverity::Warning => self.n_warning += 1,
                AnomalySeverity::Info => {}
            }
        }

        // Store events, respecting max_history.
        for event in new_events.clone() {
            self.events.push(event);
        }
        if self.events.len() > self.config.max_history {
            let drain_to = self.events.len() - self.config.max_history;
            self.events.drain(0..drain_to);
        }

        new_events
    }

    /// Increment the internal step counter by 1.
    pub fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Current training step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// All accumulated anomaly events.
    pub fn events(&self) -> &[AnomalyEvent] {
        &self.events
    }

    /// Number of Fatal events observed.
    pub fn n_fatal(&self) -> usize {
        self.n_fatal
    }

    /// Number of Critical events observed.
    pub fn n_critical(&self) -> usize {
        self.n_critical
    }

    /// Number of Warning events observed.
    pub fn n_warning(&self) -> usize {
        self.n_warning
    }

    /// Returns true if auto_pause_on_fatal is set and at least one fatal event occurred.
    pub fn should_pause(&self) -> bool {
        self.config.auto_pause_on_fatal && self.n_fatal > 0
    }

    /// Clear all stored events (does not reset severity counters).
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Return the last `n` events (or fewer if not enough have been stored).
    pub fn recent_events(&self, n: usize) -> &[AnomalyEvent] {
        let len = self.events.len();
        if n >= len {
            &self.events
        } else {
            &self.events[len - n..]
        }
    }

    /// Return a count histogram `[n_info, n_warning, n_critical, n_fatal]`.
    pub fn severity_counts(&self) -> [usize; 4] {
        let mut counts = [0usize; 4];
        for event in &self.events {
            match event.severity {
                AnomalySeverity::Info => counts[0] += 1,
                AnomalySeverity::Warning => counts[1] += 1,
                AnomalySeverity::Critical => counts[2] += 1,
                AnomalySeverity::Fatal => counts[3] += 1,
            }
        }
        counts
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyReport and helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics report for an anomaly detector.
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    /// Number of training steps at which checks were performed.
    pub n_steps_checked: usize,
    /// Total Fatal events.
    pub n_fatal: usize,
    /// Total Critical events.
    pub n_critical: usize,
    /// Total Warning events.
    pub n_warning: usize,
    /// Total Info events.
    pub n_info: usize,
    /// The most severe anomaly kind observed, if any.
    pub most_severe: Option<AnomalyKind>,
    /// Anomaly events per 100 steps checked.
    pub anomaly_rate: f32,
}

/// Generate a summary report from a detector's accumulated state.
pub fn anom_generate_report(detector: &AnomalyDetector) -> AnomalyReport {
    let steps = detector.step().max(1); // avoid div-by-zero
    let events = detector.events();
    let n_fatal = events
        .iter()
        .filter(|e| e.severity == AnomalySeverity::Fatal)
        .count();
    let n_critical = events
        .iter()
        .filter(|e| e.severity == AnomalySeverity::Critical)
        .count();
    let n_warning = events
        .iter()
        .filter(|e| e.severity == AnomalySeverity::Warning)
        .count();
    let n_info = events
        .iter()
        .filter(|e| e.severity == AnomalySeverity::Info)
        .count();
    let total = events.len();
    let anomaly_rate = total as f32 / steps as f32 * 100.0;

    // Find the most severe event.
    let most_severe = events
        .iter()
        .max_by_key(|e| e.severity)
        .map(|e| e.kind.clone());

    // Compute steps_checked as steps / check_interval, approximated by total steps.
    let n_steps_checked = steps / detector.config.check_interval.max(1);

    AnomalyReport {
        n_steps_checked,
        n_fatal,
        n_critical,
        n_warning,
        n_info,
        most_severe,
        anomaly_rate,
    }
}

/// Format a single anomaly event as a log-line string.
pub fn anom_format_event(event: &AnomalyEvent) -> String {
    format!(
        "[Step {:>6}] [{:>8}] {}",
        event.step,
        event.severity.label(),
        event.kind.description()
    )
}

/// Format a summary report as a multi-line string.
pub fn anom_format_report(report: &AnomalyReport) -> String {
    let total = report.n_fatal + report.n_critical + report.n_warning + report.n_info;
    let mut out = String::new();
    out.push_str("=== Anomaly Detection Report ===\n");
    out.push_str(&format!("  Steps checked : {}\n", report.n_steps_checked));
    out.push_str(&format!(
        "  Total events  : {} ({:.2} per 100 steps)\n",
        total, report.anomaly_rate
    ));
    out.push_str(&format!("  Fatal         : {}\n", report.n_fatal));
    out.push_str(&format!("  Critical      : {}\n", report.n_critical));
    out.push_str(&format!("  Warning       : {}\n", report.n_warning));
    out.push_str(&format!("  Info          : {}\n", report.n_info));
    if let Some(ref kind) = report.most_severe {
        out.push_str(&format!("  Most severe   : {}\n", kind.description()));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AnomalySeverity ordering ─────────────────────────────────────────────

    #[test]
    fn test_severity_ordering_fatal_gt_critical() {
        assert!(AnomalySeverity::Fatal > AnomalySeverity::Critical);
    }

    #[test]
    fn test_severity_ordering_critical_gt_warning() {
        assert!(AnomalySeverity::Critical > AnomalySeverity::Warning);
    }

    #[test]
    fn test_severity_ordering_warning_gt_info() {
        assert!(AnomalySeverity::Warning > AnomalySeverity::Info);
    }

    #[test]
    fn test_severity_ordering_full_chain() {
        let mut severities = [
            AnomalySeverity::Fatal,
            AnomalySeverity::Info,
            AnomalySeverity::Critical,
            AnomalySeverity::Warning,
        ];
        severities.sort();
        assert_eq!(
            severities,
            [
                AnomalySeverity::Info,
                AnomalySeverity::Warning,
                AnomalySeverity::Critical,
                AnomalySeverity::Fatal,
            ]
        );
    }

    // ── AnomalyKind::default_severity ───────────────────────────────────────

    #[test]
    fn test_default_severity_nan_values_is_fatal() {
        let kind = AnomalyKind::NanValues {
            n_nan: 1,
            location: "pos".to_string(),
        };
        assert_eq!(kind.default_severity(), AnomalySeverity::Fatal);
    }

    #[test]
    fn test_default_severity_inf_values_is_fatal() {
        let kind = AnomalyKind::InfValues {
            n_inf: 1,
            location: "pos".to_string(),
        };
        assert_eq!(kind.default_severity(), AnomalySeverity::Fatal);
    }

    #[test]
    fn test_default_severity_gradient_nan_inf_is_fatal() {
        let kind = AnomalyKind::GradientNanInf {
            location: "grads".to_string(),
        };
        assert_eq!(kind.default_severity(), AnomalySeverity::Fatal);
    }

    #[test]
    fn test_default_severity_loss_spike_is_warning() {
        let kind = AnomalyKind::LossSpike {
            current: 10.0,
            expected: 1.0,
            ratio: 10.0,
        };
        assert_eq!(kind.default_severity(), AnomalySeverity::Warning);
    }

    #[test]
    fn test_default_severity_exploding_gradients_is_critical() {
        let kind = AnomalyKind::ExplodingGradients {
            norm: 2000.0,
            threshold: 1000.0,
        };
        assert_eq!(kind.default_severity(), AnomalySeverity::Critical);
    }

    #[test]
    fn test_default_severity_vanishing_gradients_is_warning() {
        let kind = AnomalyKind::VanishingGradients {
            norm: 1e-12,
            threshold: 1e-10,
        };
        assert_eq!(kind.default_severity(), AnomalySeverity::Warning);
    }

    #[test]
    fn test_default_severity_opacity_collapse_is_critical() {
        let kind = AnomalyKind::OpacityCollapse {
            mean_opacity: 0.0001,
            threshold: 0.001,
        };
        assert_eq!(kind.default_severity(), AnomalySeverity::Critical);
    }

    #[test]
    fn test_default_severity_mode_collapse_is_warning() {
        let kind = AnomalyKind::ModeCollapse {
            opacity_std: 1e-8,
            threshold: 1e-6,
        };
        assert_eq!(kind.default_severity(), AnomalySeverity::Warning);
    }

    #[test]
    fn test_default_severity_slow_convergence_is_info() {
        let kind = AnomalyKind::SlowConvergence {
            improvement_rate: 0.0,
            expected: 1e-4,
        };
        assert_eq!(kind.default_severity(), AnomalySeverity::Info);
    }

    // ── AnomalyKind::description ─────────────────────────────────────────────

    #[test]
    fn test_description_non_empty_for_all_variants() {
        let kinds: Vec<AnomalyKind> = vec![
            AnomalyKind::NanValues {
                n_nan: 1,
                location: "pos".to_string(),
            },
            AnomalyKind::InfValues {
                n_inf: 2,
                location: "scale".to_string(),
            },
            AnomalyKind::ExplodingGradients {
                norm: 2000.0,
                threshold: 1000.0,
            },
            AnomalyKind::VanishingGradients {
                norm: 1e-12,
                threshold: 1e-10,
            },
            AnomalyKind::LossSpike {
                current: 10.0,
                expected: 1.0,
                ratio: 10.0,
            },
            AnomalyKind::LossDivergence {
                steps_increasing: 50,
            },
            AnomalyKind::OpacityCollapse {
                mean_opacity: 0.0001,
                threshold: 0.001,
            },
            AnomalyKind::ScaleExplosion {
                max_scale: 20.0,
                threshold: 10.0,
            },
            AnomalyKind::PositionDrift {
                max_drift: 6.0,
                threshold: 5.0,
            },
            AnomalyKind::ModeCollapse {
                opacity_std: 1e-8,
                threshold: 1e-6,
            },
            AnomalyKind::GradientNanInf {
                location: "grads".to_string(),
            },
            AnomalyKind::SlowConvergence {
                improvement_rate: 0.0,
                expected: 1e-4,
            },
        ];
        for kind in &kinds {
            let desc = kind.description();
            assert!(!desc.is_empty(), "description empty for {:?}", kind);
        }
    }

    // ── AnomalyEvent ─────────────────────────────────────────────────────────

    #[test]
    fn test_anomaly_event_new_correct_fields() {
        let kind = AnomalyKind::NanValues {
            n_nan: 3,
            location: "positions".to_string(),
        };
        let event = AnomalyEvent::new(kind.clone(), 42);
        assert_eq!(event.step, 42);
        assert_eq!(event.severity, AnomalySeverity::Fatal);
        assert!(!event.message.is_empty());
        assert_eq!(event.kind, kind);
    }

    #[test]
    fn test_anomaly_event_is_fatal_only_for_fatal() {
        let fatal = AnomalyEvent::new(
            AnomalyKind::NanValues {
                n_nan: 1,
                location: "x".to_string(),
            },
            0,
        );
        let warn = AnomalyEvent::new(
            AnomalyKind::LossSpike {
                current: 10.0,
                expected: 1.0,
                ratio: 10.0,
            },
            0,
        );
        assert!(fatal.is_fatal());
        assert!(!warn.is_fatal());
    }

    #[test]
    fn test_anomaly_event_is_critical_or_above() {
        let fatal = AnomalyEvent::new(
            AnomalyKind::NanValues {
                n_nan: 1,
                location: "x".to_string(),
            },
            0,
        );
        let critical = AnomalyEvent::new(
            AnomalyKind::ExplodingGradients {
                norm: 2000.0,
                threshold: 1000.0,
            },
            0,
        );
        let warning = AnomalyEvent::new(
            AnomalyKind::LossSpike {
                current: 10.0,
                expected: 1.0,
                ratio: 10.0,
            },
            0,
        );
        let info = AnomalyEvent::new(
            AnomalyKind::SlowConvergence {
                improvement_rate: 0.0,
                expected: 1e-4,
            },
            0,
        );
        assert!(fatal.is_critical_or_above());
        assert!(critical.is_critical_or_above());
        assert!(!warning.is_critical_or_above());
        assert!(!info.is_critical_or_above());
    }

    // ── anom_check_numerical ─────────────────────────────────────────────────

    #[test]
    fn test_check_numerical_clean_values_empty() {
        let values = vec![1.0f32, 2.0, 3.0, -1.5];
        let events = anom_check_numerical(&values, "test");
        assert!(events.is_empty(), "expected no events for clean values");
    }

    #[test]
    fn test_check_numerical_one_nan_produces_event() {
        let values = vec![1.0f32, f32::NAN, 3.0];
        let events = anom_check_numerical(&values, "test");
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].kind,
            AnomalyKind::NanValues { n_nan: 1, .. }
        ));
    }

    #[test]
    fn test_check_numerical_one_inf_produces_event() {
        let values = vec![1.0f32, f32::INFINITY, 3.0];
        let events = anom_check_numerical(&values, "test");
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].kind,
            AnomalyKind::InfValues { n_inf: 1, .. }
        ));
    }

    #[test]
    fn test_check_numerical_mixed_produces_two_events() {
        let values = vec![f32::NAN, f32::INFINITY, 1.0];
        let events = anom_check_numerical(&values, "mixed");
        // One event for NaN, one for Inf.
        assert_eq!(events.len(), 2);
        let has_nan = events
            .iter()
            .any(|e| matches!(e.kind, AnomalyKind::NanValues { .. }));
        let has_inf = events
            .iter()
            .any(|e| matches!(e.kind, AnomalyKind::InfValues { .. }));
        assert!(has_nan);
        assert!(has_inf);
    }

    // ── anom_check_gradient_norm ─────────────────────────────────────────────

    #[test]
    fn test_gradient_norm_within_bounds_empty() {
        let thresholds = AnomalyThresholds::default();
        let events = anom_check_gradient_norm(1.0, 0, &thresholds);
        assert!(events.is_empty());
    }

    #[test]
    fn test_gradient_norm_above_max_exploding() {
        let thresholds = AnomalyThresholds {
            max_gradient_norm: 100.0,
            ..Default::default()
        };
        let events = anom_check_gradient_norm(2000.0, 5, &thresholds);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].kind,
            AnomalyKind::ExplodingGradients { .. }
        ));
        assert_eq!(events[0].severity, AnomalySeverity::Critical);
    }

    #[test]
    fn test_gradient_norm_below_min_vanishing() {
        let thresholds = AnomalyThresholds {
            min_gradient_norm: 1e-10,
            ..Default::default()
        };
        let events = anom_check_gradient_norm(1e-15, 3, &thresholds);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].kind,
            AnomalyKind::VanishingGradients { .. }
        ));
    }

    // ── anom_check_gradient_numerical ───────────────────────────────────────

    #[test]
    fn test_gradient_numerical_all_finite_empty() {
        let events = anom_check_gradient_numerical(&[1.0, 2.0, -3.0], "grads", 0);
        assert!(events.is_empty());
    }

    #[test]
    fn test_gradient_numerical_nan_produces_event() {
        let events = anom_check_gradient_numerical(&[1.0, f32::NAN], "grads", 0);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, AnomalyKind::GradientNanInf { .. }));
        assert_eq!(events[0].severity, AnomalySeverity::Fatal);
    }

    #[test]
    fn test_gradient_numerical_inf_produces_event() {
        let events = anom_check_gradient_numerical(&[1.0, f32::INFINITY], "grads", 0);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, AnomalyKind::GradientNanInf { .. }));
    }

    // ── anom_check_loss_spike ────────────────────────────────────────────────

    #[test]
    fn test_loss_spike_empty_history_no_spike() {
        let thresholds = AnomalyThresholds::default();
        let events = anom_check_loss_spike(100.0, &[], 0, &thresholds);
        assert!(events.is_empty());
    }

    #[test]
    fn test_loss_spike_single_sample_no_spike() {
        let thresholds = AnomalyThresholds::default();
        let events = anom_check_loss_spike(100.0, &[10.0], 0, &thresholds);
        assert!(events.is_empty());
    }

    #[test]
    fn test_loss_spike_ten_times_mean_triggers() {
        let thresholds = AnomalyThresholds {
            loss_spike_ratio: 5.0,
            ..Default::default()
        };
        let history = vec![1.0f32; 10]; // mean = 1.0
        let events = anom_check_loss_spike(100.0, &history, 1, &thresholds);
        // ratio = 100.0 / 1.0 = 100 > 5.0
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, AnomalyKind::LossSpike { .. }));
    }

    #[test]
    fn test_loss_spike_within_bounds_no_event() {
        let thresholds = AnomalyThresholds {
            loss_spike_ratio: 10.0,
            ..Default::default()
        };
        let history = vec![2.0f32; 10]; // mean = 2.0
        let events = anom_check_loss_spike(5.0, &history, 1, &thresholds);
        // ratio = 5/2 = 2.5 < 10 → no spike
        assert!(events.is_empty());
    }

    // ── anom_check_loss_divergence ───────────────────────────────────────────

    #[test]
    fn test_loss_divergence_no_history_no_event() {
        let thresholds = AnomalyThresholds::default();
        let events = anom_check_loss_divergence(&[], 0, &thresholds);
        assert!(events.is_empty());
    }

    #[test]
    fn test_loss_divergence_n_plus_one_increases_triggers() {
        let thresholds = AnomalyThresholds {
            loss_divergence_steps: 3,
            ..Default::default()
        };
        // 4 values in strictly increasing order → tail of length 4 = n+1 where n=3
        let history = vec![1.0f32, 2.0, 3.0, 4.0];
        let events = anom_check_loss_divergence(&history, 5, &thresholds);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].kind,
            AnomalyKind::LossDivergence {
                steps_increasing: 3
            }
        ));
    }

    #[test]
    fn test_loss_divergence_dip_no_trigger() {
        let thresholds = AnomalyThresholds {
            loss_divergence_steps: 3,
            ..Default::default()
        };
        // Last value dips, so not monotone.
        let history = vec![1.0f32, 2.0, 3.0, 2.5];
        let events = anom_check_loss_divergence(&history, 5, &thresholds);
        assert!(events.is_empty());
    }

    // ── anom_check_opacity_collapse ──────────────────────────────────────────

    #[test]
    fn test_opacity_collapse_mean_half_no_event() {
        let thresholds = AnomalyThresholds::default();
        let opacities = vec![0.5f32; 100];
        let events = anom_check_opacity_collapse(&opacities, 0, &thresholds);
        assert!(events.is_empty());
    }

    #[test]
    fn test_opacity_collapse_very_low_mean_triggers() {
        let thresholds = AnomalyThresholds {
            min_mean_opacity: 0.001,
            ..Default::default()
        };
        let opacities = vec![0.0001f32; 100];
        let events = anom_check_opacity_collapse(&opacities, 0, &thresholds);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].kind,
            AnomalyKind::OpacityCollapse { .. }
        ));
        assert_eq!(events[0].severity, AnomalySeverity::Critical);
    }

    // ── anom_check_mode_collapse ─────────────────────────────────────────────

    #[test]
    fn test_mode_collapse_zero_std_triggers() {
        let thresholds = AnomalyThresholds {
            min_opacity_std: 1e-6,
            ..Default::default()
        };
        let opacities = vec![0.5f32; 100]; // all same → std=0
        let events = anom_check_mode_collapse(&opacities, 0, &thresholds);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, AnomalyKind::ModeCollapse { .. }));
    }

    #[test]
    fn test_mode_collapse_high_std_no_event() {
        let thresholds = AnomalyThresholds {
            min_opacity_std: 1e-6,
            ..Default::default()
        };
        let opacities: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect(); // uniform 0..1
        let events = anom_check_mode_collapse(&opacities, 0, &thresholds);
        // std of uniform 0..1 is ~0.29, well above 1e-6
        assert!(events.is_empty());
    }

    // ── anom_check_scale_explosion ───────────────────────────────────────────

    #[test]
    fn test_scale_explosion_within_bounds_empty() {
        let thresholds = AnomalyThresholds {
            max_gaussian_scale: 10.0,
            ..Default::default()
        };
        let log_scales = vec![5.0f32, 8.0, 9.9];
        let events = anom_check_scale_explosion(&log_scales, 0, &thresholds);
        assert!(events.is_empty());
    }

    #[test]
    fn test_scale_explosion_large_log_scale_triggers() {
        let thresholds = AnomalyThresholds {
            max_gaussian_scale: 10.0,
            ..Default::default()
        };
        let log_scales = vec![5.0f32, 20.0, 3.0]; // 20 > 10 in log space
        let events = anom_check_scale_explosion(&log_scales, 1, &thresholds);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, AnomalyKind::ScaleExplosion { .. }));
    }

    // ── anom_check_position_drift ────────────────────────────────────────────

    #[test]
    fn test_position_drift_identical_no_event() {
        let thresholds = AnomalyThresholds::default();
        let pos = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = anom_check_position_drift(&pos, &pos, 0, &thresholds);
        assert!(events.is_empty());
    }

    #[test]
    fn test_position_drift_large_drift_triggers() {
        let thresholds = AnomalyThresholds {
            max_position_drift: 1.0,
            ..Default::default()
        };
        let curr = vec![100.0f32, 200.0, 300.0];
        let refs = vec![0.0f32, 0.0, 0.0];
        let events = anom_check_position_drift(&curr, &refs, 1, &thresholds);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, AnomalyKind::PositionDrift { .. }));
    }

    // ── anom_check_convergence ────────────────────────────────────────────────

    #[test]
    fn test_convergence_history_too_short_no_event() {
        let thresholds = AnomalyThresholds {
            slow_convergence_window: 100,
            slow_convergence_min_rate: 1e-4,
            ..Default::default()
        };
        // Only 5 values, need 100.
        let psnr = vec![20.0f32; 5];
        let events = anom_check_convergence(&psnr, 0, &thresholds);
        // Not enough data → no events (handled gracefully, not an error)
        assert!(events.is_empty());
    }

    #[test]
    fn test_convergence_improving_no_event() {
        let thresholds = AnomalyThresholds {
            slow_convergence_window: 5,
            slow_convergence_min_rate: 0.001,
            ..Default::default()
        };
        // 5 values improving at rate 1.0 per step → well above 0.001 threshold.
        let psnr = vec![20.0f32, 21.0, 22.0, 23.0, 24.0];
        let events = anom_check_convergence(&psnr, 10, &thresholds);
        assert!(events.is_empty());
    }

    #[test]
    fn test_convergence_flat_triggers_slow() {
        let thresholds = AnomalyThresholds {
            slow_convergence_window: 5,
            slow_convergence_min_rate: 0.01,
            ..Default::default()
        };
        // Flat PSNR → improvement = 0, rate = 0 < 0.01
        let psnr = vec![20.0f32; 5];
        let events = anom_check_convergence(&psnr, 10, &thresholds);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].kind,
            AnomalyKind::SlowConvergence { .. }
        ));
        assert_eq!(events[0].severity, AnomalySeverity::Info);
    }

    // ── anom_mean_std ─────────────────────────────────────────────────────────

    #[test]
    fn test_mean_std_single_value() {
        let (mean, std) = anom_mean_std(&[5.0f32]);
        assert!((mean - 5.0).abs() < 1e-5, "mean should be 5.0");
        assert!(std.abs() < 1e-5, "std should be 0 for single value");
    }

    #[test]
    fn test_mean_std_known_values() {
        // [2, 4, 4, 4, 5, 5, 7, 9] mean=5.0, population std=2.0
        let data = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = anom_mean_std(&data);
        assert!((mean - 5.0).abs() < 1e-4, "expected mean 5.0, got {}", mean);
        assert!((std - 2.0).abs() < 1e-4, "expected std 2.0, got {}", std);
    }

    #[test]
    fn test_mean_std_empty_returns_zeros() {
        let (mean, std) = anom_mean_std(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    // ── anom_count_nonfinite ──────────────────────────────────────────────────

    #[test]
    fn test_count_nonfinite_clean() {
        let (n_nan, n_inf) = anom_count_nonfinite(&[1.0, 2.0, 3.0]);
        assert_eq!(n_nan, 0);
        assert_eq!(n_inf, 0);
    }

    #[test]
    fn test_count_nonfinite_known_counts() {
        let data = vec![f32::NAN, 1.0, f32::INFINITY, f32::NAN, f32::NEG_INFINITY];
        let (n_nan, n_inf) = anom_count_nonfinite(&data);
        assert_eq!(n_nan, 2);
        assert_eq!(n_inf, 2);
    }

    // ── anom_is_monotone_increasing ──────────────────────────────────────────

    #[test]
    fn test_monotone_increasing_five_values() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!(anom_is_monotone_increasing(&values, 5));
    }

    #[test]
    fn test_monotone_increasing_one_dip_false() {
        let values = vec![1.0f32, 2.0, 3.0, 2.5, 5.0];
        assert!(!anom_is_monotone_increasing(&values, 5));
    }

    #[test]
    fn test_monotone_increasing_n_exceeds_length_false() {
        let values = vec![1.0f32, 2.0];
        assert!(!anom_is_monotone_increasing(&values, 5));
    }

    #[test]
    fn test_monotone_increasing_n_zero_false() {
        let values = vec![1.0f32, 2.0, 3.0];
        assert!(!anom_is_monotone_increasing(&values, 0));
    }

    // ── anom_l2_norm ──────────────────────────────────────────────────────────

    #[test]
    fn test_l2_norm_known_vector() {
        let v = vec![3.0f32, 4.0];
        assert!((anom_l2_norm(&v) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_norm_empty() {
        assert_eq!(anom_l2_norm(&[]), 0.0);
    }

    #[test]
    fn test_l2_norm_unit_vector() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((anom_l2_norm(&v) - 1.0).abs() < 1e-5);
    }

    // ── anom_max_abs ──────────────────────────────────────────────────────────

    #[test]
    fn test_max_abs_known_value() {
        let v = vec![-5.0f32, 3.0, -2.0, 4.9];
        assert!((anom_max_abs(&v) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_abs_empty() {
        assert_eq!(anom_max_abs(&[]), 0.0);
    }

    // ── anom_max_pairwise_dist ────────────────────────────────────────────────

    #[test]
    fn test_max_pairwise_dist_identical_zero() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = anom_max_pairwise_dist(&a, &a);
        match result {
            Ok(dist) => assert!(dist.abs() < 1e-5),
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_max_pairwise_dist_different_lengths_error() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0];
        let result = anom_max_pairwise_dist(&a, &b);
        assert!(matches!(
            result,
            Err(AnomalyDetectionError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_max_pairwise_dist_known_value() {
        // Two points: (0,0,0) and (3,4,0) → dist = 5.0
        let a = vec![0.0f32, 0.0, 0.0, 10.0, 10.0, 10.0];
        let b = vec![3.0f32, 4.0, 0.0, 10.0, 10.0, 10.0];
        let result = anom_max_pairwise_dist(&a, &b);
        match result {
            Ok(dist) => assert!((dist - 5.0).abs() < 1e-4, "expected 5.0, got {}", dist),
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    // ── AnomalyDetector ───────────────────────────────────────────────────────

    #[test]
    fn test_detector_new_starts_empty() {
        let detector = AnomalyDetector::new(AnomalyDetectorConfig::default());
        assert_eq!(detector.events().len(), 0);
        assert_eq!(detector.n_fatal(), 0);
        assert_eq!(detector.n_critical(), 0);
        assert_eq!(detector.n_warning(), 0);
        assert_eq!(detector.step(), 0);
    }

    #[test]
    fn test_detector_check_step_clean_no_events() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        let events = detector.check_step(Some(1.0), 0.5, Some(30.0), None, None, None);
        // Normal values → no anomaly events
        assert!(events.is_empty(), "expected no events for clean step");
    }

    #[test]
    fn test_detector_check_step_nan_loss_fatal_event() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        let events = detector.check_step(None, f32::NAN, None, None, None, None);
        let has_fatal = events.iter().any(|e| e.is_fatal());
        assert!(has_fatal, "NaN loss should produce a fatal event");
    }

    #[test]
    fn test_detector_should_pause_no_fatal_false() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            auto_pause_on_fatal: true,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        detector.check_step(Some(1.0), 0.5, None, None, None, None);
        assert!(!detector.should_pause());
    }

    #[test]
    fn test_detector_should_pause_with_fatal() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            auto_pause_on_fatal: true,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        detector.check_step(None, f32::NAN, None, None, None, None);
        assert!(
            detector.should_pause(),
            "should pause when fatal event occurred"
        );
    }

    #[test]
    fn test_detector_should_pause_false_when_disabled() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            auto_pause_on_fatal: false,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        detector.check_step(None, f32::NAN, None, None, None, None);
        // Even with a fatal event, should_pause is false because auto_pause_on_fatal=false
        assert!(!detector.should_pause());
    }

    #[test]
    fn test_detector_severity_counts_correct() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        // Inject a NaN loss → Fatal
        detector.check_step(None, f32::NAN, None, None, None, None);
        let counts = detector.severity_counts();
        // counts[3] = n_fatal
        assert!(counts[3] >= 1, "expected at least 1 fatal in counts");
    }

    #[test]
    fn test_detector_clear_events() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        detector.check_step(None, f32::NAN, None, None, None, None);
        assert!(!detector.events().is_empty());
        detector.clear_events();
        assert!(detector.events().is_empty(), "events should be cleared");
    }

    #[test]
    fn test_detector_recent_events_returns_last_n() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        // Generate some fatal events by repeatedly passing NaN loss.
        for _ in 0..10 {
            detector.check_step(None, f32::NAN, None, None, None, None);
            detector.advance_step();
        }
        let recent = detector.recent_events(3);
        assert!(recent.len() <= 3);
    }

    #[test]
    fn test_detector_advance_step() {
        let mut detector = AnomalyDetector::new(AnomalyDetectorConfig::default());
        assert_eq!(detector.step(), 0);
        detector.advance_step();
        assert_eq!(detector.step(), 1);
        detector.advance_step();
        assert_eq!(detector.step(), 2);
    }

    // ── anom_generate_report ──────────────────────────────────────────────────

    #[test]
    fn test_generate_report_correct_counts() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        detector.check_step(None, f32::NAN, None, None, None, None);
        let report = anom_generate_report(&detector);
        // Should have at least 1 fatal from NaN loss.
        assert!(
            report.n_fatal >= 1,
            "expected at least 1 fatal in report, got {}",
            report.n_fatal
        );
    }

    #[test]
    fn test_generate_report_clean_detector() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            ..Default::default()
        };
        let detector = AnomalyDetector::new(config);
        let report = anom_generate_report(&detector);
        assert_eq!(report.n_fatal, 0);
        assert_eq!(report.n_critical, 0);
        assert_eq!(report.n_warning, 0);
        assert!(report.most_severe.is_none());
    }

    // ── anom_format_event ─────────────────────────────────────────────────────

    #[test]
    fn test_format_event_non_empty_with_severity() {
        let event = AnomalyEvent::new(
            AnomalyKind::NanValues {
                n_nan: 2,
                location: "positions".to_string(),
            },
            100,
        );
        let s = anom_format_event(&event);
        assert!(!s.is_empty(), "format_event should return non-empty string");
        assert!(s.contains("FATAL"), "should contain severity label");
    }

    #[test]
    fn test_format_event_warning_label() {
        let event = AnomalyEvent::new(
            AnomalyKind::LossSpike {
                current: 10.0,
                expected: 1.0,
                ratio: 10.0,
            },
            50,
        );
        let s = anom_format_event(&event);
        assert!(
            s.contains("WARNING") || s.contains("WARNING"),
            "should contain WARNING"
        );
    }

    // ── anom_format_report ────────────────────────────────────────────────────

    #[test]
    fn test_format_report_non_empty_with_totals() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        detector.check_step(None, f32::NAN, None, None, None, None);
        let report = anom_generate_report(&detector);
        let s = anom_format_report(&report);
        assert!(
            !s.is_empty(),
            "format_report should return non-empty string"
        );
        assert!(s.contains("Fatal") || s.contains("fatal") || s.to_lowercase().contains("fatal"));
    }

    #[test]
    fn test_format_report_contains_steps() {
        let config = AnomalyDetectorConfig {
            check_interval: 1,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        for _ in 0..5 {
            detector.check_step(Some(1.0), 0.5, None, None, None, None);
            detector.advance_step();
        }
        let report = anom_generate_report(&detector);
        let s = anom_format_report(&report);
        assert!(s.contains("Steps") || s.contains("steps"));
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_check_numerical_empty_slice_no_events() {
        let events = anom_check_numerical(&[], "empty");
        assert!(events.is_empty());
    }

    #[test]
    fn test_detector_check_interval_skips_checks() {
        let config = AnomalyDetectorConfig {
            check_interval: 5, // only check every 5 steps
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        // Step 0 → checks run (0 % 5 == 0), but loss is clean
        detector.check_step(Some(1.0), 0.5, None, None, None, None);
        // Steps 1..4 → skipped even with NaN gradient (gradient norm NaN → skipped because of
        // is_finite check in anom_check_gradient_norm, and NaN loss would be checked but
        // check_interval gates the whole check)
        for i in 1..4 {
            detector.step = i;
            // Clean loss, no anomaly even if we pass NaN gradient_norm
            // (anom_check_gradient_norm returns empty for non-finite norm)
            let events = detector.check_step(Some(f32::NAN), 0.5, None, None, None, None);
            // Step 1,2,3 with interval=5: 1%5≠0,2%5≠0,3%5≠0 → empty
            assert!(events.is_empty(), "step {} should be skipped", i);
        }
    }

    #[test]
    fn test_anomaly_detection_error_types() {
        let e1 = AnomalyDetectionError::EmptyInput;
        let e2 = AnomalyDetectionError::InvalidThreshold("bad value".to_string());
        let e3 = AnomalyDetectionError::InvalidConfig("config err".to_string());
        let e4 = AnomalyDetectionError::HistoryTooShort {
            needed: 10,
            available: 3,
        };
        assert!(!e1.to_string().is_empty());
        assert!(!e2.to_string().is_empty());
        assert!(!e3.to_string().is_empty());
        assert!(!e4.to_string().is_empty());
    }
}
