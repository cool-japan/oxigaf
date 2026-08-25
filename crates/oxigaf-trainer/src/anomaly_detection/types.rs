//! Anomaly taxonomy: error type, severity levels, anomaly kinds, events
//! and the tunable detection thresholds.

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
    /// Loss has been consistently increasing over the recent window.
    ///
    /// `steps_increasing` counts the step-to-step *increases* inside that
    /// window; the window may contain a few dips and still trigger (see
    /// [`crate::anomaly_detection::anom_check_loss_divergence`]).
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
    /// The position-drift check could not run because the current and
    /// reference position buffers had different lengths (commonly caused by
    /// a densification/pruning pass changing the Gaussian count).
    PositionDriftSkipped { reason: String },
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
            Self::PositionDriftSkipped { .. } => AnomalySeverity::Warning,
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
                    "Loss divergence: {} increasing steps in the recent window",
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
            Self::PositionDriftSkipped { reason } => {
                format!("Position drift check skipped: {reason}")
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
    /// Width of the loss window examined for divergence, in *intervals*: the
    /// check reads the last `loss_divergence_steps + 1` losses. `0` disables
    /// the check.
    pub loss_divergence_steps: usize,
    /// Fraction of the divergence window's step-to-step intervals that must
    /// be increases before LossDivergence can fire.
    ///
    /// `1.0` reproduces the old strictly-monotone rule, which a single dip in
    /// a noisy loss curve defeats. The default `0.75` tolerates a minority of
    /// dips. Note the window must have at least 4 intervals (i.e.
    /// `loss_divergence_steps >= 4`) for `0.75` to permit *any* dip.
    pub loss_divergence_min_increase_fraction: f32,
    /// Minimum least-squares slope across the divergence window, normalised
    /// by the window's mean loss magnitude (so it reads as "relative growth
    /// per step"), before LossDivergence can fire.
    ///
    /// Guards against a window that ticks up by numerically insignificant
    /// amounts. The default `1e-3` is 0.1 % of the current loss level per
    /// step.
    pub loss_divergence_min_relative_trend: f32,
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
            // 50 consecutive strictly-increasing stochastic loss steps is a
            // practically unreachable event; 10 is still robust to noise
            // but actually fires before training has fully diverged.
            loss_divergence_steps: 10,
            // 10 intervals, so at the default window up to 2 dips are
            // tolerated (8/10 = 0.8 >= 0.75) — a stochastic loss curve that
            // is genuinely diverging is rarely monotone.
            loss_divergence_min_increase_fraction: 0.75,
            loss_divergence_min_relative_trend: 1e-3,
            min_mean_opacity: 0.001,
            // A head-sized scene's genuinely exploded Gaussians sit around
            // log-scale 1–3 (real-space 3–20). The previous default of
            // 10.0 (real ≈ 22 026) was far too loose to ever trigger before
            // the render had already been destroyed.
            max_gaussian_scale: 3.0,
            max_position_drift: 5.0,
            min_opacity_std: 1e-6,
            slow_convergence_window: 1000,
            slow_convergence_min_rate: 1e-4,
        }
    }
}
