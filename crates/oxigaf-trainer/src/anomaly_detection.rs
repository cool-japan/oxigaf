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

mod checks;
mod detector;
mod report;
mod stats;
mod types;

#[cfg(test)]
mod tests;

pub use checks::{
    anom_check_convergence, anom_check_gradient_norm, anom_check_gradient_numerical,
    anom_check_loss_divergence, anom_check_loss_spike, anom_check_mode_collapse,
    anom_check_numerical, anom_check_opacity_collapse, anom_check_position_drift,
    anom_check_scale_explosion,
};
pub use detector::{AnomalyDetector, AnomalyDetectorConfig};
pub use report::{anom_format_event, anom_format_report, anom_generate_report, AnomalyReport};
pub use stats::{
    anom_count_nonfinite, anom_increase_fraction, anom_is_monotone_increasing, anom_l2_norm,
    anom_max_abs, anom_max_pairwise_dist, anom_mean_std, anom_relative_trend,
};
pub use types::{
    AnomalyDetectionError, AnomalyEvent, AnomalyKind, AnomalySeverity, AnomalyThresholds,
};
