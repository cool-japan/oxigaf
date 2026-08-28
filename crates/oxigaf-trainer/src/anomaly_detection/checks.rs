//! Stateless anomaly checks: one function per [`super::AnomalyKind`].

use super::stats::{
    anom_count_nonfinite, anom_increase_fraction, anom_max_pairwise_dist, anom_mean_std,
    anom_relative_trend,
};
use super::types::{AnomalyEvent, AnomalyKind, AnomalyThresholds};

// ─────────────────────────────────────────────────────────────────────────────
// Core detection functions
// ─────────────────────────────────────────────────────────────────────────────

/// Check for NaN/Inf values in a parameter array.
/// Returns events for each type of non-finite value found, attributed to
/// `step` (mirroring [`anom_check_gradient_numerical`]).
pub fn anom_check_numerical(values: &[f32], location: &str, step: usize) -> Vec<AnomalyEvent> {
    let (n_nan, n_inf) = anom_count_nonfinite(values);
    let mut events = Vec::new();
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

/// Check the recent loss window for divergence.
///
/// The window is the last `loss_divergence_steps + 1` losses, i.e.
/// `loss_divergence_steps` step-to-step intervals. It is flagged when **both**
/// robustness gates pass:
///
/// 1. at least [`AnomalyThresholds::loss_divergence_min_increase_fraction`] of
///    those intervals are increases ([`anom_increase_fraction`]), and
/// 2. the window's normalised least-squares slope
///    ([`anom_relative_trend`]) exceeds
///    [`AnomalyThresholds::loss_divergence_min_relative_trend`].
///
/// Gate 1 alone would let one late spike through; gate 2 alone would fire on a
/// window that merely wobbles upward. Together they survive the dips a
/// stochastic loss curve always has, which strict monotonicity
/// ([`super::anom_is_monotone_increasing`], the rule this check used to apply) does
/// not: a single down-tick anywhere in the window used to suppress the alarm
/// entirely.
///
/// Note the fraction gate is evaluated against the *actual* interval count, so
/// a short window is inherently stricter — with `loss_divergence_steps = 3`
/// there are 3 intervals and the default `0.75` still demands all three. At
/// least 4 intervals are needed before any dip is tolerated.
///
/// Returns no event when `loss_divergence_steps` is `0` (the check is
/// disabled), when there is not yet a full window of history, or when the
/// window contains a non-finite loss (NaN/Inf have their own checks).
pub fn anom_check_loss_divergence(
    loss_history: &[f32],
    step: usize,
    thresholds: &AnomalyThresholds,
) -> Vec<AnomalyEvent> {
    // N intervals need N+1 values.
    let n = thresholds.loss_divergence_steps + 1;
    if n < 2 || loss_history.len() < n {
        return Vec::new();
    }
    let tail = &loss_history[loss_history.len() - n..];
    if tail.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }

    if anom_increase_fraction(loss_history, n) < thresholds.loss_divergence_min_increase_fraction {
        return Vec::new();
    }
    if anom_relative_trend(loss_history, n) <= thresholds.loss_divergence_min_relative_trend {
        return Vec::new();
    }

    let steps_increasing = tail.windows(2).filter(|w| w[1] > w[0]).count();
    vec![AnomalyEvent::new(
        AnomalyKind::LossDivergence { steps_increasing },
        step,
    )]
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
        Err(e) => {
            // A length mismatch (typically after densification/pruning
            // changed the Gaussian count) is exactly the kind of drift the
            // monitor exists to catch — surface it instead of silently
            // reporting "no anomaly".
            let reason = e.to_string();
            return vec![AnomalyEvent::new(
                AnomalyKind::PositionDriftSkipped { reason },
                step,
            )];
        }
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
