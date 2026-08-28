//! The stateful [`AnomalyDetector`] and its configuration.

use super::checks::{
    anom_check_convergence, anom_check_gradient_norm, anom_check_loss_divergence,
    anom_check_loss_spike, anom_check_mode_collapse, anom_check_numerical,
    anom_check_opacity_collapse, anom_check_position_drift, anom_check_scale_explosion,
};
use super::types::{AnomalyEvent, AnomalySeverity, AnomalyThresholds};

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
    /// Capacity for `loss_history`/`psnr_history`, sized to cover every
    /// configured threshold that reads from them (see [`AnomalyDetector::new`]).
    history_cap: usize,
    /// Current training step. `pub(super)` so the module's own tests can drive
    /// the detector to a specific step; it stays private to outside callers,
    /// which read it through [`AnomalyDetector::step`].
    pub(super) step: usize,
    /// Number of steps at which checks actually ran (`should_check` was
    /// true, i.e. `step % check_interval == 0`).
    checks_performed: usize,
    n_fatal: usize,
    n_critical: usize,
    n_warning: usize,
    n_info: usize,
}

impl AnomalyDetector {
    /// Create a new detector with the given configuration.
    ///
    /// `loss_history`/`psnr_history` are sized to cover every threshold
    /// that reads from them, so a large `slow_convergence_window` or
    /// `loss_divergence_steps` is never starved of history by a fixed cap.
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        let history_cap = 200usize
            .max(config.thresholds.slow_convergence_window)
            .max(config.thresholds.loss_divergence_steps + 1);
        Self {
            config,
            events: Vec::new(),
            loss_history: Vec::new(),
            psnr_history: Vec::new(),
            history_cap,
            step: 0,
            checks_performed: 0,
            n_fatal: 0,
            n_critical: 0,
            n_warning: 0,
            n_info: 0,
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
        if self.loss_history.len() > self.history_cap {
            let drain_to = self.loss_history.len() - self.history_cap;
            self.loss_history.drain(0..drain_to);
        }
        if let Some(p) = psnr {
            self.psnr_history.push(p);
            if self.psnr_history.len() > self.history_cap {
                let drain_to = self.psnr_history.len() - self.history_cap;
                self.psnr_history.drain(0..drain_to);
            }
        }

        if !should_check {
            return Vec::new();
        }
        self.checks_performed += 1;

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
        new_events.extend(anom_check_numerical(&[loss], "loss", step));
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

        // Monotone counters (unlike `self.events`, never truncated by
        // `max_history`) — the source of truth for whole-lifetime reports.
        for event in &new_events {
            match event.severity {
                AnomalySeverity::Fatal => self.n_fatal += 1,
                AnomalySeverity::Critical => self.n_critical += 1,
                AnomalySeverity::Warning => self.n_warning += 1,
                AnomalySeverity::Info => self.n_info += 1,
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

    /// Number of Info events observed.
    pub fn n_info(&self) -> usize {
        self.n_info
    }

    /// Number of steps at which checks actually ran (`step % check_interval
    /// == 0`), i.e. the denominator [`super::AnomalyReport::anomaly_rate`] is
    /// documented against.
    pub fn checks_performed(&self) -> usize {
        self.checks_performed
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
