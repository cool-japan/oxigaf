//! Unit tests for [`crate::anomaly_detection`].
//!
//! Split out of the parent module to keep every file under the 2000-line cap.

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
    let events = anom_check_numerical(&values, "test", 0);
    assert!(events.is_empty(), "expected no events for clean values");
}

#[test]
fn test_check_numerical_one_nan_produces_event() {
    let values = vec![1.0f32, f32::NAN, 3.0];
    let events = anom_check_numerical(&values, "test", 0);
    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0].kind,
        AnomalyKind::NanValues { n_nan: 1, .. }
    ));
}

#[test]
fn test_check_numerical_one_inf_produces_event() {
    let values = vec![1.0f32, f32::INFINITY, 3.0];
    let events = anom_check_numerical(&values, "test", 0);
    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0].kind,
        AnomalyKind::InfValues { n_inf: 1, .. }
    ));
}

#[test]
fn test_check_numerical_reports_the_given_step() {
    // Regression: previously hardcoded step 0 regardless of caller.
    let events = anom_check_numerical(&[f32::NAN], "loss", 12345);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].step, 12345);
    assert!(events[0].message.contains("12345"));
}

#[test]
fn test_check_numerical_mixed_produces_two_events() {
    let values = vec![f32::NAN, f32::INFINITY, 1.0];
    let events = anom_check_numerical(&values, "mixed", 0);
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

#[test]
fn test_position_drift_length_mismatch_reports_event() {
    // Regression: a length mismatch (e.g. after densification/pruning
    // changed the Gaussian count) must surface as a visible event, not
    // silently report "no anomaly".
    let thresholds = AnomalyThresholds::default();
    let curr = vec![1.0f32, 2.0, 3.0];
    let refs = vec![1.0f32, 2.0];
    let events = anom_check_position_drift(&curr, &refs, 7, &thresholds);
    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0].kind,
        AnomalyKind::PositionDriftSkipped { .. }
    ));
    assert_eq!(events[0].severity, AnomalySeverity::Warning);
    assert_eq!(events[0].step, 7);
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

#[test]
fn test_generate_report_counters_survive_event_history_truncation() {
    // Regression: report counters must match the detector's monotone
    // counters even after `self.events` is truncated to `max_history`,
    // not under-report by recounting only the truncated buffer.
    let config = AnomalyDetectorConfig {
        check_interval: 1,
        max_history: 2,
        ..Default::default()
    };
    let mut detector = AnomalyDetector::new(config);
    for i in 0..5 {
        detector.check_step(None, f32::NAN, None, None, None, None);
        detector.step = i + 1;
    }
    assert_eq!(detector.events().len(), 2); // truncated buffer
    let report = anom_generate_report(&detector);
    assert_eq!(report.n_fatal, 5, "report.n_fatal={}", report.n_fatal); // full count
}

#[test]
fn test_generate_report_anomaly_rate_uses_checked_steps_not_elapsed_steps() {
    // Regression: `anomaly_rate` is documented as "per 100 steps
    // checked" but was previously computed against total elapsed
    // steps (off by `check_interval`), and `n_steps_checked` missed
    // the step-0 check.
    let config = AnomalyDetectorConfig {
        check_interval: 10,
        ..Default::default()
    };
    let mut detector = AnomalyDetector::new(config);
    detector.check_step(None, f32::NAN, None, None, None, None); // step 0: fatal
    detector.step = 10;
    detector.check_step(Some(1.0), 0.5, None, None, None, None); // step 10: clean

    let report = anom_generate_report(&detector);
    assert_eq!(report.n_steps_checked, 2, "n_steps_checked mismatch");
    // 1 fatal / 2 checked steps → 50 per 100 steps checked.
    assert!(
        (report.anomaly_rate - 50.0).abs() < 1e-3,
        "anomaly_rate={}",
        report.anomaly_rate
    );
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
    let events = anom_check_numerical(&[], "empty", 0);
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

// ── History sizing regression: psnr/loss history was hard-capped at
// 200 regardless of the configured window, so windows above ~200
// (default 1000) could never accumulate enough history to fire.

#[test]
fn test_slow_convergence_fires_with_window_above_old_200_cap() {
    let config = AnomalyDetectorConfig {
        check_interval: 1,
        thresholds: AnomalyThresholds {
            slow_convergence_window: 300, // > the old hardcoded 200 cap
            slow_convergence_min_rate: 1e-4,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut detector = AnomalyDetector::new(config);
    let mut fired = false;
    for i in 0..350 {
        detector.step = i;
        let events = detector.check_step(Some(1.0), 0.1, Some(20.0), None, None, None);
        fired |= events
            .iter()
            .any(|e| matches!(e.kind, AnomalyKind::SlowConvergence { .. }));
    }
    assert!(fired, "SlowConvergence never fired with a 300-step window");
}

#[test]
fn test_loss_divergence_fires_with_steps_above_old_200_cap() {
    let config = AnomalyDetectorConfig {
        check_interval: 1,
        thresholds: AnomalyThresholds {
            loss_divergence_steps: 250, // > the old hardcoded 200 cap
            ..Default::default()
        },
        ..Default::default()
    };
    let mut detector = AnomalyDetector::new(config);
    let mut fired = false;
    for i in 0..300 {
        detector.step = i;
        let events = detector.check_step(Some(1.0), i as f32 * 0.01 + 0.01, None, None, None, None);
        fired |= events
            .iter()
            .any(|e| matches!(e.kind, AnomalyKind::LossDivergence { .. }));
    }
    assert!(fired, "LossDivergence never fired with 250 required steps");
}

// ── Regression (F274): the divergence detector is robust to a single dip ────
// It used to require STRICT monotonicity over the whole window
// (`anom_is_monotone_increasing`), so one down-tick in a noisy loss curve
// suppressed the alarm entirely.

#[test]
fn test_increase_fraction_counts_upticks() {
    // 1 -> 2 -> 3 -> 2.5 -> 4: 3 of 4 intervals rise.
    let history = [1.0f32, 2.0, 3.0, 2.5, 4.0];
    let frac = anom_increase_fraction(&history, 5);
    assert!((frac - 0.75).abs() < 1e-6, "fraction was {frac}");
    // Strictly increasing -> 1.0
    assert!((anom_increase_fraction(&[1.0f32, 2.0, 3.0], 3) - 1.0).abs() < 1e-6);
    // Degenerate windows are not "increasing".
    assert_eq!(anom_increase_fraction(&[1.0f32], 1), 0.0);
    assert_eq!(anom_increase_fraction(&[1.0f32, 2.0], 9), 0.0);
    // Non-finite values disqualify the window.
    assert_eq!(anom_increase_fraction(&[1.0f32, f32::NAN, 3.0], 3), 0.0);
}

#[test]
fn test_relative_trend_is_scale_free() {
    // Two windows with the same *relative* growth but 1000x different scale.
    let small = [1.0f32, 1.1, 1.2, 1.3];
    let large = [1000.0f32, 1100.0, 1200.0, 1300.0];
    let t_small = anom_relative_trend(&small, 4);
    let t_large = anom_relative_trend(&large, 4);
    assert!(t_small > 0.0, "rising window must have a positive trend");
    assert!(
        (t_small - t_large).abs() < 1e-4,
        "trend must be scale-free: {t_small} vs {t_large}"
    );
    // Falling window -> negative trend.
    let falling = [4.0f32, 3.0, 2.0, 1.0];
    assert!(anom_relative_trend(&falling, 4) < 0.0);
    // Flat window -> no trend.
    assert!(anom_relative_trend(&[2.0f32; 8], 8).abs() < 1e-6);
    // Non-finite / degenerate windows.
    assert_eq!(anom_relative_trend(&[1.0f32, f32::INFINITY], 2), 0.0);
    assert_eq!(anom_relative_trend(&[1.0f32, 2.0], 1), 0.0);
}

#[test]
fn test_loss_divergence_fires_despite_a_single_dip() {
    let thresholds = AnomalyThresholds {
        loss_divergence_steps: 10,
        ..Default::default()
    };
    // 11 values: a clearly diverging run with one down-tick in the middle.
    // The old strictly-monotone rule would have stayed silent here.
    let history = vec![1.0f32, 1.2, 1.45, 1.7, 1.6, 1.9, 2.2, 2.6, 3.0, 3.5, 4.1];
    assert!(
        !anom_is_monotone_increasing(&history, 11),
        "the fixture must contain a dip, otherwise it proves nothing"
    );
    let events = anom_check_loss_divergence(&history, 42, &thresholds);
    assert_eq!(events.len(), 1, "one dip must not suppress the alarm");
    match events[0].kind {
        AnomalyKind::LossDivergence { steps_increasing } => {
            assert_eq!(steps_increasing, 9, "9 of 10 intervals rise");
        }
        ref other => panic!("expected LossDivergence, got {other:?}"),
    }
    assert_eq!(events[0].step, 42);
}

#[test]
fn test_loss_divergence_silent_on_noisy_flat_loss() {
    let thresholds = AnomalyThresholds {
        loss_divergence_steps: 10,
        ..Default::default()
    };
    // A converged-ish loss jittering around 0.5 with no real trend: the
    // intervals alternate up/down, so the up-tick gate rejects it.
    let history: Vec<f32> = vec![
        0.50, 0.51, 0.50, 0.51, 0.50, 0.51, 0.50, 0.51, 0.50, 0.51, 0.50, 0.51,
    ];
    assert!(anom_check_loss_divergence(&history, 0, &thresholds).is_empty());
}

#[test]
fn test_loss_divergence_silent_on_negligible_relative_growth() {
    let thresholds = AnomalyThresholds {
        loss_divergence_steps: 10,
        ..Default::default()
    };
    // Monotone, so gate 1 passes, but the loss climbs by 1e-6 per step on a
    // level of ~1.0 — five orders of magnitude below the 1e-3 relative-trend
    // floor. Numerical drift, not divergence.
    let history: Vec<f32> = (0..11).map(|i| 1.0 + i as f32 * 1e-6).collect();
    assert!(anom_is_monotone_increasing(&history, 11));
    assert!(
        anom_check_loss_divergence(&history, 0, &thresholds).is_empty(),
        "insignificant growth must not be reported as divergence"
    );
}

#[test]
fn test_loss_divergence_silent_on_late_spike_only() {
    let thresholds = AnomalyThresholds {
        loss_divergence_steps: 10,
        ..Default::default()
    };
    // Flat then one huge jump: the slope is positive but only 1 of 10
    // intervals rises, so this is a LossSpike, not a divergence.
    let mut history = vec![1.0f32; 10];
    history.push(50.0);
    assert!(anom_check_loss_divergence(&history, 0, &thresholds).is_empty());
}

#[test]
fn test_loss_divergence_disabled_at_zero_steps() {
    let thresholds = AnomalyThresholds {
        loss_divergence_steps: 0,
        ..Default::default()
    };
    let history: Vec<f32> = (0..20).map(|i| 1.0 + i as f32).collect();
    assert!(
        anom_check_loss_divergence(&history, 0, &thresholds).is_empty(),
        "a zero-width window disables the check instead of firing on everything"
    );
}

#[test]
fn test_loss_divergence_ignores_non_finite_window() {
    let thresholds = AnomalyThresholds {
        loss_divergence_steps: 3,
        ..Default::default()
    };
    let history = vec![1.0f32, 2.0, f32::NAN, 4.0];
    assert!(
        anom_check_loss_divergence(&history, 0, &thresholds).is_empty(),
        "NaN losses are reported by the numerical checks, not this one"
    );
}
