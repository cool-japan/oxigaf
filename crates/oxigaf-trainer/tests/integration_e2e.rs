//! End-to-end integration test scaffold for `oxigaf-trainer`.
//!
//! These tests wire together multiple trainer components — [`TrainingProfiler`],
//! [`LossScaler`], [`MixedPrecisionTrainer`], [`GradientNormTracker`],
//! [`LossTracker`], [`EmaTracker`], [`DensityStats`], and
//! [`TrainingProfileConfig`] — without requiring GPU access, real model weights,
//! or file I/O beyond temporary directories.
//!
//! All tests are synchronous (`fn`, not `async`).

use oxigaf_trainer::{
    DensityStats, EmaTracker, GradientNormTracker, LossScaler, LossTracker, MixedPrecisionTrainer,
    TrainingDiagnostics, TrainingPhase, TrainingProfile, TrainingProfiler,
};

// ============================================================================
// Test 1: TrainingProfiler + LossScaler combined timing scenario
// ============================================================================

/// Verify that the LossScaler's step() round-trip does not corrupt timing data
/// recorded by the TrainingProfiler in the same simulated training iteration.
#[test]
fn test_profiler_and_loss_scaler_independent_state() {
    let profiler = TrainingProfiler::new(true);
    let mut scaler = LossScaler::new(1024.0);

    // Simulate 10 "training steps" — record forward + backward phase timing,
    // and update the loss scaler as if the backward pass produced valid grads.
    for i in 0..10u64 {
        let duration_us = 500 + i * 50; // monotonically increasing work
        profiler.record(TrainingPhase::Forward, duration_us);
        profiler.record(TrainingPhase::Backward, duration_us * 2);

        // Scaler step: no overflow, so scale should grow after `scale_window`
        scaler.update(false);
    }

    let fwd = profiler
        .stats(TrainingPhase::Forward)
        .expect("forward phase must have stats");
    assert_eq!(fwd.count, 10, "exactly 10 forward recordings");
    assert!(fwd.total_us > 0, "total forward time must be positive");

    let bwd = profiler
        .stats(TrainingPhase::Backward)
        .expect("backward phase must have stats");
    assert_eq!(bwd.count, 10);
    assert!(bwd.total_us > fwd.total_us, "backward > forward total");

    // Scaler stats: 10 successes, no overflows
    let stats = scaler.stats();
    assert_eq!(stats.total_steps, 10);
    assert_eq!(stats.overflow_count, 0);
    assert!((stats.overflow_rate - 0.0).abs() < 1e-12);
}

// ============================================================================
// Test 2: TrainingProfiler timing is non-negative
// ============================================================================

/// Wall-clock durations recorded by `time()` must never be negative.
#[test]
fn test_profiler_time_duration_nonnegative() {
    let profiler = TrainingProfiler::new(true);

    for _ in 0..50 {
        profiler.time(TrainingPhase::LossComputation, || {
            // No-op closure — tests that even near-zero durations are >= 0.
            let _x: u32 = 1 + 1;
        });
    }

    let stats = profiler
        .stats(TrainingPhase::LossComputation)
        .expect("loss computation must have stats");

    // All u64 values are inherently non-negative; verify count and min/max order.
    assert_eq!(stats.count, 50);
    assert!(stats.min_us <= stats.max_us, "min must be <= max");
    assert!(stats.total_us < u64::MAX, "total must not overflow");
}

// ============================================================================
// Test 3: LossScaler scale + unscale round-trip
// ============================================================================

/// Scaling then unscaling must recover the original gradient values within
/// floating-point epsilon.
#[test]
fn test_loss_scaler_scale_unscale_roundtrip() {
    let scale = 512.0_f32;
    let scaler = LossScaler::new(scale);

    let original: Vec<f32> = vec![
        0.0,
        0.001,
        -0.5,
        std::f32::consts::PI,
        1e-6,
        -1e6,
        0.123_456,
    ];
    let mut grads = original.clone();

    scaler.scale_gradients(&mut grads);
    // After scaling every value must have been multiplied by `scale`.
    for (g, o) in grads.iter().zip(original.iter()) {
        assert!(
            (g - o * scale).abs() < 1e-3,
            "scaled value mismatch: {g} != {} * {scale}",
            o
        );
    }

    scaler.unscale_gradients(&mut grads);
    // After unscaling must recover original values.
    for (g, o) in grads.iter().zip(original.iter()) {
        assert!(
            (g - o).abs() < 1e-3,
            "round-trip failed: recovered {g}, original {o}"
        );
    }
}

// ============================================================================
// Test 4: LossScaler overflow detection and scale adaptation
// ============================================================================

/// Simulates a sequence of overflow → scale-halve → recovery steps.
#[test]
fn test_loss_scaler_overflow_then_recovery() {
    let mut scaler = LossScaler::new(1024.0).with_scale_window(5);

    // Trigger two overflows; scale should halve each time.
    scaler.update(true); // 1024 / 2 = 512
    scaler.update(true); // 512  / 2 = 256

    assert!(
        (scaler.stats().current_scale - 256.0).abs() < 1e-6,
        "scale must be 256 after two overflows"
    );

    // Five consecutive successes → scale doubles (256 → 512).
    for _ in 0..5 {
        scaler.update(false);
    }

    assert!(
        (scaler.stats().current_scale - 512.0).abs() < 1e-6,
        "scale must double to 512 after window successes"
    );
    assert_eq!(scaler.stats().overflow_count, 2);
    assert_eq!(scaler.stats().total_steps, 7);
}

// ============================================================================
// Test 5: GradientNormTracker — spike detection via integration
// ============================================================================

/// Build a realistic gradient-norm history (stable → spike) and verify that
/// `has_gradient_spike` correctly flags only the spike iteration.
#[test]
fn test_gradient_norm_tracker_spike_detection_integration() {
    let mut tracker = GradientNormTracker::new(50);

    // Stable phase: 20 recordings near 0.01.
    for _ in 0..20 {
        tracker.record("position", 0.01);
        tracker.record("rotation", 0.005);
    }

    // Both groups should report no spike.
    assert!(!tracker.has_gradient_spike("position"));
    assert!(!tracker.has_gradient_spike("rotation"));

    // Inject a spike into "position".
    tracker.record("position", 5.0); // >> 3 × 0.01

    assert!(
        tracker.has_gradient_spike("position"),
        "position should show a spike"
    );
    assert!(
        !tracker.has_gradient_spike("rotation"),
        "rotation should still be stable"
    );

    // Verify table formatting includes both groups.
    let table = tracker.format_table();
    assert!(table.contains("position"));
    assert!(table.contains("rotation"));
}

// ============================================================================
// Test 6: GradientNormTracker — window eviction
// ============================================================================

/// When more entries are added than the window capacity, old values should be
/// evicted and statistics should reflect only recent entries.
#[test]
fn test_gradient_norm_tracker_window_eviction() {
    let window = 10;
    let mut tracker = GradientNormTracker::new(window);

    // Fill window with value 1.0.
    for _ in 0..window {
        tracker.record("sh", 1.0);
    }

    // Add 5 more with value 10.0 — evicts 5 of the old entries.
    for _ in 0..5 {
        tracker.record("sh", 10.0);
    }

    let entry = tracker.norms.get("sh").expect("sh must be present");
    assert_eq!(entry.len(), window, "window must stay capped at {window}");

    let mean = tracker.mean_norm("sh").expect("mean must exist");
    // 5 entries at 1.0, 5 entries at 10.0 → mean = 5.5
    assert!((mean - 5.5).abs() < 1e-5, "mean should be 5.5, got {mean}");
}

// ============================================================================
// Test 7: LossTracker — convergence detection integration
// ============================================================================

/// Feed a decreasing loss sequence and verify that `is_converging` returns
/// true; feed an increasing sequence and verify it returns false.
#[test]
fn test_loss_tracker_convergence_integration() {
    let mut tracker = LossTracker::new(0.1, 20);

    // Decreasing sequence.
    for i in (0..20usize).rev() {
        tracker.record_total(i as f32 * 0.1 + 0.01);
    }
    assert!(tracker.is_converging(), "decreasing loss must converge");

    // Reset via a new tracker.
    let mut tracker2 = LossTracker::new(0.1, 20);
    // Increasing sequence.
    for i in 0..20usize {
        tracker2.record_total(i as f32 * 0.1 + 0.01);
    }
    assert!(
        !tracker2.is_converging(),
        "increasing loss must not converge"
    );
}

// ============================================================================
// Test 8: EmaTracker + TrainingProfiler combined simulation
// ============================================================================

/// Simulate 100 training iterations: track a loss via EmaTracker and record
/// forward-pass timing via TrainingProfiler.  Verify both produce self-
/// consistent results after the full loop.
#[test]
fn test_ema_tracker_and_profiler_combined_simulation() {
    let profiler = TrainingProfiler::new(true);
    let mut ema = EmaTracker::new(0.05);

    for step in 0..100u32 {
        // Simulated decreasing loss.
        let raw_loss = 1.0 / (1.0 + step as f32 * 0.1);
        ema.update(raw_loss);

        // Simulated constant 1 ms forward pass.
        profiler.record(TrainingPhase::Forward, 1_000);
    }

    assert_eq!(
        ema.update_count, 100,
        "EMA must have been updated 100 times"
    );
    let smoothed = ema.smoothed();
    assert!(
        smoothed > 0.0 && smoothed < 1.0,
        "smoothed loss must be in (0, 1), got {smoothed}"
    );

    let fwd = profiler
        .stats(TrainingPhase::Forward)
        .expect("forward stats must exist");
    assert_eq!(fwd.count, 100);
    assert_eq!(fwd.total_us, 100_000, "100 × 1 ms = 100 000 µs");
}

// ============================================================================
// Test 9: TrainingDiagnostics — integrated report contains all sections
// ============================================================================

/// Populate all sections of [`TrainingDiagnostics`] and verify that
/// `format_report` includes losses, gradients, and density events.
#[test]
fn test_training_diagnostics_full_report() {
    let mut diag = TrainingDiagnostics::new();

    for i in 0..30u64 {
        diag.next_iteration();
        diag.record_losses(&[("l1", 0.5 - i as f32 * 0.01), ("ssim", 0.2)]);
        diag.record_grad_norms(&[("position", 0.01), ("rotation", 0.005)]);
    }

    diag.record_density_event(DensityStats {
        iteration: 25,
        num_gaussians_before: 1_000,
        num_gaussians_after: 1_200,
        num_cloned: 200,
        num_split: 0,
        num_pruned: 0,
        num_opacity_reset: 50,
    });

    let report = diag.format_report();
    assert!(
        report.contains("Iteration"),
        "report must mention iteration"
    );
    assert!(report.contains("Losses"), "report must have losses section");
    assert!(
        report.contains("Gradient"),
        "report must have gradient section"
    );
    assert!(
        report.contains("Density"),
        "report must have density section"
    );
    assert!(
        report.contains("1000"),
        "report must include gaussian count before"
    );
}

// ============================================================================
// Test 10: MixedPrecisionTrainer — overflow does not corrupt gradient slice
// ============================================================================

/// When `step()` detects overflow the gradient slice should have been
/// unscaled (values divided by scale) even though the step returns false.
/// This prevents stale scaled gradients from accumulating.
#[test]
fn test_mixed_precision_trainer_overflow_unscales_grads() {
    let mut trainer = MixedPrecisionTrainer::float16(); // scale = 65536.0

    // Deliberately inject NaN to trigger overflow.
    let mut grads = vec![1.0_f32, f32::NAN, 3.0];
    let should_step = trainer.step(&mut grads);

    // Step must return false because of NaN.
    assert!(!should_step, "step must return false on overflow");

    // The non-NaN values must have been unscaled (divided by 65536).
    let expected0 = 1.0 / 65536.0;
    assert!(
        (grads[0] - expected0).abs() < 1e-7,
        "grads[0] must be unscaled: expected {expected0}, got {}",
        grads[0]
    );

    // The scaler must record the overflow.
    assert_eq!(trainer.scaler.stats().overflow_count, 1);
}

// ============================================================================
// Test 11: MixedPrecisionTrainer — clean step updates scaler correctly
// ============================================================================

#[test]
fn test_mixed_precision_trainer_clean_step_increments_success() {
    let mut trainer = MixedPrecisionTrainer::bfloat16(); // scale = 1024.0

    let mut grads: Vec<f32> = (0..64).map(|i| i as f32 * 0.001).collect();
    let should_step = trainer.step(&mut grads);

    assert!(should_step, "clean gradients must allow optimizer step");
    assert_eq!(trainer.scaler.stats().overflow_count, 0);
    assert_eq!(trainer.scaler.stats().total_steps, 1);

    // Gradients must have been unscaled.
    let expected_0 = 0.0_f32 / 1024.0;
    assert!(
        (grads[0] - expected_0).abs() < 1e-7,
        "grads[0] must be unscaled: {}",
        grads[0]
    );
    let expected_1 = 1.0_f32 * 0.001 / 1024.0;
    assert!(
        (grads[1] - expected_1).abs() < 1e-7,
        "grads[1] must be unscaled: {} (expected {})",
        grads[1],
        expected_1
    );
}

// ============================================================================
// Test 12: DensityStats — tracking changes through a simulated density loop
// ============================================================================

/// Simulate a short sequence of density-control operations and verify that
/// the cumulative statistics are consistent.
#[test]
fn test_density_stats_simulated_density_loop() {
    let mut history: Vec<DensityStats> = Vec::new();

    let mut n_gaussians = 1_000usize;

    // Step 1: clone 200
    let cloned = 200;
    let before = n_gaussians;
    n_gaussians += cloned;
    history.push(DensityStats {
        iteration: 500,
        num_gaussians_before: before,
        num_gaussians_after: n_gaussians,
        num_cloned: cloned,
        ..Default::default()
    });

    // Step 2: split 100, prune 50
    let split = 100;
    let pruned = 50;
    let before = n_gaussians;
    n_gaussians = n_gaussians + split - pruned;
    history.push(DensityStats {
        iteration: 1_000,
        num_gaussians_before: before,
        num_gaussians_after: n_gaussians,
        num_split: split,
        num_pruned: pruned,
        ..Default::default()
    });

    assert_eq!(history.len(), 2);
    assert_eq!(history[0].net_change(), 200);
    assert_eq!(history[1].net_change(), 50);
    assert!(history[0].was_densification());
    assert!(history[1].was_densification());
    assert!(!history[0].was_pruning());
    assert!(history[1].was_pruning());
    assert_eq!(n_gaussians, 1_250);
}

// ============================================================================
// Test 13: TrainingProfileConfig lifecycle — create → validate → estimate → format
// ============================================================================

#[test]
fn test_training_profile_config_lifecycle() {
    // 1. Create from preset.
    let cfg = TrainingProfile::standard().config();

    // 2. Validate.
    cfg.validate().expect("standard config must be valid");

    // 3. Estimate resources.
    let mem_mb = cfg.total_memory_estimate_mb();
    assert!(mem_mb > 0.0, "memory estimate must be positive");
    assert!(mem_mb.is_finite(), "memory estimate must be finite");

    let hours = cfg.estimated_training_hours(50.0);
    assert!(hours > 0.0, "training hours must be positive");
    assert!(hours.is_finite(), "training hours must be finite");

    // 4. Format summary.
    let summary = cfg.format_summary();
    assert!(
        summary.len() > 50,
        "summary must be non-trivial, got {} chars",
        summary.len()
    );
}

// ============================================================================
// Test 14: Profile + LossScaler combined scenario
// ============================================================================

/// Use a Production profile to drive LossScaler configuration and verify
/// that a simulated training loop with periodic overflow produces the
/// expected scaler statistics.
#[test]
fn test_profile_and_loss_scaler_combined_scenario() {
    let cfg = TrainingProfile::production().config();
    cfg.validate().expect("production config must be valid");

    // Use a tight scale_window so overflows can be observed in fewer iterations.
    let mut scaler = LossScaler::new(256.0).with_scale_window(10);

    let mut overflow_steps = 0u64;
    let mut success_steps = 0u64;

    // Simulate cfg.max_iterations is too many, just run 100 iterations.
    let sim_iters = 100u32;
    for step in 0..sim_iters {
        // Every 11th step triggers an overflow (simulates numerical instability).
        let had_overflow = step % 11 == 0;
        scaler.update(had_overflow);
        if had_overflow {
            overflow_steps += 1;
        } else {
            success_steps += 1;
        }
    }

    let stats = scaler.stats();
    assert_eq!(stats.total_steps, sim_iters as u64);
    assert_eq!(stats.overflow_count, overflow_steps);
    assert_eq!(stats.total_steps - stats.overflow_count, success_steps);
    assert!(
        stats.overflow_rate > 0.0 && stats.overflow_rate < 1.0,
        "overflow rate must be between 0 and 1"
    );
    // Scale must be within min/max bounds (LossScaler defaults: min=1, max=65536).
    assert!(stats.current_scale >= 1.0);
    assert!(stats.current_scale <= 65536.0);
}

// ============================================================================
// Test 15: TrainingDiagnostics status-line stays short across 30 000 iterations
// ============================================================================

/// Verify that the status line stays under 120 characters even when simulating
/// the maximum Production profile iteration count (30 000).
#[test]
fn test_diagnostics_status_line_length_at_scale() {
    let mut diag = TrainingDiagnostics::new();

    for i in 0..30_000u64 {
        diag.next_iteration();
        if i % 100 == 0 {
            diag.record_losses(&[
                ("l1", 0.8 / (1.0 + i as f32 * 0.001)),
                ("ssim", 0.1),
                ("lpips", 0.05),
            ]);
        }
    }

    let line = diag.format_status_line();
    assert!(
        line.len() < 120,
        "status line must be < 120 chars at 30k iters, got {} chars: {}",
        line.len(),
        line
    );
}

// ============================================================================
// Test 16: Multiple EmaTrackers — independent state
// ============================================================================

/// Multiple independent EmaTrackers must not share state even when tracking
/// the same metric name conceptually.
#[test]
fn test_multiple_ema_trackers_independent_state() {
    let mut fast = EmaTracker::new(0.9); // high alpha = fast tracking
    let mut slow = EmaTracker::new(0.01); // low alpha = slow tracking

    // Feed 50 steps with a step-function at step 25.
    for step in 0..50u32 {
        let value = if step < 25 { 0.0_f32 } else { 1.0_f32 };
        fast.update(value);
        slow.update(value);
    }

    let fast_val = fast.smoothed();
    let slow_val = slow.smoothed();

    // Fast tracker should be near 1.0, slow tracker still far below.
    assert!(
        fast_val > 0.9,
        "fast EMA should be near 1.0 after step, got {fast_val}"
    );
    assert!(
        slow_val < 0.5,
        "slow EMA should still be < 0.5 after step, got {slow_val}"
    );
    // They must have diverged.
    assert!(
        fast_val > slow_val,
        "fast EMA must exceed slow EMA, got fast={fast_val}, slow={slow_val}"
    );
}

// ============================================================================
// Test 17: TrainingProfiler — scope (RAII) guard + phase stats integration
// ============================================================================

/// Verify that multiple nested scopes record independent statistics for their
/// respective phases, and that totals add up correctly.
#[test]
fn test_profiler_scope_guard_multiple_phases() {
    let profiler = TrainingProfiler::new(true);

    let n_iters = 20;
    for _ in 0..n_iters {
        {
            let _fwd = profiler.scope(TrainingPhase::Forward);
            // Simulate forward work.
            let _: u64 = (0..1000u64).sum();
        }
        {
            let _bwd = profiler.scope(TrainingPhase::Backward);
            // Simulate backward work.
            let _: u64 = (0..500u64).sum();
        }
    }

    let fwd = profiler
        .stats(TrainingPhase::Forward)
        .expect("forward stats must exist");
    let bwd = profiler
        .stats(TrainingPhase::Backward)
        .expect("backward stats must exist");

    assert_eq!(fwd.count, n_iters);
    assert_eq!(bwd.count, n_iters);
    // Both phases must have non-negative totals.
    assert!(fwd.total_us < u64::MAX);
    assert!(bwd.total_us < u64::MAX);
    // Min ≤ Max for both.
    assert!(fwd.min_us <= fwd.max_us);
    assert!(bwd.min_us <= bwd.max_us);
}

// ============================================================================
// Test 18: Validate production profile rejects zero LR
// ============================================================================

#[test]
fn test_validate_rejects_zero_lr_in_all_fields() {
    let lr_fields = [
        "lr_position",
        "lr_scale",
        "lr_rotation",
        "lr_opacity",
        "lr_sh",
    ];

    for field in &lr_fields {
        let mut cfg = TrainingProfile::production().config();
        match *field {
            "lr_position" => cfg.lr_position = 0.0,
            "lr_scale" => cfg.lr_scale = 0.0,
            "lr_rotation" => cfg.lr_rotation = 0.0,
            "lr_opacity" => cfg.lr_opacity = 0.0,
            "lr_sh" => cfg.lr_sh = 0.0,
            _ => unreachable!(),
        }
        let result = cfg.validate();
        assert!(result.is_err(), "validation must fail when {field} = 0");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains(field),
            "error message must mention {field}, got: {err_msg}"
        );
    }
}

// ============================================================================
// Test 19: format_report round-trip — TrainingDiagnostics with no data
// ============================================================================

/// A fresh `TrainingDiagnostics` must produce a non-empty, valid report
/// even when no data has been recorded yet.
#[test]
fn test_diagnostics_format_report_no_data() {
    let diag = TrainingDiagnostics::new();
    let report = diag.format_report();
    assert!(!report.is_empty(), "report must never be empty");
    assert!(
        report.contains("Iteration"),
        "report must contain 'Iteration'"
    );
}

// ============================================================================
// Test 20: LossScaler has_overflow on empty slice
// ============================================================================

/// An empty gradient slice must not be detected as an overflow.
#[test]
fn test_loss_scaler_has_overflow_empty_slice() {
    let empty: Vec<f32> = Vec::new();
    assert!(
        !LossScaler::has_overflow(&empty),
        "empty slice must not be overflow"
    );
}
