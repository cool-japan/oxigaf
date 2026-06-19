//! Integration tests for mixed-precision training and profiling support.
//!
//! Covers `LossScaler`, `TrainingProfiler`, and `TrainingConfig` defaults.

use oxigaf_trainer::config::TrainingConfig;
use oxigaf_trainer::mixed_precision::{LossScaler, MixedPrecisionTrainer, TrainingPrecision};
use oxigaf_trainer::profiler_integration::{TrainingPhase, TrainingProfiler};

// ---------------------------------------------------------------------------
// LossScaler tests
// ---------------------------------------------------------------------------

/// Scaling then unscaling gradients should recover the original values within
/// floating-point precision.
#[test]
fn test_loss_scaler_scale_unscale_roundtrip() {
    let scaler = LossScaler::new(256.0);
    let original = vec![1.0_f32, -0.5, 0.0, std::f32::consts::PI, -100.0];
    let mut grads = original.clone();

    scaler.scale_gradients(&mut grads);
    scaler.unscale_gradients(&mut grads);

    for (orig, recovered) in original.iter().zip(grads.iter()) {
        assert!(
            (orig - recovered).abs() < 1e-4,
            "roundtrip failed: original={orig}, recovered={recovered}"
        );
    }
}

/// Gradients containing NaN or Inf should be detected as overflow.
#[test]
fn test_loss_scaler_overflow_detection() {
    let nan_grads = vec![1.0_f32, f32::NAN, 0.0];
    assert!(
        LossScaler::has_overflow(&nan_grads),
        "NaN should be detected as overflow"
    );

    let inf_grads = vec![1.0_f32, f32::INFINITY, 0.0];
    assert!(
        LossScaler::has_overflow(&inf_grads),
        "+Inf should be detected as overflow"
    );

    let neg_inf_grads = vec![1.0_f32, f32::NEG_INFINITY, 0.0];
    assert!(
        LossScaler::has_overflow(&neg_inf_grads),
        "-Inf should be detected as overflow"
    );

    let finite_grads = vec![1.0_f32, -2.5, 0.0, 100.0];
    assert!(
        !LossScaler::has_overflow(&finite_grads),
        "finite values must not trigger overflow"
    );
}

/// After an overflow, the scale should be halved.
#[test]
fn test_loss_scaler_scale_halves_on_overflow() {
    let mut scaler = LossScaler::new(1024.0);
    let initial_scale = scaler.scale();
    assert!((initial_scale - 1024.0).abs() < 1e-6);

    scaler.update(true); // simulate overflow

    let new_scale = scaler.scale();
    assert!(
        (new_scale - initial_scale / 2.0).abs() < 1e-6,
        "scale should halve on overflow: was {initial_scale}, got {new_scale}"
    );
}

/// After `scale_window` consecutive successes, the scale should double.
#[test]
fn test_loss_scaler_scale_increases_after_successes() {
    // Use a small scale_window so the test runs quickly.
    let mut scaler = LossScaler::new(512.0)
        .with_scale_window(3)
        .with_max_scale(65536.0);

    let initial_scale = scaler.scale();

    // Two successes — should not have doubled yet.
    scaler.update(false);
    scaler.update(false);
    assert!(
        (scaler.scale() - initial_scale).abs() < 1e-6,
        "scale must not change after 2/3 successes"
    );

    // Third success — scale should now double.
    scaler.update(false);
    let doubled = initial_scale * 2.0;
    assert!(
        (scaler.scale() - doubled).abs() < 1e-6,
        "scale should double after scale_window successes: expected {doubled}, got {}",
        scaler.scale()
    );
}

// ---------------------------------------------------------------------------
// TrainingProfiler tests
// ---------------------------------------------------------------------------

/// `profiler.time(phase, f)` must return the closure's return value unchanged.
#[test]
fn test_training_profiler_time_returns_correct_result() {
    let profiler = TrainingProfiler::new(true);

    let result = profiler.time(TrainingPhase::Forward, || 42u32);
    assert_eq!(result, 42, "time() must return the closure's value");

    let result_str = profiler.time(TrainingPhase::Backward, || "hello".to_string());
    assert_eq!(result_str, "hello");
}

/// A disabled profiler must be a complete no-op: never records, returns closure
/// values unchanged, and produces no stats.
#[test]
fn test_training_profiler_disabled_is_noop() {
    let profiler = TrainingProfiler::disabled();
    assert!(!profiler.is_enabled());

    // Record should silently drop.
    profiler.record(TrainingPhase::Forward, 999_999);

    // Stats must be absent.
    assert!(
        profiler.stats(TrainingPhase::Forward).is_none(),
        "disabled profiler must return None for stats"
    );
    assert!(
        profiler.all_stats().is_empty(),
        "disabled profiler must return empty all_stats()"
    );

    // time() must still call the closure and return its value.
    let val = profiler.time(TrainingPhase::Optimize, || 7u64);
    assert_eq!(val, 7);

    // Report must indicate disabled.
    let report = profiler.format_report();
    assert!(
        report.contains("disabled"),
        "disabled profiler report should mention 'disabled', got: {report}"
    );
}

// ---------------------------------------------------------------------------
// TrainingConfig default precision test
// ---------------------------------------------------------------------------

/// The default `TrainingConfig` must use Float32 precision and have profiling
/// disabled to maintain backward compatibility.
#[test]
fn test_training_config_precision_default() {
    let cfg = TrainingConfig::default();

    assert_eq!(
        cfg.precision,
        TrainingPrecision::Float32,
        "default precision must be Float32"
    );
    assert!(
        !cfg.enable_profiling,
        "profiling must be disabled by default"
    );
}

// ---------------------------------------------------------------------------
// MixedPrecisionTrainer smoke test
// ---------------------------------------------------------------------------

/// `MixedPrecisionTrainer::new(precision)` should configure the scaler's
/// initial scale according to precision mode.
#[test]
fn test_mixed_precision_trainer_initial_scale() {
    let fp32 = MixedPrecisionTrainer::new(TrainingPrecision::Float32);
    assert!(
        (fp32.scaler.scale() - 1.0).abs() < 1e-6,
        "FP32 initial scale must be 1.0"
    );

    let bf16 = MixedPrecisionTrainer::new(TrainingPrecision::BFloat16);
    assert!(
        (bf16.scaler.scale() - 1024.0).abs() < 1e-6,
        "BF16 initial scale must be 1024.0"
    );

    let fp16 = MixedPrecisionTrainer::new(TrainingPrecision::Float16);
    assert!(
        (fp16.scaler.scale() - 65536.0).abs() < 1e-6,
        "FP16 initial scale must be 65536.0"
    );
}
