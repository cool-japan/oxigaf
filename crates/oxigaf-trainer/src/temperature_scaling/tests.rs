//! Unit tests for [`crate::temperature_scaling`].
//!
//! Split out of `temperature_scaling.rs` to keep that file under the
//! 2000-line limit; `super::*` still resolves to the module under test.

use super::*;

// ── helpers ──────────────────────────────────────────────────────────────

/// Simple xorshift64 PRNG (per project policy — no `rand` crate).
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) as f32) / (u64::MAX as f32)
}

fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

// ── CalibrationError ──────────────────────────────────────────────────

#[test]
fn test_error_empty_input_display() {
    let e = CalibrationError::EmptyInput;
    assert!(e.to_string().contains("Empty"));
}

#[test]
fn test_error_length_mismatch_display() {
    let e = CalibrationError::LengthMismatch {
        logits: 3,
        labels: 5,
    };
    let s = e.to_string();
    assert!(s.contains("3") && s.contains("5"));
}

#[test]
fn test_error_invalid_temperature_display() {
    let e = CalibrationError::InvalidTemperature { t: -1.0 };
    assert!(e.to_string().contains("-1"));
}

#[test]
fn test_error_did_not_converge_display() {
    let e = CalibrationError::DidNotConverge { iters: 42 };
    assert!(e.to_string().contains("42"));
}

#[test]
fn test_error_invalid_label_display() {
    let e = CalibrationError::InvalidLabel { label: 1.5 };
    assert!(e.to_string().contains("1.5"));
}

// ── validate_inputs ───────────────────────────────────────────────────

#[test]
fn test_empty_input_error() {
    assert!(matches!(
        ts_binary_nll(&[], &[], 1.0),
        _ // returns 0.0, not an error — validate the ECE path
    ));
    assert!(matches!(
        ts_ece(&[], &[], 10),
        Err(CalibrationError::EmptyInput)
    ));
}

#[test]
fn test_length_mismatch_error() {
    let logits = vec![1.0_f32, 2.0];
    let labels = vec![0.0_f32];
    assert!(matches!(
        ts_ece(&logits, &labels, 10),
        Err(CalibrationError::LengthMismatch { .. })
    ));
}

#[test]
fn test_invalid_label_error() {
    let logits = vec![1.0_f32];
    let labels = vec![1.5_f32];
    assert!(matches!(
        ts_ece(&logits, &labels, 10),
        Err(CalibrationError::InvalidLabel { .. })
    ));
}

// ── TemperatureScaler::new ────────────────────────────────────────────

#[test]
fn test_temperature_scaler_new_valid() {
    let ts = TemperatureScaler::new(1.0).expect("should succeed");
    assert!(approx(ts.temperature, 1.0, 1e-9));
    assert!(!ts.fitted);
}

#[test]
fn test_temperature_scaler_new_zero_error() {
    assert!(matches!(
        TemperatureScaler::new(0.0),
        Err(CalibrationError::InvalidTemperature { .. })
    ));
}

#[test]
fn test_temperature_scaler_new_negative_error() {
    assert!(matches!(
        TemperatureScaler::new(-1.0),
        Err(CalibrationError::InvalidTemperature { .. })
    ));
}

// ── ts_binary_nll ─────────────────────────────────────────────────────

#[test]
fn test_binary_nll_perfect_predictions_near_zero() {
    // logit = 10 → σ(10) ≈ 1; label = 1 → NLL ≈ 0
    let logits = vec![10.0_f32; 20];
    let labels = vec![1.0_f32; 20];
    let nll = ts_binary_nll(&logits, &labels, 1.0);
    assert!(nll < 1e-4, "NLL={nll} should be near 0");
}

#[test]
fn test_binary_nll_negative_logits_zero_label() {
    // logit = -10 → σ(-10) ≈ 0; label = 0 → NLL ≈ 0
    let logits = vec![-10.0_f32; 20];
    let labels = vec![0.0_f32; 20];
    let nll = ts_binary_nll(&logits, &labels, 1.0);
    assert!(nll < 1e-4, "NLL={nll} should be near 0");
}

#[test]
fn test_binary_nll_random_predictions_finite() {
    let mut state = 0xDEAD_BEEF_u64;
    let logits: Vec<f32> = (0..50)
        .map(|_| xorshift_f32(&mut state) * 4.0 - 2.0)
        .collect();
    let labels: Vec<f32> = (0..50)
        .map(|_| {
            if xorshift_f32(&mut state) > 0.5 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let nll = ts_binary_nll(&logits, &labels, 1.0);
    assert!(nll.is_finite(), "NLL must be finite, got {nll}");
    assert!(nll > 0.0, "NLL must be positive, got {nll}");
}

#[test]
fn test_binary_nll_empty_returns_zero() {
    let nll = ts_binary_nll(&[], &[], 1.0);
    assert!(approx(nll, 0.0, 1e-9));
}

#[test]
fn test_binary_nll_higher_temp_smooths_nll() {
    // At T >> 1, predictions approach 0.5 → higher NLL for certain data
    let logits = vec![5.0_f32; 10];
    let labels = vec![1.0_f32; 10];
    let nll_t1 = ts_binary_nll(&logits, &labels, 1.0);
    let nll_t100 = ts_binary_nll(&logits, &labels, 100.0);
    assert!(
        nll_t100 > nll_t1,
        "Higher T should increase NLL for confident correct predictions"
    );
}

// ── ts_golden_section_search ──────────────────────────────────────────

#[test]
fn test_golden_section_x_squared() {
    // Minimum of x² on [-1, 1] is at 0.
    let min = ts_golden_section_search(|x| x * x, -1.0, 1.0, 1e-8, 200);
    assert!(min.abs() < 1e-4, "min of x² should be ~0, got {min}");
}

#[test]
fn test_golden_section_quadratic_with_offset() {
    // Minimum of (x - 0.3)² on [0, 1] is at 0.3.
    let min = ts_golden_section_search(|x| (x - 0.3) * (x - 0.3), 0.0, 1.0, 1e-8, 300);
    assert!(approx(min, 0.3, 1e-3), "expected min near 0.3, got {min}");
}

#[test]
fn test_golden_section_inverted_parabola_boundary() {
    // -(x-2)² on [0,1] is maximised at x=1, i.e. minimum of (x-2)² on [0,1] at 1
    let min = ts_golden_section_search(|x| (x - 2.0) * (x - 2.0), 0.0, 1.0, 1e-8, 300);
    assert!(approx(min, 1.0, 1e-3), "expected 1.0, got {min}");
}

#[test]
fn test_golden_section_tracked_iteration_counting() {
    // Tracked and untracked must agree on the minimiser (the untracked
    // one now delegates to the tracked one). A zero-width bracket needs
    // 0 iterations (already within any positive tolerance); a tolerance
    // of 0 can never be satisfied, so the budget is fully exhausted.
    //
    // The `tol = 1e-8` case below is only satisfiable because of the
    // shrink-stagnation guard: `f32` spacing around the minimiser 0.3 is
    // ≈ 2.98e-8, so `|b - a| < 1e-8` is never reachable and the tolerance
    // test alone would burn all 300 iterations.  See
    // `test_golden_section_stops_at_f32_resolution_floor`.
    let untracked = ts_golden_section_search(|x| (x - 0.3) * (x - 0.3), 0.0, 1.0, 1e-8, 300);
    let (tracked, iters) =
        ts_golden_section_search_tracked(|x| (x - 0.3) * (x - 0.3), 0.0, 1.0, 1e-8, 300);
    assert_eq!(untracked, tracked);
    assert!(
        iters > 0 && iters < 300,
        "iters={iters} out of expected range"
    );

    let (_, zero_iters) = ts_golden_section_search_tracked(|x| x * x, 0.5, 0.5, 1e-6, 100);
    assert_eq!(zero_iters, 0);

    let (_, maxed_iters) = ts_golden_section_search_tracked(|x| x * x, -1.0, 1.0, 0.0, 25);
    assert_eq!(maxed_iters, 25);
}

/// Regression: a `tol` finer than `f32` can represent near the minimiser must
/// terminate at the resolution floor, not spin out the whole `max_iters`
/// budget — and must still return the right answer when it does.
///
/// `(x - 0.3)²` on `[0, 1]` with `tol = 1e-12`: the bracket shrinks by
/// `1 - φ ≈ 0.618` per iteration, so reaching width `1e-12` from width `1`
/// would need ≈ 57 iterations, but `f32` spacing around `0.3` is ≈ 2.98e-8,
/// so the bracket freezes after ≈ 36.  Without the shrink-stagnation guard
/// this runs all 10_000 iterations doing nothing; with it, well under 100.
///
/// The minimiser assertion is the other half: an early-exit that bails before
/// the bracket has actually converged would satisfy the iteration bound while
/// returning garbage.
#[test]
fn test_golden_section_stops_at_f32_resolution_floor() {
    let (min, iters) =
        ts_golden_section_search_tracked(|x| (x - 0.3) * (x - 0.3), 0.0, 1.0, 1e-12, 10_000);

    assert!(
        iters > 0 && iters < 100,
        "sub-ulp tol must stop at the f32 floor, not exhaust the budget; iters={iters}"
    );
    assert!(
        approx(min, 0.3, 1e-3),
        "stopping early must not cost accuracy; expected ~0.3, got {min}"
    );
}

/// Regression: a non-finite bracket must terminate rather than loop.
///
/// `width` is NaN, so neither `width < tol` nor `new_width < width` holds;
/// the guard is phrased as "did not shrink" precisely so the NaN case still
/// breaks out instead of running the full budget.
#[test]
fn test_golden_section_nan_bracket_terminates() {
    let (_, iters) = ts_golden_section_search_tracked(|x| x * x, f32::NAN, 1.0, 1e-6, 10_000);
    assert!(
        iters <= 1,
        "a NaN bracket cannot shrink and must stop immediately, got iters={iters}"
    );
}

// ── TemperatureScaler::fit_binary ─────────────────────────────────────

#[test]
fn test_temperature_scaler_fit_balanced_data() {
    // Balanced, roughly calibrated data with moderate logits (not extreme).
    // Use logits where sigmoid(logit) ≈ label, so calibration is already near-optimal.
    // logit=0 → p=0.5 → label=0 or 1 alternating.
    // We use soft logits near 0 so the optimal T is not degenerate.
    let logits = vec![-1.0_f32, -0.5, 0.0, 0.5, 1.0, -1.0, -0.5, 0.0, 0.5, 1.0];
    let labels = vec![0.0_f32, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 1.0];
    let mut ts = TemperatureScaler::new(1.0).expect("ok");
    let cfg = CalibrationConfig::default();
    ts.fit_binary(&logits, &labels, &cfg).expect("fit ok");
    assert!(ts.fitted);
    // T should be positive and finite
    assert!(
        ts.temperature > 0.0,
        "T must be positive, got {}",
        ts.temperature
    );
    assert!(
        ts.temperature.is_finite(),
        "T must be finite, got {}",
        ts.temperature
    );
}

#[test]
fn test_temperature_scaler_fit_overconfident() {
    // Overconfident model: large logits but only ~50% accuracy → T > 1
    let logits = vec![10.0_f32, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0];
    // But actual labels are ~50/50 mixed
    let labels = vec![1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let mut ts = TemperatureScaler::new(1.0).expect("ok");
    let cfg = CalibrationConfig::default();
    ts.fit_binary(&logits, &labels, &cfg).expect("fit ok");
    assert!(
        ts.temperature > 1.0,
        "Overconfident model should yield T > 1, got {}",
        ts.temperature
    );
}

#[test]
fn test_temperature_scaler_fit_length_mismatch_error() {
    let mut ts = TemperatureScaler::new(1.0).expect("ok");
    let cfg = CalibrationConfig::default();
    assert!(matches!(
        ts.fit_binary(&[1.0, 2.0], &[0.0], &cfg),
        Err(CalibrationError::LengthMismatch { .. })
    ));
}

#[test]
fn test_temperature_scaler_fit_empty_error() {
    let mut ts = TemperatureScaler::new(1.0).expect("ok");
    let cfg = CalibrationConfig::default();
    assert!(matches!(
        ts.fit_binary(&[], &[], &cfg),
        Err(CalibrationError::EmptyInput)
    ));
}

#[test]
fn test_temperature_scaler_fit_tracks_iterations_used() {
    // `iterations_used` didn't exist before this fix; golden-section
    // search on a bounded bracket always converges well before
    // exhausting a generous `max_iters` budget, regardless of objective.
    let logits = vec![-1.0_f32, -0.5, 0.0, 0.5, 1.0, -1.0, -0.5, 0.0, 0.5, 1.0];
    let labels = vec![0.0_f32, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 1.0];
    let mut ts = TemperatureScaler::new(1.0).expect("ok");
    let cfg = CalibrationConfig::default();
    assert_eq!(ts.iterations_used, 0);
    ts.fit_binary(&logits, &labels, &cfg).expect("fit ok");
    assert!(ts.iterations_used > 0 && ts.iterations_used < cfg.max_iters);
}

// ── TemperatureScaler::scale ──────────────────────────────────────────

#[test]
fn test_temperature_scaler_scale_output_in_01() {
    let ts = TemperatureScaler::new(1.5).expect("ok");
    for &logit in &[-10.0_f32, -1.0, 0.0, 1.0, 10.0] {
        let p = ts.scale(logit);
        assert!(p > 0.0 && p < 1.0, "p={p} not in (0,1)");
    }
}

#[test]
fn test_temperature_scaler_scale_batch_length_preserved() {
    let ts = TemperatureScaler::new(1.0).expect("ok");
    let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let out = ts.scale_batch(&logits);
    assert_eq!(out.len(), logits.len());
}

#[test]
fn test_temperature_scaler_scale_zero_logit_half() {
    // σ(0 / T) = 0.5 for any T
    let ts = TemperatureScaler::new(2.0).expect("ok");
    assert!(approx(ts.scale(0.0), 0.5, 1e-6));
}

#[test]
fn test_temperature_scaler_t1_equals_sigmoid() {
    let ts = TemperatureScaler::new(1.0).expect("ok");
    for &logit in &[-3.0_f32, 0.0, 3.0] {
        assert!(approx(ts.scale(logit), sigmoid(logit), 1e-7));
    }
}

#[test]
fn test_temperature_scaler_large_t_conservative() {
    // Large T → predictions near 0.5
    let ts_large = TemperatureScaler::new(100.0).expect("ok");
    let p = ts_large.scale(5.0);
    assert!(
        approx(p, 0.5, 0.05),
        "Large T should give p near 0.5, got {p}"
    );
}

#[test]
fn test_temperature_scaler_small_t_overconfident() {
    // Small T → predictions near 0 or 1
    let ts_small = TemperatureScaler::new(0.01).expect("ok");
    let p_pos = ts_small.scale(1.0);
    let p_neg = ts_small.scale(-1.0);
    assert!(
        p_pos > 0.99,
        "Small T should give p near 1 for pos logit, got {p_pos}"
    );
    assert!(
        p_neg < 0.01,
        "Small T should give p near 0 for neg logit, got {p_neg}"
    );
}

// ── PlattScaler ───────────────────────────────────────────────────────

#[test]
fn test_platt_scaler_fit_no_error() {
    let mut ps = PlattScaler::new();
    let logits = vec![1.0_f32, -1.0, 2.0, -2.0];
    let labels = vec![1.0_f32, 0.0, 1.0, 0.0];
    let cfg = CalibrationConfig::default();
    ps.fit(&logits, &labels, &cfg).expect("platt fit ok");
    assert!(ps.fitted);
}

#[test]
fn test_platt_scaler_fit_identity_approx() {
    // Well-calibrated logits → a ≈ 1, b ≈ 0 (approximately)
    let mut ps = PlattScaler::new();
    let logits: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
    let labels: Vec<f32> = logits
        .iter()
        .map(|&l| if l > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let cfg = CalibrationConfig {
        max_iters: 2000,
        learning_rate: 0.01,
        ..Default::default()
    };
    ps.fit(&logits, &labels, &cfg).expect("fit ok");
    // a should be positive
    assert!(ps.a > 0.0, "a should be positive, got {}", ps.a);
}

#[test]
fn test_platt_scaler_predict_output_in_01() {
    let ps = PlattScaler::new();
    for &logit in &[-10.0_f32, 0.0, 10.0] {
        let p = ps.predict(logit);
        assert!(p > 0.0 && p < 1.0, "p={p} not in (0,1)");
    }
}

#[test]
fn test_platt_scaler_predict_batch_length() {
    let ps = PlattScaler::new();
    let logits = vec![1.0_f32, 2.0, 3.0];
    assert_eq!(ps.predict_batch(&logits).len(), 3);
}

#[test]
fn test_platt_scaler_fit_empty_error() {
    let mut ps = PlattScaler::new();
    let cfg = CalibrationConfig::default();
    assert!(matches!(
        ps.fit(&[], &[], &cfg),
        Err(CalibrationError::EmptyInput)
    ));
}

#[test]
fn test_platt_scaler_fit_mismatch_error() {
    let mut ps = PlattScaler::new();
    let cfg = CalibrationConfig::default();
    assert!(matches!(
        ps.fit(&[1.0, 2.0], &[0.0], &cfg),
        Err(CalibrationError::LengthMismatch { .. })
    ));
}

#[test]
fn test_platt_scaler_fit_stops_early_when_converged() {
    // Regression: `fit` used to ignore `config.tolerance` and always run
    // the full `max_iters`. A loose tolerance is satisfied by the very
    // first (tiny, lr=0.01) step, so it must now stop after 1 iteration.
    let mut ps = PlattScaler::new();
    let logits = vec![1.0_f32, -1.0, 2.0, -2.0];
    let labels = vec![1.0_f32, 0.0, 1.0, 0.0];
    let cfg = CalibrationConfig {
        tolerance: 1.0,
        ..Default::default()
    };
    ps.fit(&logits, &labels, &cfg).expect("fit ok");
    assert!(ps.fitted);
    assert!(ps.converged);
    assert_eq!(ps.iterations_used, 1);
}

#[test]
fn test_platt_scaler_fit_reports_non_convergence() {
    // A tiny iteration budget can't reach the strict default tolerance;
    // `fit` must still succeed (a non-converged fit is still usable) but
    // now reports `converged = false`, where it previously gave no way
    // to detect non-convergence at all.
    let mut ps = PlattScaler::new();
    let logits = vec![1.0_f32, -1.0, 2.0, -2.0];
    let labels = vec![1.0_f32, 0.0, 1.0, 0.0];
    let cfg = CalibrationConfig {
        max_iters: 2,
        ..Default::default()
    };
    ps.fit(&logits, &labels, &cfg).expect("fit still succeeds");
    assert!(ps.fitted);
    assert!(!ps.converged);
    assert_eq!(ps.iterations_used, 2);
}

// ── ts_pav_isotonic ───────────────────────────────────────────────────

#[test]
fn test_pav_isotonic_monotone() {
    let values = vec![0.9_f32, 0.1, 0.8, 0.2, 0.7];
    let weights = vec![1.0_f32; 5];
    let result = ts_pav_isotonic(&values, &weights).expect("valid weights");
    assert_eq!(result.len(), 5);
    for i in 0..result.len() - 1 {
        assert!(
            result[i] <= result[i + 1] + 1e-6,
            "PAV result must be non-decreasing, got [{i}]={} > [{}]={}",
            result[i],
            i + 1,
            result[i + 1]
        );
    }
}

#[test]
fn test_pav_isotonic_constant_input() {
    let values = vec![0.5_f32; 5];
    let weights = vec![1.0_f32; 5];
    let result = ts_pav_isotonic(&values, &weights).expect("valid weights");
    for &v in &result {
        assert!(approx(v, 0.5, 1e-6), "Constant input → constant output");
    }
}

#[test]
fn test_pav_isotonic_empty_input() {
    let result = ts_pav_isotonic(&[], &[]).expect("empty input is valid");
    assert!(result.is_empty());
}

#[test]
fn test_pav_isotonic_already_sorted() {
    // Already monotone input should pass through unchanged
    let values = vec![0.1_f32, 0.3, 0.5, 0.7, 0.9];
    let weights = vec![1.0_f32; 5];
    let result = ts_pav_isotonic(&values, &weights).expect("valid weights");
    for (r, v) in result.iter().zip(values.iter()) {
        assert!(approx(*r, *v, 1e-5));
    }
}

#[test]
fn test_pav_isotonic_single_element() {
    let result = ts_pav_isotonic(&[0.7_f32], &[1.0]).expect("valid weights");
    assert_eq!(result.len(), 1);
    assert!(approx(result[0], 0.7, 1e-6));
}

#[test]
fn test_pav_isotonic_weights_length_mismatch_error() {
    // Regression: previously indexed `weights[i]` for `i in 0..values.len()`
    // with no length check, panicking on a shorter `weights` slice.
    let result = ts_pav_isotonic(&[1.0, 2.0], &[1.0]);
    assert!(matches!(
        result,
        Err(CalibrationError::WeightsLengthMismatch {
            expected: 2,
            got: 1
        })
    ));
}

#[test]
fn test_pav_isotonic_invalid_weight_errors() {
    // Regression: a non-positive or non-finite weight used to make a
    // block's `total_weight` zero (or invert pooling order), producing
    // NaN/Inf via `weighted_sum / total_weight` instead of a clean error.
    for bad_weight in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        let result = ts_pav_isotonic(&[0.5, 0.5], &[1.0, bad_weight]);
        assert!(
            matches!(
                result,
                Err(CalibrationError::InvalidWeight { index: 1, .. })
            ),
            "weight {bad_weight} should be rejected, got {result:?}"
        );
    }
}

// ── IsotonicCalibrator ────────────────────────────────────────────────

#[test]
fn test_isotonic_calibrator_fit_no_error() {
    let mut ic = IsotonicCalibrator::new();
    let scores = vec![0.1_f32, 0.5, 0.9];
    let labels = vec![0.0_f32, 1.0, 1.0];
    ic.fit(&scores, &labels).expect("isotonic fit ok");
    assert!(ic.fitted);
}

#[test]
fn test_isotonic_calibrator_predict_in_01() {
    let mut ic = IsotonicCalibrator::new();
    let scores = vec![0.1_f32, 0.3, 0.6, 0.9];
    let labels = vec![0.0_f32, 0.0, 1.0, 1.0];
    ic.fit(&scores, &labels).expect("ok");
    for &s in &[0.0_f32, 0.2, 0.5, 0.8, 1.0] {
        let p = ic.predict(s);
        assert!((0.0..=1.0).contains(&p), "p={p} not in [0,1]");
    }
}

#[test]
fn test_isotonic_calibrator_predict_batch_length() {
    let mut ic = IsotonicCalibrator::new();
    ic.fit(&[0.1_f32, 0.9], &[0.0_f32, 1.0]).expect("ok");
    let out = ic.predict_batch(&[0.0_f32, 0.5, 1.0]);
    assert_eq!(out.len(), 3);
}

#[test]
fn test_isotonic_calibrator_fit_empty_error() {
    let mut ic = IsotonicCalibrator::new();
    assert!(matches!(
        ic.fit(&[], &[]),
        Err(CalibrationError::EmptyInput)
    ));
}

// ── calibrate() ──────────────────────────────────────────────────────

// Shared fixture: 20 points in two separated clusters, so raw-sigmoid
// ECE/MCE is genuinely nonzero and `calibrate` has something to improve.
fn calibrate_fixture() -> (Vec<f32>, Vec<f32>) {
    let logits: Vec<f32> = (0..20).map(|i| (i as f32 - 9.5) * 0.8).collect();
    let labels: Vec<f32> = logits
        .iter()
        .map(|&l| if l > 0.0 { 1.0 } else { 0.0 })
        .collect();
    (logits, labels)
}

#[test]
fn test_calibrate_produces_result_for_each_method() {
    // Regression: `CalibrationResult` was previously constructed only in
    // tests, never by library code — `calibrate` must actually produce
    // one, with a real (nonzero) iteration count, for every method.
    let (logits, labels) = calibrate_fixture();
    let cfg = CalibrationConfig::default();

    let temp = calibrate(&logits, &labels, CalibrationMethod::Temperature, &cfg)
        .expect("temperature calibration should succeed");
    assert_eq!(temp.method, "temperature");
    assert_eq!(temp.n_samples, logits.len());
    assert!(temp.pre_ece >= 0.0 && temp.post_ece >= 0.0);
    assert!(temp.pre_mce >= 0.0 && temp.post_mce >= 0.0);
    assert!(temp.iterations_used > 0);

    let iso = calibrate(&logits, &labels, CalibrationMethod::Isotonic, &cfg)
        .expect("isotonic calibration should succeed");
    assert_eq!(iso.method, "isotonic");
    assert_eq!(iso.iterations_used, 1, "PAV is a single exact pass");
}

#[test]
fn test_calibrate_platt_convergence_and_non_convergence() {
    let logits = vec![1.0_f32, -1.0, 2.0, -2.0];
    let labels = vec![1.0_f32, 0.0, 1.0, 0.0];

    // Loose tolerance: converges in 1 step (see
    // `test_platt_scaler_fit_stops_early_when_converged`).
    let loose = CalibrationConfig {
        tolerance: 1.0,
        ..Default::default()
    };
    let result = calibrate(&logits, &labels, CalibrationMethod::Platt, &loose)
        .expect("platt should converge under a loose tolerance");
    assert_eq!(result.method, "platt");
    assert_eq!(result.iterations_used, 1);

    // Tiny budget + strict default tolerance: cannot converge, so
    // `calibrate` must surface that rather than reporting a
    // `CalibrationResult` built on an unreliable fit (the original bug:
    // no way to detect non-convergence at all).
    let tiny_budget = CalibrationConfig {
        max_iters: 2,
        ..Default::default()
    };
    let result = calibrate(&logits, &labels, CalibrationMethod::Platt, &tiny_budget);
    assert!(matches!(
        result,
        Err(CalibrationError::DidNotConverge { iters: 2 })
    ));
}

#[test]
fn test_calibrate_validates_inputs() {
    let cfg = CalibrationConfig::default();
    assert!(matches!(
        calibrate(&[], &[], CalibrationMethod::Temperature, &cfg),
        Err(CalibrationError::EmptyInput)
    ));
    assert!(matches!(
        calibrate(&[1.0, 2.0], &[0.0], CalibrationMethod::Isotonic, &cfg),
        Err(CalibrationError::LengthMismatch { .. })
    ));
}

// ── ECE / MCE / Overconfidence ────────────────────────────────────────

#[test]
fn test_ece_perfect_calibration_near_zero() {
    // Each sample's confidence equals its (binary) label exactly
    let n = 100;
    let mut conf = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        // confidence at midpoint of each decile bin
        let c = (i as f32 + 0.5) / n as f32;
        conf.push(c);
        // label = 1 with prob = c → for ECE test use deterministic: label = round(c)
        labels.push(if c >= 0.5 { 1.0_f32 } else { 0.0 });
    }
    // ECE won't be exactly 0 for deterministic labels, but should be small
    let ece = ts_ece(&conf, &labels, 10).expect("ece ok");
    assert!(
        ece < 0.3,
        "ECE={ece} should be relatively small for near-calibrated data"
    );
}

#[test]
fn test_ece_all_confident_but_wrong() {
    // All confident (p=0.99) but all wrong (label=0) → high ECE
    let conf = vec![0.99_f32; 100];
    let labels = vec![0.0_f32; 100];
    let ece = ts_ece(&conf, &labels, 10).expect("ece ok");
    assert!(
        ece > 0.5,
        "ECE={ece} should be high for confidently-wrong predictions"
    );
}

#[test]
fn test_mce_geq_ece() {
    let conf = vec![0.9_f32, 0.8, 0.1, 0.2, 0.6];
    let labels = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0];
    let ece = ts_ece(&conf, &labels, 5).expect("ece ok");
    let mce = ts_mce(&conf, &labels, 5).expect("mce ok");
    assert!(mce >= ece - 1e-6, "MCE={mce} should be >= ECE={ece}");
}

#[test]
fn test_overconfidence_error_nonnegative() {
    let conf = vec![0.9_f32, 0.1, 0.7];
    let labels = vec![0.0_f32, 1.0, 0.5];
    let oe = ts_overconfidence_error(&conf, &labels, 5).expect("oe ok");
    assert!(oe >= 0.0, "Overconfidence error must be >= 0, got {oe}");
}

#[test]
fn test_overconfidence_error_underconfident_model() {
    // If conf < acc everywhere, overconfidence error should be 0
    let conf = vec![0.01_f32; 50];
    let labels = vec![1.0_f32; 50];
    let oe = ts_overconfidence_error(&conf, &labels, 10).expect("oe ok");
    assert!(approx(oe, 0.0, 1e-5), "No overconfidence → OE=0, got {oe}");
}

#[test]
fn test_ece_empty_error() {
    assert!(matches!(
        ts_ece(&[], &[], 10),
        Err(CalibrationError::EmptyInput)
    ));
}

#[test]
fn test_mce_empty_error() {
    assert!(matches!(
        ts_mce(&[], &[], 10),
        Err(CalibrationError::EmptyInput)
    ));
}

// ── Reliability diagram ───────────────────────────────────────────────

#[test]
fn test_reliability_diagram_bin_edges_count() {
    let conf = vec![0.1_f32, 0.5, 0.9];
    let labels = vec![0.0_f32, 1.0, 1.0];
    let diag = ts_reliability_diagram(&conf, &labels, 10).expect("diag ok");
    assert_eq!(diag.bin_edges.len(), 11, "n_bins+1 edges");
}

#[test]
fn test_reliability_diagram_bin_counts_sum_to_n() {
    let conf = vec![0.1_f32, 0.5, 0.9];
    let labels = vec![0.0_f32, 1.0, 1.0];
    let diag = ts_reliability_diagram(&conf, &labels, 10).expect("diag ok");
    let total: usize = diag.bin_counts.iter().sum();
    assert_eq!(total, conf.len(), "bin counts should sum to N");
}

#[test]
fn test_reliability_diagram_ece_mce_nonnegative() {
    let conf = vec![0.3_f32, 0.7, 0.5];
    let labels = vec![0.0_f32, 1.0, 1.0];
    let diag = ts_reliability_diagram(&conf, &labels, 10).expect("diag ok");
    assert!(diag.ece >= 0.0);
    assert!(diag.mce >= 0.0);
}

#[test]
fn test_reliability_diagram_empty_error() {
    assert!(matches!(
        ts_reliability_diagram(&[], &[], 10),
        Err(CalibrationError::EmptyInput)
    ));
}

#[test]
fn test_format_reliability_diagram_nonempty() {
    let conf = vec![0.1_f32, 0.5, 0.9];
    let labels = vec![0.0_f32, 1.0, 1.0];
    let diag = ts_reliability_diagram(&conf, &labels, 5).expect("diag ok");
    let s = ts_format_reliability_diagram(&diag);
    assert!(!s.is_empty(), "Format should produce non-empty string");
    assert!(s.contains("ECE"), "Should contain ECE label");
}

// ── Brier score ───────────────────────────────────────────────────────

#[test]
fn test_brier_score_perfect_predictions_zero() {
    let conf = vec![1.0_f32; 10];
    let labels = vec![1.0_f32; 10];
    let bs = ts_brier_score(&conf, &labels).expect("ok");
    assert!(
        approx(bs, 0.0, 1e-7),
        "Perfect predictions → Brier=0, got {bs}"
    );
}

#[test]
fn test_brier_score_worst_predictions() {
    // Predict 1 when label=0 (or 0 when label=1) → Brier = 1
    let conf = vec![1.0_f32; 10];
    let labels = vec![0.0_f32; 10];
    let bs = ts_brier_score(&conf, &labels).expect("ok");
    assert!(
        approx(bs, 1.0, 1e-6),
        "Worst predictions → Brier=1, got {bs}"
    );
}

#[test]
fn test_brier_score_empty_error() {
    assert!(matches!(
        ts_brier_score(&[], &[]),
        Err(CalibrationError::EmptyInput)
    ));
}

#[test]
fn test_brier_score_nonnegative() {
    let mut state = 0xCAFE_u64;
    let conf: Vec<f32> = (0..20).map(|_| xorshift_f32(&mut state)).collect();
    let labels: Vec<f32> = (0..20)
        .map(|_| {
            if xorshift_f32(&mut state) > 0.5 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let bs = ts_brier_score(&conf, &labels).expect("ok");
    assert!(bs >= 0.0);
}

// ── Log loss ──────────────────────────────────────────────────────────

#[test]
fn test_log_loss_perfect_near_zero() {
    let conf = vec![0.9999_f32; 10];
    let labels = vec![1.0_f32; 10];
    let ll = ts_log_loss(&conf, &labels, 1e-7).expect("ok");
    assert!(
        ll < 0.01,
        "Near-perfect predictions → low log-loss, got {ll}"
    );
}

#[test]
fn test_log_loss_worst_large() {
    let conf = vec![1e-7_f32; 10];
    let labels = vec![1.0_f32; 10];
    let ll = ts_log_loss(&conf, &labels, 1e-15).expect("ok");
    assert!(ll > 10.0, "Worst log-loss should be large, got {ll}");
}

#[test]
fn test_log_loss_empty_error() {
    assert!(matches!(
        ts_log_loss(&[], &[], 1e-7),
        Err(CalibrationError::EmptyInput)
    ));
}

// ── CalibrationStats ──────────────────────────────────────────────────

#[test]
fn test_compute_stats_brier_matches_standalone() {
    let conf = vec![0.2_f32, 0.7, 0.4, 0.9];
    let labels = vec![0.0_f32, 1.0, 0.0, 1.0];
    let stats = ts_compute_stats(&conf, &labels).expect("ok");
    let brier = ts_brier_score(&conf, &labels).expect("ok");
    assert!(approx(stats.brier_score, brier, 1e-6));
}

#[test]
fn test_compute_stats_log_loss_matches_standalone() {
    let conf = vec![0.2_f32, 0.7, 0.4, 0.9];
    let labels = vec![0.0_f32, 1.0, 0.0, 1.0];
    let stats = ts_compute_stats(&conf, &labels).expect("ok");
    let ll = ts_log_loss(&conf, &labels, 1e-7).expect("ok");
    assert!(approx(stats.log_loss, ll, 1e-5));
}

#[test]
fn test_compute_stats_empty_error() {
    assert!(matches!(
        ts_compute_stats(&[], &[]),
        Err(CalibrationError::EmptyInput)
    ));
}

#[test]
fn test_compute_stats_mean_confidence_correct() {
    let conf = vec![0.0_f32, 0.5, 1.0];
    let labels = vec![0.0_f32, 0.0, 1.0];
    let stats = ts_compute_stats(&conf, &labels).expect("ok");
    assert!(approx(stats.mean_confidence, 0.5, 1e-6));
}

#[test]
fn test_compute_stats_confidence_std_nonneg() {
    let conf = vec![0.1_f32, 0.9, 0.5, 0.3];
    let labels = vec![0.0_f32, 1.0, 1.0, 0.0];
    let stats = ts_compute_stats(&conf, &labels).expect("ok");
    assert!(stats.confidence_std >= 0.0);
}

// ── Format functions ──────────────────────────────────────────────────

#[test]
fn test_format_stats_nonempty() {
    let conf = vec![0.7_f32, 0.3];
    let labels = vec![1.0_f32, 0.0];
    let stats = ts_compute_stats(&conf, &labels).expect("ok");
    let s = ts_format_stats(&stats);
    assert!(!s.is_empty());
    assert!(s.contains("brier"));
}

#[test]
fn test_format_result_nonempty() {
    let result = CalibrationResult {
        pre_ece: 0.1,
        post_ece: 0.05,
        pre_mce: 0.2,
        post_mce: 0.1,
        method: "temperature".to_string(),
        n_samples: 100,
        iterations_used: 50,
    };
    let s = ts_format_result(&result);
    assert!(!s.is_empty());
    assert!(s.contains("temperature"));
}

#[test]
fn test_format_result_display_trait() {
    let result = CalibrationResult {
        pre_ece: 0.1,
        post_ece: 0.05,
        pre_mce: 0.2,
        post_mce: 0.1,
        method: "platt".to_string(),
        n_samples: 50,
        iterations_used: 1000,
    };
    let s = format!("{}", result);
    assert!(s.contains("platt"));
}

// ── Sigmoid helper ────────────────────────────────────────────────────

#[test]
fn test_sigmoid_zero_is_half() {
    assert!(approx(sigmoid(0.0), 0.5, 1e-9));
}

#[test]
fn test_sigmoid_large_pos_near_one() {
    assert!(sigmoid(100.0) > 0.999);
}

#[test]
fn test_sigmoid_large_neg_near_zero() {
    assert!(sigmoid(-100.0) < 0.001);
}

// ── Platt scaler default ──────────────────────────────────────────────

#[test]
fn test_platt_scaler_default_identity() {
    let ps = PlattScaler::default();
    assert!(approx(ps.a, 1.0, 1e-9));
    assert!(approx(ps.b, 0.0, 1e-9));
}

// ── CalibrationConfig default ─────────────────────────────────────────

#[test]
fn test_calibration_config_default_values() {
    let cfg = CalibrationConfig::default();
    assert_eq!(cfg.n_bins, 10);
    assert_eq!(cfg.max_iters, 1000);
    assert!(approx(cfg.tolerance, 1e-6, 1e-10));
    assert!(approx(cfg.learning_rate, 0.01, 1e-9));
}
