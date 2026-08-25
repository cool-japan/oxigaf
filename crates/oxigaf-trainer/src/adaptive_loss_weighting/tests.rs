//! Unit tests for [`crate::adaptive_loss_weighting`].
//!
//! Split out of the parent module to keep every file under the 2000-line cap.

use super::*;

// ── LossTask ──────────────────────────────────────────────────────────────

#[test]
fn loss_task_new_defaults() {
    let t = LossTask::new("photo", 1.0);
    assert_eq!(t.name, "photo");
    assert!((t.initial_weight - 1.0).abs() < 1e-6);
    assert!((t.min_weight - 0.001).abs() < 1e-6);
    assert!((t.max_weight - 100.0).abs() < 1e-6);
    assert!(!t.is_primary);
}

#[test]
fn loss_task_with_bounds() {
    let t = LossTask::new("reg", 0.1).with_bounds(0.01, 10.0);
    assert!((t.min_weight - 0.01).abs() < 1e-6);
    assert!((t.max_weight - 10.0).abs() < 1e-6);
}

#[test]
fn loss_task_primary() {
    let t = LossTask::new("photo", 1.0).primary();
    assert!(t.is_primary);
}

#[test]
fn loss_task_not_primary_by_default() {
    let t = LossTask::new("reg", 0.1);
    assert!(!t.is_primary);
}

// ── HomoscedasticWeighter ─────────────────────────────────────────────────

#[test]
fn homo_new_empty_error() {
    let result = HomoscedasticWeighter::new(vec![]);
    assert!(matches!(result, Err(LossWeightError::EmptyTaskList)));
}

#[test]
fn homo_new_two_tasks_ok() {
    let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0), LossTask::new("b", 0.5)]);
    assert!(w.is_ok());
    assert_eq!(w.unwrap().n_tasks(), 2);
}

#[test]
fn homo_weight_log_sigma_zero_gives_one() {
    let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
    let weight = w.weight(0).unwrap();
    // exp(-2*0) = 1
    assert!((weight - 1.0).abs() < 1e-5);
}

#[test]
fn homo_weight_log_sigma_ln2_gives_quarter() {
    let mut w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
    // log_sigma = ln(2) → weight = exp(-2*ln(2)) = exp(ln(0.25)) = 0.25
    w.update_log_sigma(0, std::f32::consts::LN_2).unwrap();
    let weight = w.weight(0).unwrap();
    assert!((weight - 0.25).abs() < 1e-4);
}

#[test]
fn homo_weight_out_of_bounds_error() {
    let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
    assert!(w.weight(1).is_err());
}

#[test]
fn homo_regularization_zero_log_sigma() {
    let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
    let reg = w.regularization(0).unwrap();
    assert!((reg - 0.0).abs() < 1e-6);
}

#[test]
fn homo_regularization_nonzero_log_sigma() {
    let mut w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
    w.update_log_sigma(0, 0.5).unwrap();
    let reg = w.regularization(0).unwrap();
    assert!((reg - 0.5).abs() < 1e-5);
}

#[test]
fn homo_total_loss_correct_formula() {
    // Two tasks, log_sigma = [0, ln(2)]
    // task 0: weight=1, reg=0 → contribution = 1*loss0 + 0
    // task 1: weight=0.25, reg=ln(2) → contribution = 0.25*loss1 + ln(2)
    let mut w =
        HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)]).unwrap();
    w.update_log_sigma(1, std::f32::consts::LN_2).unwrap();
    let losses = [2.0_f32, 4.0_f32];
    let total = w.total_loss(&losses).unwrap();
    let expected = 1.0 * 2.0 + 0.0 + 0.25 * 4.0 + std::f32::consts::LN_2;
    assert!((total - expected).abs() < 1e-4);
}

#[test]
fn homo_total_loss_dimension_mismatch() {
    let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
    assert!(matches!(
        w.total_loss(&[1.0, 2.0]),
        Err(LossWeightError::DimensionMismatch { .. })
    ));
}

#[test]
fn homo_weights_all_computed() {
    let mut w =
        HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)]).unwrap();
    w.update_log_sigma(0, 0.0).unwrap();
    w.update_log_sigma(1, 1.0).unwrap();
    let weights = w.weights().unwrap();
    assert_eq!(weights.len(), 2);
    assert!((weights[0] - 1.0).abs() < 1e-5);
    assert!((weights[1] - (-2.0_f32).exp()).abs() < 1e-5);
}

#[test]
fn homo_weight_is_clipped_to_task_bounds() {
    // Regression: LossTask::min_weight/max_weight are documented as
    // hard bounds but were previously ignored by weight()/total_loss().
    let mut w_hi =
        HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0).with_bounds(0.1, 2.0)]).unwrap();
    // log_sigma very negative → raw weight exp(-2*log_sigma) explodes
    // far past max_weight=2.0.
    w_hi.update_log_sigma(0, -10.0).unwrap();
    let weight_hi = w_hi.weight(0).unwrap();
    assert!((weight_hi - 2.0).abs() < 1e-5, "weight_hi={weight_hi}");

    let mut w_lo =
        HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0).with_bounds(0.1, 2.0)]).unwrap();
    // log_sigma very positive → raw weight collapses far below min_weight=0.1.
    w_lo.update_log_sigma(0, 10.0).unwrap();
    let weight_lo = w_lo.weight(0).unwrap();
    assert!((weight_lo - 0.1).abs() < 1e-5, "weight_lo={weight_lo}");

    // total_loss() must apply the same clipping to its per-task weight.
    let total = w_hi.total_loss(&[3.0]).unwrap();
    let expected = 2.0 * 3.0 + w_hi.log_sigmas()[0];
    assert!((total - expected).abs() < 1e-4, "total={total}");
}

#[test]
fn homo_update_log_sigma_applies_delta() {
    let mut w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
    w.update_log_sigma(0, 0.3).unwrap();
    assert!((w.log_sigmas()[0] - 0.3).abs() < 1e-6);
    w.update_log_sigma(0, 0.2).unwrap();
    assert!((w.log_sigmas()[0] - 0.5).abs() < 1e-6);
}

#[test]
fn homo_update_log_sigma_out_of_bounds() {
    let mut w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
    assert!(w.update_log_sigma(5, 0.1).is_err());
}

#[test]
fn homo_reset_sets_log_sigmas_to_zero() {
    let mut w =
        HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)]).unwrap();
    w.update_log_sigma(0, 2.5).unwrap();
    w.update_log_sigma(1, -1.0).unwrap();
    w.reset();
    for &ls in w.log_sigmas() {
        assert!((ls - 0.0).abs() < 1e-6);
    }
}

// ── GradNormWeighter ──────────────────────────────────────────────────────

#[test]
fn gradnorm_new_empty_error() {
    let r = GradNormWeighter::new(vec![], 1.5, 0.9);
    assert!(matches!(r, Err(LossWeightError::EmptyTaskList)));
}

#[test]
fn gradnorm_new_correct_alpha() {
    let g = GradNormWeighter::new(
        vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
        1.5,
        0.9,
    )
    .unwrap();
    assert!((g.alpha - 1.5).abs() < 1e-6);
}

#[test]
fn gradnorm_new_invalid_ema_decay() {
    let r = GradNormWeighter::new(vec![LossTask::new("a", 1.0)], 1.5, 1.0);
    assert!(r.is_err());
}

#[test]
fn gradnorm_update_increments_step() {
    let mut g = GradNormWeighter::new(
        vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
        1.5,
        0.9,
    )
    .unwrap();
    g.update(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
    assert_eq!(g.step(), 1);
    g.update(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
    assert_eq!(g.step(), 2);
}

#[test]
fn gradnorm_update_uniform_norms_gives_uniform_weights() {
    let mut g = GradNormWeighter::new(
        vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
        1.5,
        0.0, // no EMA smoothing for a clean test
    )
    .unwrap();
    // Uniform gradient norms and uniform loss ratios should yield equal weights.
    for _ in 0..5 {
        g.update(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
    }
    let w = g.weights();
    assert!(
        (w[0] - w[1]).abs() < 0.1,
        "weights should be near-equal: {:?}",
        w
    );
}

#[test]
fn gradnorm_update_dimension_mismatch_losses() {
    let mut g = GradNormWeighter::new(
        vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
        1.5,
        0.9,
    )
    .unwrap();
    assert!(matches!(
        g.update(&[1.0], &[1.0, 1.0]),
        Err(LossWeightError::DimensionMismatch { .. })
    ));
}

#[test]
fn gradnorm_update_dimension_mismatch_norms() {
    let mut g = GradNormWeighter::new(
        vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
        1.5,
        0.9,
    )
    .unwrap();
    assert!(matches!(
        g.update(&[1.0, 1.0], &[1.0]),
        Err(LossWeightError::DimensionMismatch { .. })
    ));
}

#[test]
fn gradnorm_weights_update_based_on_norms() {
    let mut g = GradNormWeighter::new(
        vec![
            LossTask::new("a", 1.0).with_bounds(0.001, 1000.0),
            LossTask::new("b", 1.0).with_bounds(0.001, 1000.0),
        ],
        1.0,
        0.0,
    )
    .unwrap();
    // Task b has 10x larger gradient norm → b should get lower weight.
    g.update(&[1.0, 1.0], &[1.0, 10.0]).unwrap();
    let w = g.weights();
    // After normalization, task a (smaller norm) should have higher weight.
    assert!(w[0] > w[1], "a={:.4}, b={:.4}", w[0], w[1]);
}

#[test]
fn gradnorm_new_invalid_alpha() {
    assert!(GradNormWeighter::new(vec![LossTask::new("a", 1.0)], f32::NAN, 0.9).is_err());
    assert!(GradNormWeighter::new(vec![LossTask::new("a", 1.0)], -1.0, 0.9).is_err());
}

#[test]
fn gradnorm_primary_task_weight_is_pinned() {
    // Regression: LossTask::is_primary was documented as anchoring
    // training to the dominant objective but was never consulted by
    // any strategy.
    let mut g = GradNormWeighter::new(
        vec![
            LossTask::new("photo", 2.0).primary(),
            LossTask::new("reg", 1.0),
        ],
        1.5,
        0.0,
    )
    .unwrap();
    // Heavily unbalanced gradient norms would normally shift every
    // weight; the primary task's weight must stay pinned.
    for _ in 0..5 {
        g.update(&[1.0, 1.0], &[1.0, 50.0]).unwrap();
    }
    let w = g.weights();
    assert!(
        (w[0] - 2.0).abs() < 1e-5,
        "primary weight drifted: {}",
        w[0]
    );
}

#[test]
fn gradnorm_negative_loss_ratio_does_not_poison_weights() {
    // Regression: a negative current loss (e.g. a signed regularizer)
    // makes the relative training rate negative; `powf` of a negative
    // base with the non-integer default alpha is NaN unless clamped.
    let mut g = GradNormWeighter::new(
        vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
        1.5,
        0.0,
    )
    .unwrap();
    g.update(&[1.0, 1.0], &[1.0, 1.0]).unwrap(); // seeds initial_losses
    g.update(&[-2.0, 1.0], &[1.0, 1.0]).unwrap();
    let w = g.weights();
    assert!(w.iter().all(|v| v.is_finite()), "weights={:?}", w);
}

#[test]
fn gradnorm_weights_stay_within_tight_bounds_after_rescale() {
    // Regression: `update()` previously clipped each weight to
    // [min_weight, max_weight] and THEN unconditionally rescaled the
    // whole vector so the mean matched the initial mean, which could
    // push an already-clipped weight straight back out of bounds. The
    // fix clips as the LAST step of each rescale/clip iteration, so the
    // hard-bound invariant must hold even with very tight bounds and a
    // heavily lopsided gradient-norm ratio that forces a large rescale.
    let mut g = GradNormWeighter::new(
        vec![
            LossTask::new("a", 1.0).with_bounds(0.9, 1.1),
            LossTask::new("b", 1.0).with_bounds(0.9, 1.1),
        ],
        1.5,
        0.0,
    )
    .unwrap();
    g.update(&[1.0, 1.0], &[1.0, 1.0]).unwrap(); // seeds initial_losses

    // A 1000x gradient-norm imbalance drives a large target-norm spread
    // and hence a large rescale factor.
    g.update(&[1.0, 1.0], &[1.0, 1000.0]).unwrap();
    for &w in g.weights() {
        assert!(
            (0.9..=1.1).contains(&w),
            "weight {w} escaped tight bounds [0.9, 1.1]"
        );
    }
}

// ── WeightScheduleKind ────────────────────────────────────────────────────

#[test]
fn schedule_constant_returns_same() {
    let s = WeightScheduleKind::Constant(3.5);
    assert!((s.weight_at(0) - 3.5).abs() < 1e-6);
    assert!((s.weight_at(999) - 3.5).abs() < 1e-6);
}

#[test]
fn schedule_linear_start() {
    let s = WeightScheduleKind::Linear {
        start: 0.0,
        end: 1.0,
        n_steps: 100,
    };
    assert!((s.weight_at(0) - 0.0).abs() < 1e-5);
}

#[test]
fn schedule_linear_end() {
    let s = WeightScheduleKind::Linear {
        start: 0.0,
        end: 1.0,
        n_steps: 100,
    };
    assert!((s.weight_at(100) - 1.0).abs() < 1e-5);
}

#[test]
fn schedule_linear_midpoint() {
    let s = WeightScheduleKind::Linear {
        start: 0.0,
        end: 2.0,
        n_steps: 100,
    };
    let mid = s.weight_at(50);
    assert!((mid - 1.0).abs() < 1e-4, "mid={mid}");
}

#[test]
fn schedule_cosine_start() {
    let s = WeightScheduleKind::Cosine {
        start: 1.0,
        end: 0.0,
        n_steps: 100,
    };
    assert!((s.weight_at(0) - 1.0).abs() < 1e-5);
}

#[test]
fn schedule_cosine_end() {
    let s = WeightScheduleKind::Cosine {
        start: 1.0,
        end: 0.0,
        n_steps: 100,
    };
    assert!((s.weight_at(100) - 0.0).abs() < 1e-5);
}

#[test]
fn schedule_cosine_monotone_decreasing() {
    let s = WeightScheduleKind::Cosine {
        start: 1.0,
        end: 0.0,
        n_steps: 10,
    };
    let mut prev = s.weight_at(0);
    for step in 1..=10 {
        let cur = s.weight_at(step);
        assert!(
            cur <= prev + 1e-4,
            "not monotone at step {step}: prev={prev}, cur={cur}"
        );
        prev = cur;
    }
}

#[test]
fn schedule_exponential_decays() {
    let s = WeightScheduleKind::Exponential {
        start: 1.0,
        decay: 0.5,
    };
    assert!((s.weight_at(0) - 1.0).abs() < 1e-6);
    assert!((s.weight_at(1) - 0.5).abs() < 1e-6);
    assert!((s.weight_at(2) - 0.25).abs() < 1e-6);
    assert!((s.weight_at(3) - 0.125).abs() < 1e-6);
}

#[test]
fn schedule_piecewise_exact_keyframes() {
    let s = WeightScheduleKind::Piecewise {
        keyframes: vec![(0, 1.0), (10, 2.0), (20, 0.5)],
    };
    assert!((s.weight_at(0) - 1.0).abs() < 1e-5);
    assert!((s.weight_at(10) - 2.0).abs() < 1e-5);
    assert!((s.weight_at(20) - 0.5).abs() < 1e-5);
}

#[test]
fn schedule_piecewise_interpolation() {
    let s = WeightScheduleKind::Piecewise {
        keyframes: vec![(0, 0.0), (10, 10.0)],
    };
    // step 5 should be exactly 5.0
    let v = s.weight_at(5);
    assert!((v - 5.0).abs() < 1e-4, "v={v}");
}

#[test]
fn schedule_piecewise_before_first_keyframe() {
    let s = WeightScheduleKind::Piecewise {
        keyframes: vec![(5, 2.0), (10, 4.0)],
    };
    // steps before the first keyframe clamp to the first value
    assert!((s.weight_at(0) - 2.0).abs() < 1e-5);
    assert!((s.weight_at(3) - 2.0).abs() < 1e-5);
}

#[test]
fn schedule_piecewise_after_last_keyframe() {
    let s = WeightScheduleKind::Piecewise {
        keyframes: vec![(0, 1.0), (10, 3.0)],
    };
    assert!((s.weight_at(100) - 3.0).abs() < 1e-5);
}

#[test]
fn schedule_piecewise_unsorted_keyframes_do_not_panic() {
    // Regression: constructing this variant directly (bypassing
    // `ScheduledWeighter::new`'s validation) with out-of-order
    // keyframes must not panic (underflow / out-of-bounds index) —
    // it should return a defined, if approximate, value.
    let s = WeightScheduleKind::Piecewise {
        keyframes: vec![(10, 1.0), (0, 2.0), (20, 3.0)],
    };
    let _ = s.weight_at(5);
    let _ = s.weight_at(15);
}

// ── ScheduledWeighter ─────────────────────────────────────────────────────

#[test]
fn scheduled_new_empty_error() {
    assert!(matches!(
        ScheduledWeighter::new(vec![]),
        Err(LossWeightError::EmptyTaskList)
    ));
}

#[test]
fn scheduled_new_rejects_unsorted_piecewise_keyframes() {
    let result = ScheduledWeighter::new(vec![TaskWeightSchedule {
        task_name: "reg".into(),
        schedule: WeightScheduleKind::Piecewise {
            keyframes: vec![(10, 1.0), (0, 2.0)],
        },
    }]);
    assert!(matches!(result, Err(LossWeightError::InvalidConfig(_))));
}

#[test]
fn scheduled_weights_at_correct() {
    let sw = ScheduledWeighter::new(vec![
        TaskWeightSchedule {
            task_name: "photo".into(),
            schedule: WeightScheduleKind::Constant(2.0),
        },
        TaskWeightSchedule {
            task_name: "reg".into(),
            schedule: WeightScheduleKind::Linear {
                start: 0.0,
                end: 1.0,
                n_steps: 10,
            },
        },
    ])
    .unwrap();
    let w = sw.weights_at(5);
    assert!((w[0] - 2.0).abs() < 1e-5);
    assert!((w[1] - 0.5).abs() < 1e-4);
}

#[test]
fn scheduled_advance_increments_step() {
    let mut sw = ScheduledWeighter::new(vec![TaskWeightSchedule {
        task_name: "x".into(),
        schedule: WeightScheduleKind::Constant(1.0),
    }])
    .unwrap();
    assert_eq!(sw.step(), 0);
    sw.advance();
    assert_eq!(sw.step(), 1);
    sw.advance();
    assert_eq!(sw.step(), 2);
}

#[test]
fn scheduled_current_weights_tracks_step() {
    let mut sw = ScheduledWeighter::new(vec![TaskWeightSchedule {
        task_name: "x".into(),
        schedule: WeightScheduleKind::Linear {
            start: 0.0,
            end: 10.0,
            n_steps: 10,
        },
    }])
    .unwrap();
    sw.advance(); // step = 1
    let w = sw.current_weights();
    assert!((w[0] - 1.0).abs() < 1e-4, "w={}", w[0]);
}

#[test]
fn scheduled_schedule_for_found() {
    let sw = ScheduledWeighter::new(vec![
        TaskWeightSchedule {
            task_name: "alpha".into(),
            schedule: WeightScheduleKind::Constant(1.0),
        },
        TaskWeightSchedule {
            task_name: "beta".into(),
            schedule: WeightScheduleKind::Constant(2.0),
        },
    ])
    .unwrap();
    let s = sw.schedule_for("beta");
    assert!(s.is_some());
    assert_eq!(s.unwrap().task_name, "beta");
}

#[test]
fn scheduled_schedule_for_not_found() {
    let sw = ScheduledWeighter::new(vec![TaskWeightSchedule {
        task_name: "alpha".into(),
        schedule: WeightScheduleKind::Constant(1.0),
    }])
    .unwrap();
    assert!(sw.schedule_for("gamma").is_none());
}

#[test]
fn scheduled_task_names() {
    let sw = ScheduledWeighter::new(vec![
        TaskWeightSchedule {
            task_name: "photo".into(),
            schedule: WeightScheduleKind::Constant(1.0),
        },
        TaskWeightSchedule {
            task_name: "reg".into(),
            schedule: WeightScheduleKind::Constant(0.1),
        },
    ])
    .unwrap();
    let names = sw.task_names();
    assert_eq!(names, vec!["photo", "reg"]);
}

// ── LossStatTracker ───────────────────────────────────────────────────────

#[test]
fn stat_tracker_zero_tasks_error() {
    assert!(matches!(
        LossStatTracker::new(0, 0.9),
        Err(LossWeightError::EmptyTaskList)
    ));
}

#[test]
fn stat_tracker_invalid_ema_decay() {
    assert!(LossStatTracker::new(2, 0.0).is_err());
    assert!(LossStatTracker::new(2, 1.0).is_err());
}

#[test]
fn stat_tracker_update_ema_converges() {
    let mut t = LossStatTracker::new(1, 0.99).unwrap();
    // Feed many identical losses; the EMA mean should converge to the value.
    for _ in 0..500 {
        t.update(&[5.0]).unwrap();
    }
    assert!((t.means()[0] - 5.0).abs() < 0.1, "mean={}", t.means()[0]);
}

#[test]
fn stat_tracker_first_update_seeds_mean_exactly() {
    // Regression: the mean/variance were previously EMA-blended against
    // an artificial mean=0.0/variance=1.0 starting point, so even a
    // single observation of a large loss would read far below its true
    // value (with decay=0.9, only 10% of it) until many steps had
    // passed.
    let mut t = LossStatTracker::new(1, 0.9).unwrap();
    t.update(&[1000.0]).unwrap();
    assert!(
        (t.means()[0] - 1000.0).abs() < 1e-3,
        "mean={}",
        t.means()[0]
    );
    assert!(
        (t.variances()[0]).abs() < 1e-6,
        "variance={}",
        t.variances()[0]
    );
}

#[test]
fn stat_tracker_update_dimension_mismatch() {
    let mut t = LossStatTracker::new(2, 0.9).unwrap();
    assert!(matches!(
        t.update(&[1.0]),
        Err(LossWeightError::DimensionMismatch { .. })
    ));
}

#[test]
fn stat_tracker_inverse_variance_uniform() {
    let mut t = LossStatTracker::new(2, 0.9).unwrap();
    // Feed identical losses to both tasks → variances should equalize.
    for _ in 0..100 {
        t.update(&[1.0, 1.0]).unwrap();
    }
    let w = t.inverse_variance_weights(1e-4);
    // Both should have the same weight.
    assert!((w[0] - w[1]).abs() < 0.01, "w={:?}", w);
}

#[test]
fn stat_tracker_inverse_variance_high_var_lower_weight() {
    let mut t = LossStatTracker::new(2, 0.5).unwrap();
    // Task 0: constant; task 1: oscillating → higher variance.
    let mut sign = 1.0_f32;
    for _ in 0..200 {
        t.update(&[1.0, 1.0 + sign * 10.0]).unwrap();
        sign = -sign;
    }
    let w = t.inverse_variance_weights(1e-6);
    assert!(w[0] > w[1], "task0={:.4}, task1={:.4}", w[0], w[1]);
}

#[test]
fn stat_tracker_relative_magnitude_uniform() {
    let mut t = LossStatTracker::new(3, 0.5).unwrap();
    for _ in 0..200 {
        t.update(&[2.0, 2.0, 2.0]).unwrap();
    }
    let w = t.relative_magnitude_weights().unwrap();
    for wi in &w {
        assert!((wi - 1.0).abs() < 0.01, "wi={wi}");
    }
}

#[test]
fn stat_tracker_step_increments() {
    let mut t = LossStatTracker::new(1, 0.9).unwrap();
    assert_eq!(t.step(), 0);
    t.update(&[1.0]).unwrap();
    assert_eq!(t.step(), 1);
}

// ── Utility functions ─────────────────────────────────────────────────────

#[test]
fn normalize_weights_sums_to_n_tasks() {
    let w = alw_normalize_weights(&[1.0, 2.0, 3.0]).unwrap();
    let n = 3.0_f32;
    assert!((w.iter().sum::<f32>() - n).abs() < 1e-5);
}

#[test]
fn normalize_weights_empty_error() {
    assert!(matches!(
        alw_normalize_weights(&[]),
        Err(LossWeightError::EmptyTaskList)
    ));
}

#[test]
fn normalize_weights_all_zero_error() {
    assert!(alw_normalize_weights(&[0.0, 0.0]).is_err());
}

#[test]
fn clip_weights_clips_to_bounds() {
    let tasks = vec![
        LossTask::new("a", 1.0).with_bounds(0.5, 2.0),
        LossTask::new("b", 1.0).with_bounds(0.5, 2.0),
    ];
    let clipped = alw_clip_weights(&[0.1, 5.0], &tasks);
    assert!((clipped[0] - 0.5).abs() < 1e-6);
    assert!((clipped[1] - 2.0).abs() < 1e-6);
}

#[test]
fn clip_weights_in_bounds_unchanged() {
    let tasks = vec![LossTask::new("a", 1.0).with_bounds(0.1, 10.0)];
    let clipped = alw_clip_weights(&[1.5], &tasks);
    assert!((clipped[0] - 1.5).abs() < 1e-6);
}

#[test]
fn relative_training_rate_no_change() {
    // When losses don't change, all rates should be 1.0.
    let rates = alw_relative_training_rate(&[2.0, 3.0], &[2.0, 3.0]).unwrap();
    for r in &rates {
        assert!((r - 1.0).abs() < 1e-4, "r={r}");
    }
}

#[test]
fn relative_training_rate_one_task_doubled() {
    // Task 0 doubled, task 1 unchanged → task 0 rate is 2, task 1 is 1.
    // Mean ratio = 1.5, so r0=2/1.5≈1.333, r1=1/1.5≈0.667.
    let rates = alw_relative_training_rate(&[2.0, 1.0], &[1.0, 1.0]).unwrap();
    assert!(rates[0] > rates[1], "r0={:.4} r1={:.4}", rates[0], rates[1]);
    let expected_r0 = 2.0_f32 / 1.5;
    assert!((rates[0] - expected_r0).abs() < 1e-3, "r0={}", rates[0]);
}

#[test]
fn relative_training_rate_dimension_mismatch() {
    assert!(matches!(
        alw_relative_training_rate(&[1.0], &[1.0, 1.0]),
        Err(LossWeightError::DimensionMismatch { .. })
    ));
}

#[test]
fn weighted_sum_known_values() {
    let s = alw_weighted_sum(&[2.0, 3.0], &[4.0, 5.0]).unwrap();
    assert!((s - 23.0).abs() < 1e-5);
}

#[test]
fn weighted_sum_mismatch_error() {
    assert!(matches!(
        alw_weighted_sum(&[1.0], &[1.0, 2.0]),
        Err(LossWeightError::DimensionMismatch { .. })
    ));
}

#[test]
fn imbalance_ratio_all_equal() {
    assert!((alw_imbalance_ratio(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-5);
}

#[test]
fn imbalance_ratio_known_ratio() {
    let r = alw_imbalance_ratio(&[1.0, 4.0]);
    assert!((r - 4.0).abs() < 1e-4, "r={r}");
}

#[test]
fn imbalance_ratio_empty() {
    assert!((alw_imbalance_ratio(&[]) - 1.0).abs() < 1e-6);
}

// ── WeightHistory ─────────────────────────────────────────────────────────

#[test]
fn weight_history_record_length_grows() {
    let mut h = WeightHistory::new(vec!["a".into(), "b".into()]);
    assert_eq!(h.len(), 0);
    h.record(&[1.0, 2.0]).unwrap();
    assert_eq!(h.len(), 1);
    h.record(&[1.5, 2.5]).unwrap();
    assert_eq!(h.len(), 2);
}

#[test]
fn weight_history_record_capped_at_1000() {
    let mut h = WeightHistory::new(vec!["a".into()]);
    for i in 0..1100_usize {
        h.record(&[i as f32]).unwrap();
    }
    assert_eq!(h.len(), 1000);
}

#[test]
fn weight_history_record_dimension_mismatch() {
    let mut h = WeightHistory::new(vec!["a".into(), "b".into()]);
    assert!(matches!(
        h.record(&[1.0]),
        Err(LossWeightError::DimensionMismatch { .. })
    ));
}

#[test]
fn weight_history_latest_is_last_recorded() {
    let mut h = WeightHistory::new(vec!["a".into()]);
    h.record(&[1.0]).unwrap();
    h.record(&[9.0]).unwrap();
    let latest = h.latest().unwrap();
    assert!((latest[0] - 9.0).abs() < 1e-5);
}

#[test]
fn weight_history_mean_weights_correct() {
    let mut h = WeightHistory::new(vec!["a".into(), "b".into()]);
    h.record(&[2.0, 4.0]).unwrap();
    h.record(&[4.0, 8.0]).unwrap();
    let means = h.mean_weights();
    assert!((means[0] - 3.0).abs() < 1e-5);
    assert!((means[1] - 6.0).abs() < 1e-5);
}

#[test]
fn weight_history_trend_increasing() {
    let mut h = WeightHistory::new(vec!["a".into()]);
    for i in 0..20_usize {
        h.record(&[i as f32]).unwrap();
    }
    let slope = h.weight_trend(0);
    assert!(slope > 0.0, "slope={slope}");
}

#[test]
fn weight_history_trend_constant_near_zero() {
    let mut h = WeightHistory::new(vec!["a".into()]);
    for _ in 0..20 {
        h.record(&[3.1]).unwrap();
    }
    let slope = h.weight_trend(0);
    assert!(slope.abs() < 1e-4, "slope={slope}");
}

#[test]
fn weight_history_trend_no_data_zero() {
    let h = WeightHistory::new(vec!["a".into()]);
    assert!((h.weight_trend(0) - 0.0).abs() < 1e-6);
}

// ── Format functions ──────────────────────────────────────────────────────

#[test]
fn format_weights_nonempty_string() {
    let tasks = vec![LossTask::new("photo", 1.0), LossTask::new("reg", 0.1)];
    let s = alw_format_weights(&tasks, &[1.5, 0.05]);
    assert!(!s.is_empty());
    assert!(s.contains("photo"));
    assert!(s.contains("reg"));
}

#[test]
fn format_history_summary_nonempty() {
    let mut h = WeightHistory::new(vec!["a".into()]);
    h.record(&[1.0]).unwrap();
    let s = alw_format_history_summary(&h);
    assert!(!s.is_empty());
    assert!(s.contains("WeightHistory"));
}

#[test]
fn format_history_summary_empty_history() {
    let h = WeightHistory::new(vec!["a".into()]);
    let s = alw_format_history_summary(&h);
    assert!(s.contains("no data"));
}

// ── Regression (F273): TaskNotFound / NegativeLogSigma are constructible ─────
// Both variants used to be declared but never constructed anywhere in the
// workspace; they now back the by-name lookups and the σ-based constructors.

#[test]
fn homoscedastic_with_sigmas_sets_log_sigmas() {
    let tasks = vec![LossTask::new("photo", 1.0), LossTask::new("reg", 1.0)];
    let w = HomoscedasticWeighter::with_sigmas(tasks, &[1.0, std::f32::consts::E])
        .expect("positive finite sigmas are valid");
    let log_sigmas = w.log_sigmas();
    assert!(log_sigmas[0].abs() < 1e-6, "ln(1) must be 0");
    assert!((log_sigmas[1] - 1.0).abs() < 1e-6, "ln(e) must be 1");
    let sigmas = w.sigmas();
    assert!((sigmas[1] - std::f32::consts::E).abs() < 1e-5);
}

#[test]
fn homoscedastic_with_sigmas_rejects_non_positive_sigma() {
    let tasks = vec![LossTask::new("photo", 1.0), LossTask::new("reg", 1.0)];
    let err = HomoscedasticWeighter::with_sigmas(tasks.clone(), &[1.0, 0.0])
        .expect_err("sigma = 0 has no logarithm");
    assert!(
        matches!(err, LossWeightError::NegativeLogSigma(1)),
        "{err:?}"
    );

    let err = HomoscedasticWeighter::with_sigmas(tasks.clone(), &[-2.0, 1.0])
        .expect_err("negative sigma has no logarithm");
    assert!(
        matches!(err, LossWeightError::NegativeLogSigma(0)),
        "{err:?}"
    );

    let err = HomoscedasticWeighter::with_sigmas(tasks.clone(), &[f32::NAN, 1.0])
        .expect_err("NaN sigma has no logarithm");
    assert!(
        matches!(err, LossWeightError::NegativeLogSigma(0)),
        "{err:?}"
    );

    let err = HomoscedasticWeighter::with_sigmas(tasks, &[1.0])
        .expect_err("one sigma for two tasks is a dimension mismatch");
    assert!(
        matches!(err, LossWeightError::DimensionMismatch { .. }),
        "{err:?}"
    );
}

#[test]
fn homoscedastic_set_sigma_validates_and_updates_weight() {
    let tasks = vec![LossTask::new("photo", 1.0)];
    let mut w = HomoscedasticWeighter::new(tasks).expect("non-empty task list");
    // σ = 1 → weight exp(-2·0) = 1
    let before = w.weight(0).expect("task 0 exists");
    assert!((before - 1.0).abs() < 1e-6);

    w.set_sigma(0, 2.0).expect("positive sigma is valid");
    // σ = 2 → weight exp(-2·ln 2) = 1/4
    let after = w.weight(0).expect("task 0 exists");
    assert!((after - 0.25).abs() < 1e-5, "weight was {after}");

    let err = w
        .set_sigma(0, -1.0)
        .expect_err("negative sigma is rejected");
    assert!(
        matches!(err, LossWeightError::NegativeLogSigma(0)),
        "{err:?}"
    );
    // A rejected update must not have changed the stored parameter.
    let unchanged = w.weight(0).expect("task 0 exists");
    assert!((unchanged - 0.25).abs() < 1e-5);

    let err = w
        .set_sigma(7, 1.0)
        .expect_err("index 7 is out of range for one task");
    assert!(
        matches!(err, LossWeightError::DimensionMismatch { .. }),
        "{err:?}"
    );
}

#[test]
fn homoscedastic_weight_by_name_reports_task_not_found() {
    let tasks = vec![LossTask::new("photo", 1.0), LossTask::new("reg", 1.0)];
    let w = HomoscedasticWeighter::new(tasks).expect("non-empty task list");
    assert_eq!(w.task_index("reg").expect("task exists"), 1);
    let weight = w.weight_by_name("photo").expect("task exists");
    assert!((weight - 1.0).abs() < 1e-6);

    let err = w
        .weight_by_name("typo")
        .expect_err("unknown name must be reported");
    match err {
        LossWeightError::TaskNotFound(name) => assert_eq!(name, "typo"),
        other => panic!("expected TaskNotFound, got {other:?}"),
    }
}

#[test]
fn gradnorm_weight_by_name_reports_task_not_found() {
    let tasks = vec![LossTask::new("photo", 2.0), LossTask::new("reg", 0.5)];
    let w = GradNormWeighter::new(tasks, 1.5, 0.9).expect("valid config");
    let photo = w.weight_by_name("photo").expect("task exists");
    assert!((photo - 2.0).abs() < 1e-6);
    let err = w.weight_by_name("nope").expect_err("unknown name");
    assert!(matches!(err, LossWeightError::TaskNotFound(_)), "{err:?}");
}

#[test]
fn scheduled_weight_for_task_reports_task_not_found() {
    let sw = ScheduledWeighter::new(vec![TaskWeightSchedule {
        task_name: "photo".to_string(),
        schedule: WeightScheduleKind::Constant(3.0),
    }])
    .expect("non-empty schedule list");

    let w = sw.weight_for_task("photo", 42).expect("task exists");
    assert!((w - 3.0).abs() < 1e-6);
    let w_now = sw.current_weight_for_task("photo").expect("task exists");
    assert!((w_now - 3.0).abs() < 1e-6);

    let err = sw
        .weight_for_task("missing", 0)
        .expect_err("unknown name must be reported");
    match err {
        LossWeightError::TaskNotFound(name) => assert_eq!(name, "missing"),
        other => panic!("expected TaskNotFound, got {other:?}"),
    }
}

#[test]
fn weight_history_task_index_reports_task_not_found() {
    let mut h = WeightHistory::new(vec!["photo".to_string(), "reg".to_string()]);
    h.record(&[1.0, 1.0]).expect("two weights for two tasks");
    h.record(&[2.0, 1.0]).expect("two weights for two tasks");

    assert_eq!(h.task_index("reg").expect("task exists"), 1);
    let trend = h.weight_trend_by_name("photo").expect("task exists");
    assert!(trend > 0.0, "photo weight is rising, trend was {trend}");

    let err = h.task_index("ghost").expect_err("unknown name");
    assert!(matches!(err, LossWeightError::TaskNotFound(_)), "{err:?}");
    let err = h.weight_trend_by_name("ghost").expect_err("unknown name");
    assert!(matches!(err, LossWeightError::TaskNotFound(_)), "{err:?}");
}
