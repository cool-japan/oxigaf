//! Test suite for [`crate::rectified_flow`], split into its own file
//! (via a `#[path]`-attributed `mod tests;` declaration) to keep the
//! production-code file under the COOLJAPAN 2000-line policy limit.

use super::*;

// ── rf_interpolate ────────────────────────────────────────────────────────

#[test]
fn test_interpolate_t0_gives_x0() {
    let x0 = vec![1.0, 2.0, 3.0];
    let x1 = vec![4.0, 5.0, 6.0];
    let result = rf_interpolate(&x0, &x1, 0.0).unwrap();
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 2.0).abs() < 1e-6);
    assert!((result[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_interpolate_t1_gives_x1() {
    let x0 = vec![1.0, 2.0, 3.0];
    let x1 = vec![4.0, 5.0, 6.0];
    let result = rf_interpolate(&x0, &x1, 1.0).unwrap();
    assert!((result[0] - 4.0).abs() < 1e-6);
    assert!((result[1] - 5.0).abs() < 1e-6);
    assert!((result[2] - 6.0).abs() < 1e-6);
}

#[test]
fn test_interpolate_t_half_gives_midpoint() {
    let x0 = vec![0.0, 0.0];
    let x1 = vec![2.0, 4.0];
    let result = rf_interpolate(&x0, &x1, 0.5).unwrap();
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_interpolate_dimension_mismatch_returns_error() {
    let x0 = vec![1.0, 2.0];
    let x1 = vec![3.0, 4.0, 5.0];
    assert!(matches!(
        rf_interpolate(&x0, &x1, 0.5),
        Err(RectifiedFlowError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_interpolate_invalid_t_negative() {
    let x0 = vec![0.0];
    let x1 = vec![1.0];
    assert!(matches!(
        rf_interpolate(&x0, &x1, -0.1),
        Err(RectifiedFlowError::InvalidTimestep { .. })
    ));
}

#[test]
fn test_interpolate_invalid_t_above_one() {
    let x0 = vec![0.0];
    let x1 = vec![1.0];
    assert!(matches!(
        rf_interpolate(&x0, &x1, 1.1),
        Err(RectifiedFlowError::InvalidTimestep { .. })
    ));
}

// ── rf_target_velocity ────────────────────────────────────────────────────

#[test]
fn test_target_velocity_correctness() {
    let x0 = vec![1.0, 2.0, 3.0];
    let x1 = vec![4.0, 6.0, 9.0];
    let v = rf_target_velocity(&x0, &x1).unwrap();
    assert!((v[0] - 3.0).abs() < 1e-6);
    assert!((v[1] - 4.0).abs() < 1e-6);
    assert!((v[2] - 6.0).abs() < 1e-6);
}

#[test]
fn test_target_velocity_zero_when_equal() {
    let x = vec![1.0, 2.0, 3.0];
    let v = rf_target_velocity(&x, &x).unwrap();
    for vi in &v {
        assert!(vi.abs() < 1e-6);
    }
}

#[test]
fn test_target_velocity_dimension_mismatch() {
    let x0 = vec![1.0, 2.0];
    let x1 = vec![1.0];
    assert!(matches!(
        rf_target_velocity(&x0, &x1),
        Err(RectifiedFlowError::DimensionMismatch { .. })
    ));
}

// ── rf_flow_matching_loss ─────────────────────────────────────────────────

#[test]
fn test_loss_zero_when_perfect() {
    let x0 = vec![0.0, 0.0];
    let x1 = vec![1.0, 1.0];
    let v_pred = vec![1.0, 1.0]; // exact target
    let loss = rf_flow_matching_loss(&v_pred, &x0, &x1).unwrap();
    assert!(loss.abs() < 1e-6);
}

#[test]
fn test_loss_positive_when_not_matching() {
    let x0 = vec![0.0, 0.0];
    let x1 = vec![1.0, 1.0];
    let v_pred = vec![0.0, 0.0]; // off by 1,1
    let loss = rf_flow_matching_loss(&v_pred, &x0, &x1).unwrap();
    assert!(loss > 0.0);
}

#[test]
fn test_loss_empty_returns_error() {
    assert!(matches!(
        rf_flow_matching_loss(&[], &[], &[]),
        Err(RectifiedFlowError::EmptyBatch)
    ));
}

// ── rf_loss_per_sample ────────────────────────────────────────────────────

#[test]
fn test_loss_per_sample_length_equals_n() {
    let n = 4;
    let d = 3;
    let x0 = vec![0.0f32; n * d];
    let x1 = vec![1.0f32; n * d];
    let v_pred = vec![1.0f32; n * d];
    let losses = rf_loss_per_sample(&v_pred, &x0, &x1, d).unwrap();
    assert_eq!(losses.len(), n);
}

#[test]
fn test_loss_per_sample_zero_when_perfect() {
    let n = 2;
    let d = 2;
    let x0 = vec![0.0f32; n * d];
    let x1 = vec![2.0f32; n * d];
    let v_pred = vec![2.0f32; n * d]; // exact target
    let losses = rf_loss_per_sample(&v_pred, &x0, &x1, d).unwrap();
    for l in losses {
        assert!(l.abs() < 1e-6);
    }
}

// ── rf_weighted_loss ──────────────────────────────────────────────────────

#[test]
fn test_weighted_loss_scales_with_weights() {
    let x0 = vec![0.0, 0.0];
    let x1 = vec![1.0, 1.0];
    let v_pred = vec![0.0, 0.0]; // off by 1
    let w1 = vec![1.0];
    let w2 = vec![2.0];
    let l1 = rf_weighted_loss(&v_pred, &x0, &x1, &w1).unwrap();
    let l2 = rf_weighted_loss(&v_pred, &x0, &x1, &w2).unwrap();
    // Both should give the same normalized loss (weight / weight_sum = 1)
    assert!((l1 - l2).abs() < 1e-6);
}

#[test]
fn test_weighted_loss_dimension_mismatch_wrong_weights() {
    let x0 = vec![0.0, 0.0, 0.0, 0.0]; // 2 samples of d=2
    let x1 = vec![1.0, 1.0, 1.0, 1.0];
    let v_pred = vec![0.0, 0.0, 0.0, 0.0];
    let weights = vec![1.0, 1.0, 1.0]; // wrong: 3 instead of 2
    assert!(matches!(
        rf_weighted_loss(&v_pred, &x0, &x1, &weights),
        Err(RectifiedFlowError::DimensionMismatch { .. })
    ));
}

// ── rf_linspace ───────────────────────────────────────────────────────────

#[test]
fn test_linspace_first_and_last() {
    let v = rf_linspace(0.0, 1.0, 11);
    assert_eq!(v.len(), 11);
    assert!((v[0] - 0.0).abs() < 1e-6);
    assert!((v[10] - 1.0).abs() < 1e-6);
}

#[test]
fn test_linspace_evenly_spaced() {
    let v = rf_linspace(0.0, 1.0, 5);
    let step = 0.25;
    for (i, &val) in v.iter().enumerate().take(5) {
        assert!((val - i as f32 * step).abs() < 1e-5);
    }
}

#[test]
fn test_linspace_n1_gives_start() {
    let v = rf_linspace(3.0, 7.0, 1);
    assert_eq!(v.len(), 1);
    assert!((v[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_linspace_n2_gives_endpoints() {
    let v = rf_linspace(2.0, 5.0, 2);
    assert_eq!(v.len(), 2);
    assert!((v[0] - 2.0).abs() < 1e-6);
    assert!((v[1] - 5.0).abs() < 1e-6);
}

#[test]
fn test_linspace_n0_empty() {
    let v = rf_linspace(0.0, 1.0, 0);
    assert!(v.is_empty());
}

// ── rf_sample_t ───────────────────────────────────────────────────────────

#[test]
fn test_sample_t_in_range() {
    let ts = rf_sample_t(42, 100, 0.001, 0.999);
    for t in &ts {
        assert!(*t >= 0.001 && *t <= 0.999, "t={} out of range", t);
    }
}

#[test]
fn test_sample_t_correct_length() {
    let ts = rf_sample_t(1, 32, 0.0, 1.0);
    assert_eq!(ts.len(), 32);
}

#[test]
fn test_sample_t_deterministic() {
    let ts1 = rf_sample_t(123, 10, 0.0, 1.0);
    let ts2 = rf_sample_t(123, 10, 0.0, 1.0);
    assert_eq!(ts1, ts2);
}

#[test]
fn test_sample_t_different_seeds_differ() {
    let ts1 = rf_sample_t(42, 10, 0.0, 1.0);
    let ts2 = rf_sample_t(43, 10, 0.0, 1.0);
    assert_ne!(ts1, ts2);
}

// ── rf_logit_normal_t ─────────────────────────────────────────────────────

#[test]
fn test_logit_normal_t_in_range() {
    let ts = rf_logit_normal_t(7, 200, 0.0, 1.0, 0.001, 0.999);
    for t in &ts {
        assert!(*t >= 0.001 && *t <= 0.999, "t={} out of range", t);
    }
}

#[test]
fn test_logit_normal_t_correct_length() {
    let ts = rf_logit_normal_t(99, 50, 0.0, 1.0, 0.0, 1.0);
    assert_eq!(ts.len(), 50);
}

#[test]
fn test_logit_normal_t_deterministic() {
    let ts1 = rf_logit_normal_t(55, 20, 0.0, 1.0, 0.0, 1.0);
    let ts2 = rf_logit_normal_t(55, 20, 0.0, 1.0, 0.0, 1.0);
    assert_eq!(ts1, ts2);
}

#[test]
fn test_logit_normal_t_bell_shaped_near_half() {
    // For mean=0, std=1: sigmoid(N(0,1)) should concentrate near 0.5
    let ts = rf_logit_normal_t(42, 1000, 0.0, 1.0, 0.0, 1.0);
    let mean: f32 = ts.iter().sum::<f32>() / ts.len() as f32;
    assert!((mean - 0.5).abs() < 0.05, "mean={} not near 0.5", mean);
}

// ── rf_make_batch ─────────────────────────────────────────────────────────

#[test]
fn test_make_batch_correct_shapes() {
    let n = 4;
    let d = 3;
    let x1 = vec![1.0f32; n * d];
    let ts = vec![0.5f32; n];
    let batch = rf_make_batch(&x1, n, d, &ts, 42).unwrap();
    assert_eq!(batch.x0.len(), n * d);
    assert_eq!(batch.x_t.len(), n * d);
    assert_eq!(batch.target_v.len(), n * d);
    assert_eq!(batch.t.len(), n);
    assert_eq!(batch.n, n);
    assert_eq!(batch.d, d);
}

#[test]
fn test_make_batch_target_v_is_x1_minus_x0() {
    let n = 2;
    let d = 2;
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let ts = vec![0.3, 0.7];
    let batch = rf_make_batch(&x1, n, d, &ts, 1).unwrap();
    for i in 0..n * d {
        let expected = batch.x1[i] - batch.x0[i];
        assert!((batch.target_v[i] - expected).abs() < 1e-6);
    }
}

#[test]
fn test_make_batch_deterministic() {
    let n = 3;
    let d = 4;
    let x1 = vec![0.5f32; n * d];
    let ts = vec![0.2, 0.5, 0.8];
    let b1 = rf_make_batch(&x1, n, d, &ts, 77).unwrap();
    let b2 = rf_make_batch(&x1, n, d, &ts, 77).unwrap();
    assert_eq!(b1.x0, b2.x0);
}

#[test]
fn test_make_batch_noise_approx_gaussian() {
    // With N=1000, d=1, x0 should be approximately N(0,1)
    let n = 1000;
    let d = 1;
    let x1 = vec![0.0f32; n * d];
    let ts = vec![0.5f32; n];
    let batch = rf_make_batch(&x1, n, d, &ts, 314).unwrap();
    let mean: f32 = batch.x0.iter().sum::<f32>() / n as f32;
    let var: f32 = batch
        .x0
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f32>()
        / n as f32;
    let std = var.sqrt();
    assert!(mean.abs() < 0.15, "noise mean={} not near 0", mean);
    assert!((std - 1.0).abs() < 0.15, "noise std={} not near 1", std);
}

// ── ODE step functions ────────────────────────────────────────────────────

#[test]
fn test_euler_step_correctness() {
    let x = vec![1.0, 2.0];
    let v = vec![0.5, -0.5];
    let dt = 0.1;
    let next = rf_euler_step(&x, &v, dt);
    assert!((next[0] - 1.05).abs() < 1e-6);
    assert!((next[1] - 1.95).abs() < 1e-6);
}

#[test]
fn test_heun_step_with_equal_velocities_matches_euler() {
    let x = vec![1.0, 2.0];
    let v = vec![0.5, -0.3];
    let dt = 0.2;
    let euler = rf_euler_step(&x, &v, dt);
    let heun = rf_heun_step(&x, &v, &v, dt);
    for (e, h) in euler.iter().zip(heun.iter()) {
        assert!((e - h).abs() < 1e-6);
    }
}

#[test]
fn test_midpoint_step_correctness() {
    // x_next = x + dt * v_mid
    let x = vec![0.0, 0.0];
    let v_start = vec![1.0, 1.0]; // used externally to get x_mid
    let v_mid = vec![2.0, 3.0];
    let dt = 0.1;
    let next = rf_midpoint_step(&x, &v_start, &v_mid, dt);
    assert!((next[0] - 0.2).abs() < 1e-6);
    assert!((next[1] - 0.3).abs() < 1e-6);
}

#[test]
fn test_rk4_step_equal_k_matches_euler() {
    // If k1=k2=k3=k4=v, rk4 = x + dt/6*(v + 2v + 2v + v) = x + dt*v (Euler)
    let x = vec![1.0, 2.0];
    let v = vec![0.3, -0.1];
    let dt = 0.5;
    let euler = rf_euler_step(&x, &v, dt);
    let rk4 = rf_rk4_step(&x, &v, &v, &v, &v, dt);
    for (e, r) in euler.iter().zip(rk4.iter()) {
        assert!((e - r).abs() < 1e-5);
    }
}

// Regression: rf_rk4_step used to index k1[i]/k2[i]/k3[i]/k4[i] directly
// and panic when any was shorter than x; it must not panic now.
#[test]
fn test_rk4_step_short_k_does_not_panic() {
    let x = vec![1.0, 2.0, 3.0];
    let short = vec![0.1]; // shorter than x
    let _ = rf_rk4_step(&x, &short, &short, &short, &short, 0.1);
}

// Regression: RfOdeSolver::integrate's Rk4 arm used to check only k1's
// length, letting a short k2/k3/k4 from a buggy velocity_fn reach
// rf_rk4_step and panic; it must now surface as IntegrationFailed.
#[test]
fn test_solver_integrate_rk4_rejects_short_k2() {
    let config = RectifiedFlowConfig {
        n_steps: 1,
        solver: RfSolverKind::Rk4,
        ..Default::default()
    };
    let solver = RfOdeSolver::new(config).unwrap();
    let sched = solver.generate_t_schedule();
    let x0 = vec![0.0f32; 2];
    // 1st call is k1 (correct length 2); every later call (k2/k3/k4) is
    // short, which the pre-fix code never checked.
    let call = std::cell::Cell::new(0usize);
    let result = solver.integrate(&x0, &sched, |_, _| {
        let n = call.get();
        call.set(n + 1);
        if n == 0 {
            vec![1.0, 1.0]
        } else {
            vec![1.0]
        }
    });
    assert!(matches!(
        result,
        Err(RectifiedFlowError::IntegrationFailed { .. })
    ));
}

// ── rf_euler_integrate ────────────────────────────────────────────────────

#[test]
fn test_euler_integrate_state_count() {
    let x0 = vec![0.0, 0.0];
    let velocities = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
    let times = vec![0.0, 0.5, 1.0];
    let traj = rf_euler_integrate(&x0, &velocities, &times).unwrap();
    assert_eq!(traj.states.len(), 3); // n_steps+1 = 3
    assert_eq!(traj.n_steps, 2);
}

#[test]
fn test_euler_integrate_last_state_closer_to_x1() {
    // Starting at x0=[0,0], constant velocity toward x1=[1,0]
    let x0 = vec![0.0f32, 0.0];
    let x1 = [1.0f32, 0.0];
    let n_steps = 10;
    let v: Vec<f32> = x1.iter().zip(x0.iter()).map(|(&b, &a)| b - a).collect();
    let velocities = vec![v; n_steps];
    let times = rf_linspace(0.0, 1.0, n_steps + 1);
    let traj = rf_euler_integrate(&x0, &velocities, &times).unwrap();
    let last = &traj.states[n_steps];
    let dist_to_x1: f32 = last
        .iter()
        .zip(x1.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    let dist_to_x0: f32 = last
        .iter()
        .zip(x0.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    assert!(dist_to_x1 < dist_to_x0);
}

// ── RfOdeSolver ───────────────────────────────────────────────────────────

#[test]
fn test_solver_new_valid_config() {
    let config = RectifiedFlowConfig::default();
    assert!(RfOdeSolver::new(config).is_ok());
}

#[test]
fn test_solver_new_invalid_n_steps_zero() {
    let config = RectifiedFlowConfig {
        n_steps: 0,
        ..Default::default()
    };
    assert!(matches!(
        RfOdeSolver::new(config),
        Err(RectifiedFlowError::InvalidConfig { .. })
    ));
}

#[test]
fn test_solver_new_invalid_t_min_gte_t_max() {
    let config = RectifiedFlowConfig {
        t_min: 0.9,
        t_max: 0.5,
        ..Default::default()
    };
    assert!(matches!(
        RfOdeSolver::new(config),
        Err(RectifiedFlowError::InvalidConfig { .. })
    ));
}

#[test]
fn test_solver_generate_t_schedule_length() {
    let config = RectifiedFlowConfig::default(); // n_steps=100
    let solver = RfOdeSolver::new(config).unwrap();
    let sched = solver.generate_t_schedule();
    assert_eq!(sched.len(), 101);
}

#[test]
fn test_solver_generate_t_schedule_endpoints() {
    let config = RectifiedFlowConfig::default();
    let solver = RfOdeSolver::new(config).unwrap();
    let sched = solver.generate_t_schedule();
    assert!((sched[0] - 0.0).abs() < 1e-6);
    assert!((sched[100] - 1.0).abs() < 1e-6);
}

#[test]
fn test_solver_integrate_euler_correct_n_steps() {
    let config = RectifiedFlowConfig {
        n_steps: 5,
        solver: RfSolverKind::Euler,
        ..Default::default()
    };
    let solver = RfOdeSolver::new(config).unwrap();
    let sched = solver.generate_t_schedule();
    let x0 = vec![0.0f32; 4];
    let traj = solver
        .integrate(&x0, &sched, |_, _| vec![1.0, 0.0, 0.0, 0.0])
        .unwrap();
    assert_eq!(traj.n_steps, 5);
    assert_eq!(traj.states.len(), 6);
}

#[test]
fn test_solver_integrate_midpoint() {
    let config = RectifiedFlowConfig {
        n_steps: 4,
        solver: RfSolverKind::Midpoint,
        ..Default::default()
    };
    let solver = RfOdeSolver::new(config).unwrap();
    let sched = solver.generate_t_schedule();
    let x0 = vec![0.0f32; 2];
    // Constant velocity: x1 - x0 = [1, 1]
    let traj = solver
        .integrate(&x0, &sched, |_, _| vec![1.0, 1.0])
        .unwrap();
    assert_eq!(traj.n_steps, 4);
}

#[test]
fn test_solver_integrate_heun() {
    let config = RectifiedFlowConfig {
        n_steps: 4,
        solver: RfSolverKind::Heun,
        ..Default::default()
    };
    let solver = RfOdeSolver::new(config).unwrap();
    let sched = solver.generate_t_schedule();
    let x0 = vec![0.0f32; 2];
    let traj = solver
        .integrate(&x0, &sched, |_, _| vec![1.0, 1.0])
        .unwrap();
    assert_eq!(traj.n_steps, 4);
}

#[test]
fn test_solver_integrate_rk4() {
    let config = RectifiedFlowConfig {
        n_steps: 4,
        solver: RfSolverKind::Rk4,
        ..Default::default()
    };
    let solver = RfOdeSolver::new(config).unwrap();
    let sched = solver.generate_t_schedule();
    let x0 = vec![0.0f32; 2];
    let traj = solver
        .integrate(&x0, &sched, |_, _| vec![1.0, 1.0])
        .unwrap();
    assert_eq!(traj.n_steps, 4);
}

// ── ReFlow utilities ──────────────────────────────────────────────────────

#[test]
fn test_reflow_pair_returns_x0_and_final() {
    let x0 = vec![0.0, 0.0];
    let states = vec![vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0]];
    let traj = RfTrajectory {
        states,
        times: vec![0.0, 0.5, 1.0],
        n_steps: 2,
        d: 2,
    };
    let (got_x0, x_final) = rf_reflow_pair(&x0, &traj);
    assert_eq!(got_x0, x0);
    assert_eq!(x_final, vec![1.0, 1.0]);
}

#[test]
fn test_trajectory_curvature_straight_is_zero() {
    // Straight trajectory: all steps move in the same direction
    let states = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ];
    let traj = RfTrajectory {
        states,
        times: vec![0.0, 0.333, 0.667, 1.0],
        n_steps: 3,
        d: 2,
    };
    let curv = rf_trajectory_curvature(&traj);
    assert!(curv.abs() < 1e-5, "curvature={} expected ~0", curv);
}

#[test]
fn test_trajectory_length_positive() {
    let states = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
    let traj = RfTrajectory {
        states,
        times: vec![0.0, 0.5, 1.0],
        n_steps: 2,
        d: 2,
    };
    let len = rf_trajectory_length(&traj);
    assert!(len > 0.0);
    assert!((len - 2.0).abs() < 1e-5);
}

#[test]
fn test_straight_path_length_le_total_length() {
    let states = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 2.0], // curved path
    ];
    let traj = RfTrajectory {
        states,
        times: vec![0.0, 0.5, 1.0],
        n_steps: 2,
        d: 2,
    };
    let total = rf_trajectory_length(&traj);
    let straight = rf_straight_path_length(&traj);
    assert!(straight <= total + 1e-6);
}

#[test]
fn test_straightness_ratio_straight_gives_one() {
    let states = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let traj = RfTrajectory {
        states,
        times: vec![0.0, 0.333, 0.667, 1.0],
        n_steps: 3,
        d: 1,
    };
    let ratio = rf_straightness_ratio(&traj);
    assert!((ratio - 1.0).abs() < 1e-5, "ratio={}", ratio);
}

#[test]
fn test_straightness_ratio_curved_less_than_one() {
    // Detour: go right then back-left, ending up less far than path traveled
    let states = vec![
        vec![0.0, 0.0],
        vec![2.0, 0.0],
        vec![1.0, 0.0], // doubled back
    ];
    let traj = RfTrajectory {
        states,
        times: vec![0.0, 0.5, 1.0],
        n_steps: 2,
        d: 2,
    };
    let ratio = rf_straightness_ratio(&traj);
    assert!(ratio < 1.0, "ratio={} should be <1 for curved path", ratio);
}

// ── RfCoupling ────────────────────────────────────────────────────────────

#[test]
fn test_coupling_independent_sample_x0_shape() {
    let coupling = RfCoupling::new(CouplingMode::Independent);
    let n = 5;
    let d = 4;
    let x1 = vec![0.0f32; n * d];
    let x0 = coupling.sample_x0(&x1, n, d, 42).expect("sample_x0");
    assert_eq!(x0.len(), n * d);
}

#[test]
fn test_coupling_independent_deterministic() {
    let coupling = RfCoupling::new(CouplingMode::Independent);
    let n = 3;
    let d = 2;
    let x1 = vec![0.0f32; n * d];
    let x0a = coupling.sample_x0(&x1, n, d, 99).expect("sample_x0 a");
    let x0b = coupling.sample_x0(&x1, n, d, 99).expect("sample_x0 b");
    assert_eq!(x0a, x0b);
}

// ── rf_greedy_ot_match ────────────────────────────────────────────────────

#[test]
fn test_greedy_ot_match_length_n() {
    let n = 4;
    let d = 2;
    let x0 = vec![0.0f32; n * d];
    let x1 = vec![1.0f32; n * d];
    let assignment = rf_greedy_ot_match(&x0, &x1, n, d).expect("match");
    assert_eq!(assignment.len(), n);
}

// Regression: rf_greedy_ot_match used to index x0/x1 by `n*d` with no
// length check, panicking on a slice-index-out-of-bounds for a
// wrongly-sized batch instead of returning an error.
#[test]
fn test_greedy_ot_match_rejects_wrong_length() {
    let n = 4;
    let d = 2;
    let x0 = vec![0.0f32; n * d];
    let x1_short = vec![1.0f32; n * d - 1];
    assert!(matches!(
        rf_greedy_ot_match(&x0, &x1_short, n, d),
        Err(RectifiedFlowError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        rf_greedy_ot_match(&x0, &x1_short, 0, d),
        Err(RectifiedFlowError::InvalidConfig { .. })
    ));
}

#[test]
fn test_greedy_ot_match_assigns_nearest() {
    // x0 has two rows: [0,0] and [10,10]
    // x1 has two rows: [9,9] and [1,1]
    // Expected: x1[0]=[9,9] nearest x0[1]=[10,10], x1[1]=[1,1] nearest x0[0]=[0,0]
    let n = 2;
    let d = 2;
    let x0 = vec![0.0, 0.0, 10.0, 10.0]; // row0=[0,0], row1=[10,10]
    let x1 = vec![9.0, 9.0, 1.0, 1.0]; // row0=[9,9], row1=[1,1]
    let assignment = rf_greedy_ot_match(&x0, &x1, n, d).expect("match");
    // x1[0]=[9,9] should map to x0[1]=[10,10] (greedy: closest first)
    assert_eq!(assignment[0], 1);
    // x1[1]=[1,1] should map to x0[0]=[0,0] (only one left)
    assert_eq!(assignment[1], 0);
}

// ── rf_compute_stats ──────────────────────────────────────────────────────

#[test]
fn test_compute_stats_correct_mean_min_max() {
    let losses = vec![1.0, 2.0, 3.0];
    let curvatures = vec![0.1, 0.2];
    let straightnesses = vec![0.9, 0.8];
    let stats = rf_compute_stats(&losses, &curvatures, &straightnesses);
    assert!((stats.mean_loss - 2.0).abs() < 1e-6);
    assert!((stats.min_loss - 1.0).abs() < 1e-6);
    assert!((stats.max_loss - 3.0).abs() < 1e-6);
    assert_eq!(stats.n_batches, 3);
}

#[test]
fn test_compute_stats_empty() {
    let stats = rf_compute_stats(&[], &[], &[]);
    assert_eq!(stats.n_batches, 0);
}

// ── rf_format_stats / rf_format_config ────────────────────────────────────

#[test]
fn test_format_stats_non_empty() {
    let stats = RfStats {
        mean_loss: 0.5,
        min_loss: 0.1,
        max_loss: 0.9,
        mean_curvature: 0.01,
        mean_straightness: 0.99,
        n_batches: 10,
    };
    let s = rf_format_stats(&stats);
    assert!(!s.is_empty());
    assert!(s.contains("n_batches"));
}

#[test]
fn test_format_config_non_empty() {
    let config = RectifiedFlowConfig::default();
    let s = rf_format_config(&config);
    assert!(!s.is_empty());
    assert!(s.contains("n_steps"));
}

// ── Additional coverage ───────────────────────────────────────────────────

#[test]
fn test_interpolate_t_quarter() {
    let x0 = vec![0.0, 0.0];
    let x1 = vec![4.0, 8.0];
    let result = rf_interpolate(&x0, &x1, 0.25).unwrap();
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 2.0).abs() < 1e-5);
}

#[test]
fn test_loss_per_sample_empty_input_error() {
    assert!(matches!(
        rf_loss_per_sample(&[], &[], &[], 2),
        Err(RectifiedFlowError::EmptyBatch)
    ));
}

#[test]
fn test_make_batch_wrong_x1_length_error() {
    let x1 = vec![1.0f32; 5]; // should be n*d = 6
    let ts = vec![0.5; 2];
    assert!(matches!(
        rf_make_batch(&x1, 2, 3, &ts, 1),
        Err(RectifiedFlowError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_make_batch_wrong_t_length_error() {
    let n = 3;
    let d = 2;
    let x1 = vec![1.0f32; n * d];
    let ts = vec![0.5f32; 2]; // should be n=3
    assert!(matches!(
        rf_make_batch(&x1, n, d, &ts, 1),
        Err(RectifiedFlowError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_make_batch_invalid_t_in_batch() {
    let n = 2;
    let d = 2;
    let x1 = vec![1.0f32; n * d];
    let ts = vec![0.5f32, 1.5f32]; // 1.5 is invalid
    assert!(matches!(
        rf_make_batch(&x1, n, d, &ts, 1),
        Err(RectifiedFlowError::InvalidTimestep { .. })
    ));
}

#[test]
fn test_heun_step_correctness() {
    let x = vec![0.0, 0.0];
    let v_start = vec![1.0, 0.0];
    let v_end = vec![0.0, 1.0];
    let dt = 1.0;
    // x_next = x + 0.5*(v_start + v_end) = [0.5, 0.5]
    let next = rf_heun_step(&x, &v_start, &v_end, dt);
    assert!((next[0] - 0.5).abs() < 1e-6);
    assert!((next[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_rk4_step_correctness() {
    let x = vec![0.0f32];
    let k1 = vec![1.0f32];
    let k2 = vec![2.0f32];
    let k3 = vec![2.0f32];
    let k4 = vec![3.0f32];
    let dt = 1.0;
    // x + dt/6*(1 + 2*2 + 2*2 + 3) = dt/6*12 = 2.0
    let next = rf_rk4_step(&x, &k1, &k2, &k3, &k4, dt);
    assert!((next[0] - 2.0).abs() < 1e-5);
}

#[test]
fn test_curvature_two_states_is_zero() {
    // Less than 3 states → no angle to compute
    let states = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let traj = RfTrajectory {
        states,
        times: vec![0.0, 1.0],
        n_steps: 1,
        d: 2,
    };
    assert_eq!(rf_trajectory_curvature(&traj), 0.0);
}

#[test]
fn test_straight_path_length_single_state_zero() {
    let states = vec![vec![1.0, 2.0]];
    let traj = RfTrajectory {
        states,
        times: vec![0.0],
        n_steps: 0,
        d: 2,
    };
    assert_eq!(rf_straight_path_length(&traj), 0.0);
}

#[test]
fn test_straightness_ratio_degenerate_zero_length_gives_one() {
    let states = vec![vec![1.0, 1.0], vec![1.0, 1.0]]; // stationary
    let traj = RfTrajectory {
        states,
        times: vec![0.0, 1.0],
        n_steps: 1,
        d: 2,
    };
    let ratio = rf_straightness_ratio(&traj);
    assert!((ratio - 1.0).abs() < 1e-6);
}

#[test]
fn test_coupling_minibt_ot_shape() {
    let coupling = RfCoupling::new(CouplingMode::MiniBatchOt);
    let n = 4;
    let d = 3;
    let x1 = vec![0.0f32; n * d];
    let x0 = coupling.sample_x0(&x1, n, d, 7).expect("sample_x0");
    assert_eq!(x0.len(), n * d);
}

#[test]
fn test_sample_t_zero_seed_still_works() {
    // seed=0 should be replaced by 1 internally
    let ts = rf_sample_t(0, 5, 0.0, 1.0);
    assert_eq!(ts.len(), 5);
    for t in &ts {
        assert!(*t >= 0.0 && *t <= 1.0);
    }
}

#[test]
fn test_solver_integrate_constant_velocity_reaches_x1() {
    // Constant velocity v = x1 - x0; Euler should reach x1 after n steps
    let config = RectifiedFlowConfig {
        n_steps: 50,
        solver: RfSolverKind::Euler,
        ..Default::default()
    };
    let solver = RfOdeSolver::new(config).unwrap();
    let sched = solver.generate_t_schedule();
    let x0 = vec![0.0f32, 0.0];
    let x1 = vec![1.0f32, 2.0];
    let x1_c = x1.clone();
    let traj = solver
        .integrate(&x0, &sched, move |_, _| {
            x1_c.iter()
                .zip([0.0, 0.0].iter())
                .map(|(&b, &a)| b - a)
                .collect()
        })
        .unwrap();
    let last = traj.states.last().unwrap();
    for (l, x) in last.iter().zip(x1.iter()) {
        assert!((l - x).abs() < 1e-4, "last={} x1={}", l, x);
    }
}

#[test]
fn test_format_stats_contains_mean_loss() {
    let stats = rf_compute_stats(&[0.1, 0.3], &[0.0], &[1.0]);
    let s = rf_format_stats(&stats);
    assert!(s.contains("loss"));
}

#[test]
fn test_format_config_contains_solver() {
    let config = RectifiedFlowConfig {
        solver: RfSolverKind::Rk4,
        ..Default::default()
    };
    let s = rf_format_config(&config);
    assert!(s.contains("Rk4"));
}
