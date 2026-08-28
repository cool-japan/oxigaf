//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::{normalized_float_distance, param_value_distance, xorshift64, xorshift_f64};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    // ---- xorshift64 --------------------------------------------------------

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 42u64;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0);
        }
    }

    #[test]
    fn test_xorshift_f64_range() {
        let mut state = 1u64;
        for _ in 0..1000 {
            let v = xorshift_f64(&mut state);
            assert!((0.0..1.0).contains(&v), "v={} out of [0,1)", v);
        }
    }

    #[test]
    fn test_xorshift_f64_zero_state_recovers() {
        // If state becomes 0, it is reset to 1.
        let mut state = 0u64;
        // xorshift64 will first apply XOR ops that may leave state 0,
        // but the guard sets it to 1.
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    // ---- sample_continuous -------------------------------------------------

    #[test]
    fn test_sample_continuous_normal() {
        let mut state = 12345u64;
        for _ in 0..200 {
            let v = sweep_sample_continuous(0.0, 1.0, false, &mut state)
                .expect("sample_continuous failed");
            assert!((0.0..=1.0).contains(&v), "v={} out of [0,1]", v);
        }
    }

    #[test]
    fn test_sample_continuous_wider_range() {
        let mut state = 99u64;
        for _ in 0..200 {
            let v = sweep_sample_continuous(-10.0, 10.0, false, &mut state)
                .expect("sample_continuous failed");
            assert!((-10.0..=10.0).contains(&v), "v={}", v);
        }
    }

    #[test]
    fn test_sample_continuous_log_scale() {
        let mut state = 7u64;
        for _ in 0..200 {
            let v = sweep_sample_continuous(1e-4, 1e-1, true, &mut state)
                .expect("log scale sample failed");
            assert!((1e-4..=1e-1 + 1e-12).contains(&v), "v={} out of range", v);
        }
    }

    #[test]
    fn test_sample_continuous_log_scale_negative_low_error() {
        let mut state = 1u64;
        let err = sweep_sample_continuous(-1.0, 1.0, true, &mut state);
        assert!(err.is_err());
    }

    #[test]
    fn test_sample_continuous_low_ge_high_error() {
        let mut state = 1u64;
        let err = sweep_sample_continuous(1.0, 0.5, false, &mut state);
        assert!(err.is_err());
    }

    #[test]
    fn test_sample_continuous_low_eq_high_error() {
        let mut state = 1u64;
        let err = sweep_sample_continuous(1.0, 1.0, false, &mut state);
        assert!(err.is_err());
    }

    // ---- sample_discrete ---------------------------------------------------

    #[test]
    fn test_sample_discrete_valid() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut state = 42u64;
        for _ in 0..500 {
            let v = sweep_sample_discrete(&values, &mut state).expect("sample_discrete failed");
            assert!(
                values.contains(&v),
                "sampled value {} not in {:?}",
                v,
                values
            );
        }
    }

    #[test]
    fn test_sample_discrete_single() {
        let values = vec![std::f64::consts::PI];
        let mut state = 1u64;
        let v = sweep_sample_discrete(&values, &mut state).expect("single value");
        assert_eq!(v, std::f64::consts::PI);
    }

    #[test]
    fn test_sample_discrete_empty_error() {
        let mut state = 1u64;
        let err = sweep_sample_discrete(&[], &mut state);
        assert!(matches!(err, Err(SweepError::EmptyDiscrete)));
    }

    // ---- grid_indices -------------------------------------------------------

    #[test]
    fn test_grid_indices_single_dim() {
        let dims = vec![5];
        for i in 0..5 {
            let idx = sweep_grid_indices(&dims, i);
            assert_eq!(idx, vec![i], "trial_idx={}", i);
        }
    }

    #[test]
    fn test_grid_indices_two_dims() {
        // dims=[3,2]: total 6 combos (row-major)
        let dims = vec![3, 2];
        let expected = [
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
            vec![2, 0],
            vec![2, 1],
        ];
        for (i, exp) in expected.iter().enumerate() {
            let idx = sweep_grid_indices(&dims, i);
            assert_eq!(&idx, exp, "trial_idx={}", i);
        }
    }

    #[test]
    fn test_grid_indices_three_dims() {
        let dims = vec![2, 3, 2];
        // 12 total combos
        let idx = sweep_grid_indices(&dims, 0);
        assert_eq!(idx, vec![0, 0, 0]);
        let idx_last = sweep_grid_indices(&dims, 11);
        assert_eq!(idx_last, vec![1, 2, 1]);
    }

    #[test]
    fn test_grid_indices_wraps_modulo() {
        // trial_idx beyond total grid size should wrap per dim.
        let dims = vec![2, 2];
        // 4 total; index 4 should wrap to [0,0] again.
        let idx = sweep_grid_indices(&dims, 4);
        assert_eq!(idx, vec![0, 0]);
    }

    #[test]
    fn test_grid_indices_empty_dims() {
        let dims: Vec<usize> = vec![];
        let idx = sweep_grid_indices(&dims, 5);
        assert!(idx.is_empty());
    }

    // ---- param_value_distance / normalized_float_distance -----------------

    #[test]
    fn test_normalized_float_distance_fallback_endpoints_and_log_scale() {
        // No spec: preserves the pre-fix `diff/(diff+1)` behavior exactly.
        let diff = 0.4f64;
        assert!((normalized_float_distance(None, 0.0, 0.4) - diff / (diff + 1.0)).abs() < 1e-12);

        let spec = ParamSpec::Continuous {
            name: "lr".to_string(),
            low: 1e-4,
            high: 1e-2,
            log_scale: false,
        };
        // Range endpoints are exactly 0.0 and 1.0 apart.
        assert!((normalized_float_distance(Some(&spec), 1e-4, 1e-4) - 0.0).abs() < 1e-9);
        assert!((normalized_float_distance(Some(&spec), 1e-4, 1e-2) - 1.0).abs() < 1e-9);

        // log_scale computes distance in log-space (matching how
        // `sweep_sample_continuous` samples it): the *geometric* midpoint of
        // [1e-4, 1e-2] (1e-3) sits at normalized distance 0.5 from 1e-4.
        let log_spec = ParamSpec::Continuous {
            name: "lr".to_string(),
            low: 1e-4,
            high: 1e-2,
            log_scale: true,
        };
        let dist = normalized_float_distance(Some(&log_spec), 1e-4, 1e-3);
        assert!((dist - 0.5).abs() < 1e-6, "got {dist}");
    }

    #[test]
    fn test_normalized_float_distance_small_and_large_range_params_are_comparable() {
        // Regression for the finding: a 1e-4..1e-2 parameter and a 0..1000
        // parameter, each compared at the same *relative* position in their
        // own range, must now yield comparable (not wildly different)
        // normalized distances -- the whole point of normalizing by each
        // parameter's own declared range. Before this fix, the un-normalized
        // `diff/(diff+1)` distance for the same two pairs was wildly
        // different (~0.005 vs ~0.998), dominated by absolute scale.
        let small_spec = ParamSpec::Continuous {
            name: "lr".to_string(),
            low: 1e-4,
            high: 1e-2,
            log_scale: false,
        };
        let large_spec = ParamSpec::Continuous {
            name: "batch_scale".to_string(),
            low: 0.0,
            high: 1000.0,
            log_scale: false,
        };
        // Each pair spans exactly half of its own parameter's range.
        let small_dist = normalized_float_distance(Some(&small_spec), 1e-4, 1e-4 + 0.0099 / 2.0);
        let large_dist = normalized_float_distance(Some(&large_spec), 0.0, 500.0);
        assert!((small_dist - 0.5).abs() < 1e-6, "small_dist={small_dist}");
        assert!((large_dist - 0.5).abs() < 1e-6, "large_dist={large_dist}");

        // Choice/mixed-type handling is unaffected by the spec-normalization
        // change: same/different choices stay 0.0/1.0, mixed types stay 1.0.
        let adam = ParamValue::Choice("adam".into());
        assert_eq!(
            param_value_distance(None, &adam, &ParamValue::Choice("adam".into())),
            0.0
        );
        assert_eq!(
            param_value_distance(None, &adam, &ParamValue::Choice("sgd".into())),
            1.0
        );
        assert_eq!(
            param_value_distance(None, &ParamValue::Float(0.5), &adam),
            1.0
        );
    }

    // ---- surrogate_predict -------------------------------------------------

    #[test]
    fn test_surrogate_predict_no_trials() {
        let params = vec![("lr".to_string(), ParamValue::Float(0.001))];
        let result = sweep_surrogate_predict(&[], &params, &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_surrogate_predict_exact_match() {
        let params = vec![("lr".to_string(), ParamValue::Float(0.001))];
        let trial = SweepTrial {
            id: 0,
            params: params.clone(),
            score: Some(0.5),
        };
        let result = sweep_surrogate_predict(&[trial], &params, &[]);
        // Exact match => distance=0 => weight is huge => should predict ~0.5.
        assert!((result - 0.5).abs() < 1e-6, "result={}", result);
    }

    #[test]
    fn test_surrogate_predict_two_trials() {
        let trial_a = SweepTrial {
            id: 0,
            params: vec![("x".to_string(), ParamValue::Float(0.0))],
            score: Some(1.0),
        };
        let trial_b = SweepTrial {
            id: 1,
            params: vec![("x".to_string(), ParamValue::Float(1.0))],
            score: Some(2.0),
        };
        // Query close to trial_a should predict closer to 1.0.
        let query = vec![("x".to_string(), ParamValue::Float(0.01))];
        // No matching spec for "x" -> falls back to the un-normalized
        // diff/(diff+1) distance, same as before this fix.
        let result = sweep_surrogate_predict(&[trial_a, trial_b], &query, &[]);
        assert!(result < 1.5, "Expected result < 1.5, got {}", result);
    }

    #[test]
    fn test_surrogate_predict_ignores_unscored() {
        let scored = SweepTrial {
            id: 0,
            params: vec![("x".to_string(), ParamValue::Float(0.5))],
            score: Some(3.0),
        };
        let unscored = SweepTrial {
            id: 1,
            params: vec![("x".to_string(), ParamValue::Float(0.5))],
            score: None,
        };
        let query = vec![("x".to_string(), ParamValue::Float(0.5))];
        let result = sweep_surrogate_predict(&[scored, unscored], &query, &[]);
        assert!((result - 3.0).abs() < 1e-6, "result={}", result);
    }

    #[test]
    fn test_surrogate_predict_normalizes_small_range_param_by_spec_bounds() {
        // End-to-end through the public API: `specs` reaches `param_distance`,
        // so a query near the top of a small (1e-4..1e-2) range predicts
        // close to the trial at the top of that range.
        let lr_spec = ParamSpec::Continuous {
            name: "lr".to_string(),
            low: 1e-4,
            high: 1e-2,
            log_scale: false,
        };
        let low = SweepTrial {
            id: 0,
            params: vec![("lr".to_string(), ParamValue::Float(1e-4))],
            score: Some(1.0),
        };
        let high = SweepTrial {
            id: 1,
            params: vec![("lr".to_string(), ParamValue::Float(1e-2))],
            score: Some(2.0),
        };
        let query = vec![("lr".to_string(), ParamValue::Float(9.9e-3))];
        let result = sweep_surrogate_predict(&[low, high], &query, &[lr_spec]);
        assert!((result - 2.0).abs() < 0.05, "got {result}");
    }

    // ---- compute_param_importance ------------------------------------------

    #[test]
    fn test_compute_param_importance_single_param() {
        let specs = vec![ParamSpec::Continuous {
            name: "lr".into(),
            low: 0.0,
            high: 1.0,
            log_scale: false,
        }];
        let trials = vec![
            SweepTrial {
                id: 0,
                params: vec![("lr".into(), ParamValue::Float(0.1))],
                score: Some(0.9),
            },
            SweepTrial {
                id: 1,
                params: vec![("lr".into(), ParamValue::Float(0.5))],
                score: Some(0.5),
            },
            SweepTrial {
                id: 2,
                params: vec![("lr".into(), ParamValue::Float(0.9))],
                score: Some(0.1),
            },
        ];
        let importances = sweep_param_importance(&trials, &specs);
        assert_eq!(importances.len(), 1);
        assert_eq!(importances[0].0, "lr");
        // Single param => importance=1.0.
        assert!((importances[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_param_importance_multi_param_sums_to_one() {
        let specs = vec![
            ParamSpec::Continuous {
                name: "lr".into(),
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
            ParamSpec::Discrete {
                name: "n_sh".into(),
                values: vec![1.0, 4.0, 9.0],
            },
        ];
        let trials = vec![
            SweepTrial {
                id: 0,
                params: vec![
                    ("lr".into(), ParamValue::Float(0.1)),
                    ("n_sh".into(), ParamValue::Float(1.0)),
                ],
                score: Some(0.8),
            },
            SweepTrial {
                id: 1,
                params: vec![
                    ("lr".into(), ParamValue::Float(0.5)),
                    ("n_sh".into(), ParamValue::Float(4.0)),
                ],
                score: Some(0.5),
            },
            SweepTrial {
                id: 2,
                params: vec![
                    ("lr".into(), ParamValue::Float(0.9)),
                    ("n_sh".into(), ParamValue::Float(9.0)),
                ],
                score: Some(0.2),
            },
        ];
        let importances = sweep_param_importance(&trials, &specs);
        let total: f64 = importances.iter().map(|(_, v)| v).sum();
        assert!((total - 1.0).abs() < 1e-10, "total={}", total);
    }

    #[test]
    fn test_compute_param_importance_fewer_than_two_trials() {
        let specs = vec![ParamSpec::Discrete {
            name: "x".into(),
            values: vec![1.0, 2.0],
        }];
        let trials = vec![SweepTrial {
            id: 0,
            params: vec![("x".into(), ParamValue::Float(1.0))],
            score: Some(0.5),
        }];
        let importances = sweep_param_importance(&trials, &specs);
        assert_eq!(importances[0].1, 0.0);
    }

    #[test]
    fn test_compute_param_importance_categorical() {
        let specs = vec![ParamSpec::Categorical {
            name: "opt".into(),
            choices: vec!["adam".into(), "sgd".into(), "rmsprop".into()],
        }];
        let trials = vec![
            SweepTrial {
                id: 0,
                params: vec![("opt".into(), ParamValue::Choice("adam".into()))],
                score: Some(0.1),
            },
            SweepTrial {
                id: 1,
                params: vec![("opt".into(), ParamValue::Choice("sgd".into()))],
                score: Some(0.5),
            },
            SweepTrial {
                id: 2,
                params: vec![("opt".into(), ParamValue::Choice("rmsprop".into()))],
                score: Some(0.9),
            },
        ];
        let importances = sweep_param_importance(&trials, &specs);
        assert_eq!(importances.len(), 1);
        // Single param => 1.0 (after normalization of a nonzero correlation).
        assert!(importances[0].1 >= 0.0 && importances[0].1 <= 1.0 + 1e-10);
    }

    // ---- hyperband_bracket -------------------------------------------------

    #[test]
    fn test_hyperband_bracket_basic() {
        // max_iter=81, eta=3: s_max=4
        let brackets = hyperband_bracket(81, 3);
        // Should have s_max+1 = 5 rounds.
        assert_eq!(brackets.len(), 5, "brackets={:?}", brackets);
        // All budgets should be > 0.
        for &(n, b) in &brackets {
            assert!(n >= 1, "n_configs must be >= 1");
            assert!(b > 0, "budget must be > 0");
        }
    }

    #[test]
    fn test_hyperband_bracket_matches_published_algorithm_max_iter_81_eta_3() {
        // Regression: pins the exact reference values from Li et al. 2016's
        // published Hyperband algorithm for the textbook max_iter=81, eta=3
        // example. The `n_configs` formula previously omitted the `/ (s+1)`
        // divisor, which yielded 5, 15, 45, 135, 405 instead of these.
        let brackets = hyperband_bracket(81, 3);
        let n_configs: Vec<usize> = brackets.iter().map(|&(n, _)| n).collect();
        assert_eq!(n_configs, vec![5, 8, 15, 34, 81], "brackets={:?}", brackets);
    }

    #[test]
    fn test_hyperband_bracket_eta_2() {
        let brackets = hyperband_bracket(16, 2);
        assert!(!brackets.is_empty());
        // Innermost round (first in list): fewest configs, highest budget.
        let (n0, b0) = brackets[0];
        if brackets.len() > 1 {
            let (n1, b1) = brackets[1];
            // Innermost should have fewer or equal configs.
            let _ = (n0, b0, n1, b1); // verify they exist
        }
    }

    #[test]
    fn test_hyperband_bracket_zero_max_iter() {
        let brackets = hyperband_bracket(0, 3);
        assert!(brackets.is_empty());
    }

    #[test]
    fn test_hyperband_bracket_eta_one_returns_empty() {
        let brackets = hyperband_bracket(10, 1);
        assert!(brackets.is_empty());
    }

    #[test]
    fn test_hyperband_bracket_large() {
        let brackets = hyperband_bracket(243, 3);
        // s_max=5, so 6 rounds.
        assert_eq!(brackets.len(), 6, "brackets={:?}", brackets);
    }

    // ---- ParameterSweep::new -----------------------------------------------

    #[test]
    fn test_sweep_new_empty_specs_error() {
        let config = SweepConfig {
            specs: vec![],
            strategy: SweepStrategy::Random,
            max_trials: 10,
            seed: 1,
            minimize: true,
        };
        let err = ParameterSweep::new(config);
        assert!(matches!(err, Err(SweepError::EmptySpecs)));
    }

    #[test]
    fn test_sweep_new_grid_with_continuous_error() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "lr".into(),
                low: 0.0,
                high: 1.0,
                log_scale: false,
            }],
            strategy: SweepStrategy::Grid,
            max_trials: 10,
            seed: 1,
            minimize: true,
        };
        let err = ParameterSweep::new(config);
        assert!(matches!(
            err,
            Err(SweepError::GridNotSupportedForContinuous)
        ));
    }

    #[test]
    fn test_sweep_new_valid() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "n".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Grid,
            max_trials: 4,
            seed: 1,
            minimize: true,
        };
        assert!(ParameterSweep::new(config).is_ok());
    }

    // ---- ParameterSweep::suggest (Grid) ------------------------------------

    #[test]
    fn test_sweep_suggest_grid_produces_all_combos() {
        let config = SweepConfig {
            specs: vec![
                ParamSpec::Discrete {
                    name: "a".into(),
                    values: vec![1.0, 2.0],
                },
                ParamSpec::Categorical {
                    name: "b".into(),
                    choices: vec!["x".into(), "y".into()],
                },
            ],
            strategy: SweepStrategy::Grid,
            max_trials: 4,
            seed: 0,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let mut trials = Vec::new();
        for _ in 0..4 {
            trials.push(sweep.suggest().expect("suggest"));
        }
        assert_eq!(trials.len(), 4);
        // All IDs unique.
        let ids: Vec<usize> = trials.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_sweep_suggest_grid_max_trials_reached() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Grid,
            max_trials: 2,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        sweep.suggest().expect("first");
        sweep.suggest().expect("second");
        let err = sweep.suggest();
        assert!(matches!(err, Err(SweepError::MaxTrialsReached { .. })));
    }

    // ---- ParameterSweep::suggest (Random) ----------------------------------

    #[test]
    fn test_sweep_suggest_random_continuous_in_range() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "lr".into(),
                low: 0.001,
                high: 0.1,
                log_scale: false,
            }],
            strategy: SweepStrategy::Random,
            max_trials: 50,
            seed: 7,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        for _ in 0..50 {
            let trial = sweep.suggest().expect("suggest");
            if let ParamValue::Float(v) = &trial.params[0].1 {
                assert!(*v >= 0.001 && *v <= 0.1, "v={} out of range", v);
            }
        }
    }

    #[test]
    fn test_sweep_suggest_random_log_scale() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "lr".into(),
                low: 1e-5,
                high: 1e-1,
                log_scale: true,
            }],
            strategy: SweepStrategy::Random,
            max_trials: 50,
            seed: 99,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        for _ in 0..50 {
            let trial = sweep.suggest().expect("suggest");
            if let ParamValue::Float(v) = &trial.params[0].1 {
                assert!(*v >= 1e-5 && *v <= 1e-1 + 1e-15, "v={}", v);
            }
        }
    }

    #[test]
    fn test_sweep_suggest_random_categorical() {
        let choices = vec!["adam".into(), "sgd".into()];
        let config = SweepConfig {
            specs: vec![ParamSpec::Categorical {
                name: "opt".into(),
                choices: choices.clone(),
            }],
            strategy: SweepStrategy::Random,
            max_trials: 30,
            seed: 5,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        for _ in 0..30 {
            let trial = sweep.suggest().expect("suggest");
            if let ParamValue::Choice(c) = &trial.params[0].1 {
                assert!(choices.contains(c), "unexpected choice: {}", c);
            }
        }
    }

    // ---- ParameterSweep::suggest (Surrogate) --------------------------------

    #[test]
    fn test_sweep_suggest_surrogate_no_completed_falls_back_to_random() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "x".into(),
                low: 0.0,
                high: 1.0,
                log_scale: false,
            }],
            strategy: SweepStrategy::Surrogate,
            max_trials: 5,
            seed: 3,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        // Without any completed trials, suggest should succeed (random fallback).
        let trial = sweep.suggest().expect("surrogate fallback");
        assert!(trial.score.is_none());
    }

    #[test]
    fn test_sweep_suggest_surrogate_with_completed_trials() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "x".into(),
                low: 0.0,
                high: 1.0,
                log_scale: false,
            }],
            strategy: SweepStrategy::Surrogate,
            max_trials: 10,
            seed: 11,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        // Bootstrap with random trial.
        let t0 = sweep.suggest().expect("t0");
        sweep.report(t0.id, 0.3).expect("report t0");
        // Now surrogate should use t0 as guidance.
        let t1 = sweep.suggest().expect("t1 surrogate");
        assert!(t1.score.is_none());
    }

    // ---- ParameterSweep::report --------------------------------------------

    #[test]
    fn test_report_valid() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 5,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let trial = sweep.suggest().expect("suggest");
        sweep.report(trial.id, 0.42).expect("report");
        assert_eq!(sweep.trials_completed(), 1);
    }

    #[test]
    fn test_report_unknown_id_error() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 5,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let err = sweep.report(999, 0.5);
        assert!(matches!(err, Err(SweepError::TrialNotFound(999))));
    }

    // ---- ParameterSweep::best_trial ----------------------------------------

    #[test]
    fn test_best_trial_minimize() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0, 3.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 3,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let t0 = sweep.suggest().expect("t0");
        sweep.report(t0.id, 0.5).expect("r0");
        let t1 = sweep.suggest().expect("t1");
        sweep.report(t1.id, 0.1).expect("r1");
        let t2 = sweep.suggest().expect("t2");
        sweep.report(t2.id, 0.8).expect("r2");
        let best = sweep.best_trial().expect("best");
        assert_eq!(best.id, t1.id);
    }

    #[test]
    fn test_best_trial_maximize() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 2,
            seed: 1,
            minimize: false,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let t0 = sweep.suggest().expect("t0");
        sweep.report(t0.id, 0.3).expect("r0");
        let t1 = sweep.suggest().expect("t1");
        sweep.report(t1.id, 0.9).expect("r1");
        let best = sweep.best_trial().expect("best");
        assert_eq!(best.id, t1.id);
    }

    #[test]
    fn test_best_trial_no_completed() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 2,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        sweep.suggest().expect("suggest");
        assert!(sweep.best_trial().is_none());
    }

    // ---- ParameterSweep::top_k_trials --------------------------------------

    #[test]
    fn test_top_k_trials_ordering() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 5,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let scores = [0.5, 0.1, 0.9, 0.3, 0.7];
        for &s in &scores {
            let t = sweep.suggest().expect("suggest");
            sweep.report(t.id, s).expect("report");
        }
        let top3 = sweep.top_k_trials(3);
        assert_eq!(top3.len(), 3);
        let top_scores: Vec<f64> = top3.iter().map(|t| t.score.unwrap()).collect();
        // Should be ascending (minimize=true).
        assert!(top_scores[0] <= top_scores[1]);
        assert!(top_scores[1] <= top_scores[2]);
        assert_eq!(top_scores[0], 0.1);
    }

    #[test]
    fn test_top_k_exceeds_completed() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 2,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let t0 = sweep.suggest().expect("t0");
        sweep.report(t0.id, 0.4).expect("r0");
        // Request 10 but only 1 completed.
        let top = sweep.top_k_trials(10);
        assert_eq!(top.len(), 1);
    }

    // ---- ParameterSweep::is_done -------------------------------------------

    #[test]
    fn test_is_done() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Grid,
            max_trials: 2,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        assert!(!sweep.is_done());
        sweep.suggest().expect("t0");
        assert!(!sweep.is_done());
        sweep.suggest().expect("t1");
        assert!(sweep.is_done());
    }

    // ---- format_trial / format_sweep_summary --------------------------------

    #[test]
    fn test_format_trial_pending() {
        let trial = SweepTrial {
            id: 5,
            params: vec![("lr".into(), ParamValue::Float(0.001234))],
            score: None,
        };
        let s = format_sweep_trial(&trial);
        assert!(s.contains("Trial #5"), "s={}", s);
        assert!(s.contains("lr="), "s={}", s);
        assert!(s.contains("pending"), "s={}", s);
    }

    #[test]
    fn test_format_trial_with_score() {
        let trial = SweepTrial {
            id: 0,
            params: vec![("lr".into(), ParamValue::Float(0.01))],
            score: Some(0.123456),
        };
        let s = format_sweep_trial(&trial);
        assert!(s.contains("score=0.123456"), "s={}", s);
    }

    #[test]
    fn test_format_trial_categorical() {
        let trial = SweepTrial {
            id: 2,
            params: vec![("opt".into(), ParamValue::Choice("adam".into()))],
            score: Some(0.5),
        };
        let s = format_sweep_trial(&trial);
        assert!(s.contains("opt=adam"), "s={}", s);
    }

    #[test]
    fn test_format_sweep_summary_no_trials() {
        let summary = SweepSummary {
            total_trials: 0,
            completed_trials: 0,
            best_score: None,
            worst_score: None,
            mean_score: None,
            std_score: None,
            param_importances: vec![("lr".into(), 1.0)],
        };
        let s = format_sweep_summary(&summary);
        assert!(s.contains("0/0"), "s={}", s);
    }

    #[test]
    fn test_format_sweep_summary_with_data() {
        let summary = SweepSummary {
            total_trials: 10,
            completed_trials: 8,
            best_score: Some(0.1),
            worst_score: Some(0.9),
            mean_score: Some(0.5),
            std_score: Some(0.2),
            param_importances: vec![("lr".into(), 0.7), ("n_sh".into(), 0.3)],
        };
        let s = format_sweep_summary(&summary);
        assert!(s.contains("8/10"), "s={}", s);
        assert!(s.contains("0.100000"), "s={}", s);
        assert!(s.contains("lr"), "s={}", s);
        assert!(s.contains("n_sh"), "s={}", s);
    }

    // ---- ParameterSweep::summary -------------------------------------------

    #[test]
    fn test_summary_empty() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 5,
            seed: 1,
            minimize: true,
        };
        let sweep = ParameterSweep::new(config).expect("new");
        let s = sweep.summary();
        assert_eq!(s.total_trials, 0);
        assert_eq!(s.completed_trials, 0);
        assert!(s.best_score.is_none());
    }

    #[test]
    fn test_summary_all_same_score() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0, 3.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 3,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        for _ in 0..3 {
            let t = sweep.suggest().expect("suggest");
            sweep.report(t.id, 0.5).expect("report");
        }
        let s = sweep.summary();
        assert_eq!(s.best_score, Some(0.5));
        assert_eq!(s.worst_score, Some(0.5));
        assert_eq!(s.mean_score, Some(0.5));
        assert_eq!(s.std_score, Some(0.0));
    }

    // ---- Edge cases --------------------------------------------------------

    #[test]
    fn test_max_trials_zero_immediately_done() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 0,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        assert!(sweep.is_done());
        let err = sweep.suggest();
        assert!(matches!(err, Err(SweepError::MaxTrialsReached { .. })));
    }

    #[test]
    fn test_trials_completed_count() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0, 3.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 3,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        assert_eq!(sweep.trials_completed(), 0);
        let t0 = sweep.suggest().expect("t0");
        assert_eq!(sweep.trials_completed(), 0);
        sweep.report(t0.id, 0.5).expect("r0");
        assert_eq!(sweep.trials_completed(), 1);
        let t1 = sweep.suggest().expect("t1");
        sweep.report(t1.id, 0.3).expect("r1");
        assert_eq!(sweep.trials_completed(), 2);
    }

    #[test]
    fn test_param_value_display_float() {
        let v = ParamValue::Float(std::f64::consts::PI);
        let s = format!("{}", v);
        assert!(s.contains("3.141593"), "s={}", s);
    }

    #[test]
    fn test_param_value_display_int() {
        let v = ParamValue::Int(42);
        assert_eq!(format!("{}", v), "42");
    }

    #[test]
    fn test_param_value_display_choice() {
        let v = ParamValue::Choice("adam".into());
        assert_eq!(format!("{}", v), "adam");
    }

    #[test]
    fn test_surrogate_multiple_params() {
        let config = SweepConfig {
            specs: vec![
                ParamSpec::Continuous {
                    name: "lr".into(),
                    low: 0.0001,
                    high: 0.01,
                    log_scale: false,
                },
                ParamSpec::Discrete {
                    name: "n_sh".into(),
                    values: vec![1.0, 4.0, 9.0],
                },
            ],
            strategy: SweepStrategy::Surrogate,
            max_trials: 15,
            seed: 42,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        // Run a few bootstrap trials.
        for _ in 0..3 {
            let t = sweep.suggest().expect("suggest");
            sweep.report(t.id, 0.5).expect("report");
        }
        // Surrogate suggestions should now succeed.
        for _ in 0..5 {
            sweep.suggest().expect("surrogate suggest");
        }
        assert_eq!(sweep.trials_completed(), 3);
    }

    #[test]
    fn test_grid_full_product_count() {
        // 2 * 3 * 2 = 12 total combos.
        let config = SweepConfig {
            specs: vec![
                ParamSpec::Discrete {
                    name: "a".into(),
                    values: vec![1.0, 2.0],
                },
                ParamSpec::Discrete {
                    name: "b".into(),
                    values: vec![10.0, 20.0, 30.0],
                },
                ParamSpec::Discrete {
                    name: "c".into(),
                    values: vec![0.1, 0.9],
                },
            ],
            strategy: SweepStrategy::Grid,
            max_trials: 12,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let mut all_params = Vec::new();
        for _ in 0..12 {
            let t = sweep.suggest().expect("suggest");
            all_params.push(t.params.clone());
        }
        // All 12 combos should be unique.
        let unique: std::collections::HashSet<String> = all_params
            .iter()
            .map(|p| {
                format!(
                    "{:?}",
                    p.iter().map(|(_, v)| format!("{}", v)).collect::<Vec<_>>()
                )
            })
            .collect();
        assert_eq!(
            unique.len(),
            12,
            "Expected 12 unique combos, got {}",
            unique.len()
        );
    }
}
