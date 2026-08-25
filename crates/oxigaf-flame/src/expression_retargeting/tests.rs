//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::retar_solve_ridge;
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
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
        xorshift64(state) as f32 / u64::MAX as f32
    }
    fn random_state(expr_dim: usize, rng: &mut u64) -> ExpressionState {
        let params: Vec<f32> = (0..expr_dim)
            .map(|_| xorshift_f32(rng) * 2.0 - 1.0)
            .collect();
        ExpressionState::from_params(params)
    }
    #[test]
    fn test_neutral_all_zeros() {
        let s = ExpressionState::neutral(8);
        assert_eq!(s.expr_dim, 8);
        assert!(s.expression_params.iter().all(|&v| v == 0.0));
        assert_eq!(s.jaw_pose, [0.0, 0.0, 0.0]);
    }
    #[test]
    fn test_neutral_zero_dim() {
        let s = ExpressionState::neutral(0);
        assert_eq!(s.expr_dim, 0);
        assert!(s.expression_params.is_empty());
    }
    #[test]
    fn test_from_params_sets_dim() {
        let s = ExpressionState::from_params(vec![1.0, 2.0, 3.0]);
        assert_eq!(s.expr_dim, 3);
        assert_eq!(s.jaw_pose, [0.0, 0.0, 0.0]);
    }
    #[test]
    fn test_from_params_and_jaw() {
        let s = ExpressionState::from_params_and_jaw(vec![1.0; 4], [0.1, 0.2, 0.3]);
        assert_eq!(s.jaw_pose, [0.1, 0.2, 0.3]);
        assert_eq!(s.expr_dim, 4);
    }
    #[test]
    fn test_with_scale_doubles() {
        let s = ExpressionState::from_params_and_jaw(vec![1.0, -0.5, 0.25], [0.1, 0.0, 0.0]);
        let s2 = s.with_scale(2.0);
        assert!((s2.expression_params[0] - 2.0).abs() < 1e-6);
        assert!((s2.expression_params[1] + 1.0).abs() < 1e-6);
        assert!((s2.expression_params[2] - 0.5).abs() < 1e-6);
        assert!((s2.jaw_pose[0] - 0.2).abs() < 1e-6);
    }
    #[test]
    fn test_with_scale_zero() {
        let s = ExpressionState::from_params(vec![1.0, 2.0]);
        let s0 = s.with_scale(0.0);
        assert!(s0.expression_params.iter().all(|&v| v == 0.0));
    }
    #[test]
    fn test_blend_t0_returns_a() {
        let a = ExpressionState::from_params(vec![1.0, 0.0]);
        let b = ExpressionState::from_params(vec![0.0, 1.0]);
        let r = a.blend(&b, 0.0).unwrap();
        assert!((r.expression_params[0] - 1.0).abs() < 1e-6);
        assert!((r.expression_params[1]).abs() < 1e-6);
    }
    #[test]
    fn test_blend_t1_returns_b() {
        let a = ExpressionState::from_params(vec![1.0, 0.0]);
        let b = ExpressionState::from_params(vec![0.0, 1.0]);
        let r = a.blend(&b, 1.0).unwrap();
        assert!(r.expression_params[0].abs() < 1e-6);
        assert!((r.expression_params[1] - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_blend_t_half() {
        let a = ExpressionState::from_params(vec![0.0, 0.0]);
        let b = ExpressionState::from_params(vec![2.0, 2.0]);
        let r = a.blend(&b, 0.5).unwrap();
        assert!((r.expression_params[0] - 1.0).abs() < 1e-5);
        assert!((r.expression_params[1] - 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_blend_mismatched_dims_error() {
        let a = ExpressionState::from_params(vec![1.0, 0.0]);
        let b = ExpressionState::from_params(vec![0.0, 1.0, 0.5]);
        assert!(matches!(
            a.blend(&b, 0.5),
            Err(RetargetError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_compute_variance_identical_states() {
        let s = ExpressionState::from_params(vec![1.0, 2.0, 3.0]);
        let states = vec![s.clone(), s.clone(), s.clone()];
        let stats = retar_compute_variance(&states).unwrap();
        assert!(stats.per_dim_variance.iter().all(|&v| v < 1e-6));
        assert!(stats.total_variance < 1e-6);
    }
    #[test]
    fn test_compute_variance_single_state() {
        let states = vec![ExpressionState::from_params(vec![1.0, -2.0])];
        let stats = retar_compute_variance(&states).unwrap();
        assert!(stats.per_dim_variance.iter().all(|&v| v < 1e-6));
    }
    #[test]
    fn test_compute_variance_empty_error() {
        let result = retar_compute_variance(&[]);
        assert!(matches!(result, Err(RetargetError::EmptySequence)));
    }
    #[test]
    fn test_compute_variance_top_k_ordered() {
        let states = vec![
            ExpressionState::from_params(vec![0.0, 10.0]),
            ExpressionState::from_params(vec![0.0, -10.0]),
        ];
        let stats = retar_compute_variance(&states).unwrap();
        assert_eq!(stats.top_k_dims[0], 1);
    }
    #[test]
    fn test_standardize_roundtrip() {
        let mut rng = 0xDEAD_BEEF_u64;
        let states: Vec<ExpressionState> = (0..10).map(|_| random_state(8, &mut rng)).collect();
        let stats = retar_compute_variance(&states).unwrap();
        for s in &states {
            let std_v = retar_standardize(s, &stats).unwrap();
            let restored = retar_unstandardize(&std_v, &stats, s.expr_dim).unwrap();
            for (a, b) in s
                .expression_params
                .iter()
                .zip(restored.expression_params.iter())
            {
                assert!((a - b).abs() < 1e-5, "roundtrip failed: {a} vs {b}");
            }
        }
    }
    #[test]
    fn test_standardize_dim_mismatch() {
        let s = ExpressionState::from_params(vec![1.0, 2.0, 3.0]);
        let stats =
            retar_compute_variance(&[ExpressionState::from_params(vec![0.0, 0.0])]).unwrap();
        assert!(matches!(
            retar_standardize(&s, &stats),
            Err(RetargetError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_unstandardize_dim_mismatch() {
        let stats =
            retar_compute_variance(&[ExpressionState::from_params(vec![0.0, 0.0])]).unwrap();
        let result = retar_unstandardize(&[1.0, 2.0, 3.0], &stats, 3);
        assert!(matches!(
            result,
            Err(RetargetError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_solve_ridge_1d_simple() {
        let a = vec![1.0_f32];
        let b = vec![3.0_f32];
        let x = retar_solve_ridge(&a, &b, 1, 1, 1e-7).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-3, "expected ~3, got {}", x[0]);
    }
    #[test]
    fn test_solve_ridge_2d() {
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b = vec![2.0_f32, 5.0];
        let x = retar_solve_ridge(&a, &b, 2, 2, 1e-7).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-3, "x[0] = {}", x[0]);
        assert!((x[1] - 5.0).abs() < 1e-3, "x[1] = {}", x[1]);
    }
    #[test]
    fn test_solve_ridge_regularization_shrinks() {
        let a = vec![1.0_f32];
        let b = vec![3.0_f32];
        let x_small = retar_solve_ridge(&a, &b, 1, 1, 1e-7).unwrap();
        let x_large = retar_solve_ridge(&a, &b, 1, 1, 1e6).unwrap();
        assert!(x_large[0].abs() < x_small[0].abs());
    }
    #[test]
    fn test_identity_retargeter_passthrough() {
        let config = RetargetConfig {
            expr_dim: 4,
            include_jaw: false,
            ..Default::default()
        };
        let r = LinearExpressionRetargeter::identity(config);
        let s = ExpressionState::from_params(vec![0.5, -0.3, 0.1, 0.0]);
        let out = r.retarget(&s).unwrap();
        for (a, b) in s.expression_params.iter().zip(out.expression_params.iter()) {
            assert!((a - b).abs() < 1e-5, "identity failed: {a} vs {b}");
        }
    }
    #[test]
    fn test_identity_retargeter_neutral_to_neutral() {
        let config = RetargetConfig {
            expr_dim: 6,
            ..Default::default()
        };
        let r = LinearExpressionRetargeter::identity(config);
        let s = ExpressionState::neutral(6);
        let out = r.retarget(&s).unwrap();
        assert!(out.expression_params.iter().all(|&v| v.abs() < 1e-5));
    }
    #[test]
    fn test_identity_n_training_pairs_zero() {
        let config = RetargetConfig::default();
        let r = LinearExpressionRetargeter::identity(config);
        assert_eq!(r.n_training_pairs(), 0);
    }
    #[test]
    fn test_fit_requires_at_least_2_pairs() {
        let config = RetargetConfig {
            expr_dim: 4,
            ..Default::default()
        };
        let pair = RetargetPair {
            source: ExpressionState::neutral(4),
            target: ExpressionState::neutral(4),
        };
        let result = LinearExpressionRetargeter::fit(&[pair], config);
        assert!(matches!(result, Err(RetargetError::NotEnoughPairs { .. })));
    }
    #[test]
    fn test_fit_empty_pairs() {
        let config = RetargetConfig {
            expr_dim: 4,
            ..Default::default()
        };
        let result = LinearExpressionRetargeter::fit(&[], config);
        assert!(matches!(result, Err(RetargetError::NotEnoughPairs { .. })));
    }
    #[test]
    fn test_fit_identity_mapping() {
        let dim = 4;
        let mut rng = 0xABCD_1234_u64;
        let config = RetargetConfig {
            expr_dim: dim,
            regularization: 1e-6,
            scale_by_variance: false,
            include_jaw: false,
            ..Default::default()
        };
        let pairs: Vec<RetargetPair> = (0..8)
            .map(|_| {
                let s = random_state(dim, &mut rng);
                RetargetPair {
                    source: s.clone(),
                    target: s,
                }
            })
            .collect();
        let r = LinearExpressionRetargeter::fit(&pairs, config).unwrap();
        let test = random_state(dim, &mut rng);
        let out = r.retarget(&test).unwrap();
        for (a, b) in test
            .expression_params
            .iter()
            .zip(out.expression_params.iter())
        {
            assert!((a - b).abs() < 0.05, "identity mapping failed: {a} vs {b}");
        }
    }
    #[test]
    fn test_fit_minimum_2_pairs() {
        let config = RetargetConfig {
            expr_dim: 2,
            include_jaw: false,
            ..Default::default()
        };
        let pairs = vec![
            RetargetPair {
                source: ExpressionState::neutral(2),
                target: ExpressionState::neutral(2),
            },
            RetargetPair {
                source: ExpressionState::from_params(vec![1.0, 0.0]),
                target: ExpressionState::from_params(vec![1.0, 0.0]),
            },
        ];
        let r = LinearExpressionRetargeter::fit(&pairs, config);
        assert!(r.is_ok(), "fit with 2 pairs should succeed: {:?}", r.err());
    }
    #[test]
    fn test_fit_n_training_pairs() {
        let config = RetargetConfig {
            expr_dim: 3,
            include_jaw: false,
            ..Default::default()
        };
        let mut rng = 0x1234_5678_u64;
        let pairs: Vec<RetargetPair> = (0..5)
            .map(|_| RetargetPair {
                source: random_state(3, &mut rng),
                target: random_state(3, &mut rng),
            })
            .collect();
        let r = LinearExpressionRetargeter::fit(&pairs, config).unwrap();
        assert_eq!(r.n_training_pairs(), 5);
    }
    #[test]
    fn test_fit_source_dim_accessors() {
        let config = RetargetConfig {
            expr_dim: 6,
            include_jaw: true,
            ..Default::default()
        };
        let r = LinearExpressionRetargeter::identity(config);
        assert_eq!(r.source_dim(), 9);
        assert_eq!(r.target_dim(), 9);
    }
    #[test]
    fn test_retarget_dim_mismatch() {
        let config = RetargetConfig {
            expr_dim: 4,
            ..Default::default()
        };
        let r = LinearExpressionRetargeter::identity(config);
        let wrong = ExpressionState::from_params(vec![1.0, 2.0]);
        assert!(matches!(
            r.retarget(&wrong),
            Err(RetargetError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_retarget_sequence_length_preserved() {
        let config = RetargetConfig {
            expr_dim: 4,
            include_jaw: false,
            ..Default::default()
        };
        let r = LinearExpressionRetargeter::identity(config);
        let seq: Vec<ExpressionState> = (0..7).map(|_| ExpressionState::neutral(4)).collect();
        let out = r.retarget_sequence(&seq).unwrap();
        assert_eq!(out.len(), 7);
    }
    #[test]
    fn test_retarget_sequence_empty_error() {
        let config = RetargetConfig {
            expr_dim: 4,
            ..Default::default()
        };
        let r = LinearExpressionRetargeter::identity(config);
        assert!(matches!(
            r.retarget_sequence(&[]),
            Err(RetargetError::EmptySequence)
        ));
    }
    #[test]
    fn test_velocity_length() {
        let seq: Vec<ExpressionState> = (0..5).map(|_| ExpressionState::neutral(4)).collect();
        let vel = retar_expression_velocity(&seq).unwrap();
        assert_eq!(vel.len(), 4);
    }
    #[test]
    fn test_velocity_constant_sequence_near_zero() {
        let s = ExpressionState::from_params(vec![1.0, -2.0, 0.5]);
        let seq = vec![s.clone(), s.clone(), s.clone(), s];
        let vel = retar_expression_velocity(&seq).unwrap();
        for v in &vel {
            assert!(v.expression_params.iter().all(|&x| x.abs() < 1e-6));
        }
    }
    #[test]
    fn test_velocity_single_frame_error() {
        let seq = vec![ExpressionState::neutral(4)];
        assert!(matches!(
            retar_expression_velocity(&seq),
            Err(RetargetError::EmptySequence)
        ));
    }
    #[test]
    fn test_acceleration_length() {
        let seq: Vec<ExpressionState> = (0..6).map(|_| ExpressionState::neutral(4)).collect();
        let acc = retar_expression_acceleration(&seq).unwrap();
        assert_eq!(acc.len(), 4);
    }
    #[test]
    fn test_acceleration_linear_sequence_near_zero() {
        let seq: Vec<ExpressionState> = (0..5)
            .map(|i| ExpressionState::from_params(vec![i as f32, 0.0]))
            .collect();
        let acc = retar_expression_acceleration(&seq).unwrap();
        for a in &acc {
            assert!(a.expression_params.iter().all(|&v| v.abs() < 1e-5));
        }
    }
    #[test]
    fn test_acceleration_too_short_error() {
        let seq = vec![ExpressionState::neutral(4), ExpressionState::neutral(4)];
        assert!(matches!(
            retar_expression_acceleration(&seq),
            Err(RetargetError::EmptySequence)
        ));
    }
    #[test]
    fn test_smooth_constant_sequence_unchanged() {
        let val = vec![0.3_f32, -0.1, 0.5, 0.0];
        let s = ExpressionState::from_params(val.clone());
        let seq = vec![s.clone(); 10];
        let smoothed = retar_smooth_sequence(&seq, 1.5).unwrap();
        assert_eq!(smoothed.len(), 10);
        for out in &smoothed {
            for (a, &b) in out.expression_params.iter().zip(val.iter()) {
                assert!((a - b).abs() < 1e-4, "smooth constant failed: {a} vs {b}");
            }
        }
    }
    #[test]
    fn test_smooth_single_element() {
        let seq = vec![ExpressionState::from_params(vec![1.0, 2.0])];
        let out = retar_smooth_sequence(&seq, 2.0).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0].expression_params[0] - 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_smooth_empty_error() {
        let result = retar_smooth_sequence(&[], 1.0);
        assert!(matches!(result, Err(RetargetError::EmptySequence)));
    }
    #[test]
    fn test_smooth_sigma_zero_passthrough() {
        let seq: Vec<ExpressionState> = (0..4)
            .map(|i| ExpressionState::from_params(vec![i as f32]))
            .collect();
        let out = retar_smooth_sequence(&seq, 0.0).unwrap();
        for (i, s) in out.iter().enumerate() {
            assert!((s.expression_params[0] - i as f32).abs() < 1e-5);
        }
    }
    #[test]
    fn test_resample_same_length() {
        let seq: Vec<ExpressionState> = (0..5)
            .map(|i| ExpressionState::from_params(vec![i as f32]))
            .collect();
        let out = retar_resample_sequence(&seq, 5).unwrap();
        assert_eq!(out.len(), 5);
    }
    #[test]
    fn test_resample_target_1() {
        let seq: Vec<ExpressionState> = (0..5)
            .map(|i| ExpressionState::from_params(vec![i as f32]))
            .collect();
        let out = retar_resample_sequence(&seq, 1).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0].expression_params[0]).abs() < 1e-5);
    }
    #[test]
    fn test_resample_zero_len_error() {
        let seq = vec![ExpressionState::neutral(4)];
        assert!(matches!(
            retar_resample_sequence(&seq, 0),
            Err(RetargetError::InvalidConfig(_))
        ));
    }
    #[test]
    fn test_resample_upsample() {
        let seq: Vec<ExpressionState> = (0..3)
            .map(|i| ExpressionState::from_params(vec![i as f32 * 2.0]))
            .collect();
        let out = retar_resample_sequence(&seq, 5).unwrap();
        assert_eq!(out.len(), 5);
        assert!((out[0].expression_params[0]).abs() < 1e-4);
        assert!((out[4].expression_params[0] - 4.0).abs() < 1e-4);
    }
    #[test]
    fn test_similarity_identical() {
        let s = ExpressionState::from_params(vec![1.0, 0.0, 0.5]);
        let sim = retar_expression_similarity(&s, &s).unwrap();
        assert!((sim - 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_similarity_orthogonal() {
        let a = ExpressionState::from_params(vec![1.0, 0.0]);
        let b = ExpressionState::from_params(vec![0.0, 1.0]);
        let sim = retar_expression_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-5);
    }
    #[test]
    fn test_similarity_opposite() {
        let a = ExpressionState::from_params(vec![1.0, 0.0]);
        let b = ExpressionState::from_params(vec![-1.0, 0.0]);
        let sim = retar_expression_similarity(&a, &b).unwrap();
        assert!((sim + 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_similarity_zero_vector() {
        let z = ExpressionState::from_params(vec![0.0, 0.0]);
        let a = ExpressionState::from_params(vec![1.0, 0.0]);
        let sim = retar_expression_similarity(&z, &a).unwrap();
        assert!(sim.abs() < 1e-5);
    }
    #[test]
    fn test_similarity_dim_mismatch() {
        let a = ExpressionState::from_params(vec![1.0, 0.0]);
        let b = ExpressionState::from_params(vec![1.0, 0.0, 0.5]);
        assert!(matches!(
            retar_expression_similarity(&a, &b),
            Err(RetargetError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_find_neutral_frame_correct_index() {
        let seq = vec![
            ExpressionState::from_params(vec![1.0, 1.0]),
            ExpressionState::from_params(vec![0.01, 0.0]),
            ExpressionState::from_params(vec![2.0, 2.0]),
        ];
        let idx = retar_find_neutral_frame(&seq).unwrap();
        assert_eq!(idx, 1);
    }
    #[test]
    fn test_find_neutral_frame_actual_neutral() {
        let seq = vec![
            ExpressionState::from_params(vec![1.0, 2.0]),
            ExpressionState::neutral(2),
            ExpressionState::from_params(vec![-1.0, 0.5]),
        ];
        let idx = retar_find_neutral_frame(&seq).unwrap();
        assert_eq!(idx, 1);
    }
    #[test]
    fn test_find_neutral_frame_empty_error() {
        assert!(matches!(
            retar_find_neutral_frame(&[]),
            Err(RetargetError::EmptySequence)
        ));
    }
    #[test]
    fn test_mirror_double_mirror_roundtrip() {
        // `retar_mirror_expression` now takes an explicit coefficient-space
        // mirror matrix (see `retar_build_expression_mirror_matrix`'s docs
        // for why a fixed sign pattern can't represent it for a real FLAME
        // basis) and returns a `Result`. Feed it a hand-built involution — a
        // block-diagonal matrix of 3 swap blocks on a 6-dim state — so that
        // applying it twice is guaranteed to reproduce the input exactly,
        // independent of whether `retar_build_expression_mirror_matrix`
        // itself is correct (that's covered separately in `functions.rs`).
        let dim = 6;
        let mut mirror = vec![0.0_f32; dim * dim];
        for block in 0..dim / 2 {
            let i0 = block * 2;
            let i1 = block * 2 + 1;
            mirror[i0 * dim + i1] = 1.0;
            mirror[i1 * dim + i0] = 1.0;
        }
        let s = ExpressionState::from_params(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m = retar_mirror_expression(&s, &mirror).expect("6x6 matrix on a 6-dim state");
        let mm = retar_mirror_expression(&m, &mirror).expect("mirroring again");
        for (a, b) in s.expression_params.iter().zip(mm.expression_params.iter()) {
            assert!((a - b).abs() < 1e-6, "double mirror failed: {a} vs {b}");
        }
    }
    #[test]
    fn test_mirror_applies_a_diagonal_sign_matrix() {
        // A diagonal matrix is the degenerate case where the general
        // matrix-vector mirror reduces to per-coefficient signs — useful
        // here as a simple, independently checkable input for the matrix
        // application itself (not a claim that FLAME mirroring is diagonal
        // in general; see `retar_build_expression_mirror_matrix`'s docs).
        let s = ExpressionState::from_params(vec![1.0, 1.0, 1.0, 1.0]);
        #[rustfmt::skip]
        let mirror = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, -1.0,
        ];
        let m = retar_mirror_expression(&s, &mirror).expect("4x4 diagonal matrix on a 4-dim state");
        assert!((m.expression_params[0] - 1.0).abs() < 1e-6);
        assert!((m.expression_params[1] + 1.0).abs() < 1e-6);
        assert!((m.expression_params[2] - 1.0).abs() < 1e-6);
        assert!((m.expression_params[3] + 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_mirror_flips_jaw_yaw_and_roll_but_preserves_pitch() {
        // Reflecting across the mid-sagittal plane (x = 0) conjugates the
        // jaw rotation by diag(-1, 1, 1): pitch (index 0) is preserved, yaw
        // and roll (indices 1, 2) are negated. The expression coefficients
        // pass through an identity matrix unchanged so only the jaw-pose
        // behavior is under test here.
        let s = ExpressionState::from_params_and_jaw(vec![1.0, 2.0], [0.1, 0.2, 0.3]);
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let m = retar_mirror_expression(&s, &identity).expect("2x2 identity on a 2-dim state");
        assert_eq!(m.expression_params, vec![1.0, 2.0]);
        assert!(
            (m.jaw_pose[0] - 0.1).abs() < 1e-6,
            "pitch must be preserved"
        );
        assert!((m.jaw_pose[1] + 0.2).abs() < 1e-6, "yaw must be negated");
        assert!((m.jaw_pose[2] + 0.3).abs() < 1e-6, "roll must be negated");
    }
    #[test]
    fn test_blend_states_equal_weights() {
        let val = vec![1.0_f32, -0.5, 0.2];
        let s = ExpressionState::from_params(val.clone());
        let out = retar_blend_states(&[s.clone(), s.clone()], &[1.0, 1.0]).unwrap();
        for (a, &b) in out.expression_params.iter().zip(val.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
    #[test]
    fn test_blend_states_empty_error() {
        assert!(matches!(
            retar_blend_states(&[], &[]),
            Err(RetargetError::EmptySequence)
        ));
    }
    #[test]
    fn test_blend_states_mismatched_weights_error() {
        let s = ExpressionState::neutral(4);
        let result = retar_blend_states(&[s], &[0.5, 0.5]);
        assert!(matches!(
            result,
            Err(RetargetError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_blend_states_negative_weight_error() {
        let s = ExpressionState::from_params(vec![1.0]);
        let t = ExpressionState::from_params(vec![2.0]);
        assert!(matches!(
            retar_blend_states(&[s, t], &[0.5, -0.1]),
            Err(RetargetError::InvalidWeight(_))
        ));
    }
    #[test]
    fn test_blend_states_unnormalized_weights() {
        let a = ExpressionState::from_params(vec![0.0]);
        let b = ExpressionState::from_params(vec![4.0]);
        let out = retar_blend_states(&[a, b], &[2.0, 2.0]).unwrap();
        assert!((out.expression_params[0] - 2.0).abs() < 1e-5);
    }
    #[test]
    fn test_slerp_t0_returns_a() {
        let a = ExpressionState::from_params(vec![1.0, 0.0, 0.0]);
        let b = ExpressionState::from_params(vec![0.0, 1.0, 0.0]);
        let out = retar_slerp_states(&a, &b, 0.0).unwrap();
        let norm: f32 = out
            .expression_params
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();
        let a_norm: f32 = a
            .expression_params
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!((norm - a_norm).abs() < 1e-4, "magnitude should match a");
    }
    #[test]
    fn test_slerp_t1_direction_matches_b() {
        let a = ExpressionState::from_params(vec![1.0, 0.0, 0.0]);
        let b = ExpressionState::from_params(vec![0.0, 2.0, 0.0]);
        let out = retar_slerp_states(&a, &b, 1.0).unwrap();
        assert!(out.expression_params[1] > out.expression_params[0]);
    }
    #[test]
    fn test_slerp_dim_mismatch() {
        let a = ExpressionState::from_params(vec![1.0, 0.0]);
        let b = ExpressionState::from_params(vec![1.0, 0.0, 0.5]);
        assert!(matches!(
            retar_slerp_states(&a, &b, 0.5),
            Err(RetargetError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn test_slerp_zero_vector_fallback() {
        let z = ExpressionState::neutral(3);
        let a = ExpressionState::from_params(vec![1.0, 0.0, 0.0]);
        let out = retar_slerp_states(&z, &a, 0.5).unwrap();
        assert_eq!(out.expr_dim, 3);
    }
    #[test]
    fn test_compute_stats_identity_retargeter_low_error() {
        let dim = 4;
        let config = RetargetConfig {
            expr_dim: dim,
            include_jaw: false,
            regularization: 1e-6,
            scale_by_variance: false,
            ..Default::default()
        };
        let mut rng = 0xCAFE_BABE_u64;
        let pairs: Vec<RetargetPair> = (0..6)
            .map(|_| {
                let s = random_state(dim, &mut rng);
                RetargetPair {
                    source: s.clone(),
                    target: s,
                }
            })
            .collect();
        let retargeter = LinearExpressionRetargeter::fit(&pairs, config).unwrap();
        let stats = retar_compute_stats(&retargeter, &pairs).unwrap();
        assert!(
            stats.mean_error < 0.5,
            "mean_error should be low for identity pairs: {}",
            stats.mean_error
        );
    }
    #[test]
    fn test_compute_stats_empty_error() {
        let config = RetargetConfig::default();
        let r = LinearExpressionRetargeter::identity(config);
        assert!(matches!(
            retar_compute_stats(&r, &[]),
            Err(RetargetError::EmptySequence)
        ));
    }
    #[test]
    fn test_format_stats_non_empty() {
        let stats = RetargetStats {
            mean_error: 0.1,
            max_error: 0.5,
            per_dim_error: vec![0.05, 0.1],
            mapping_frobenius: 2.0,
        };
        let s = retar_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("RetargetStats"));
    }
    #[test]
    fn test_format_config_non_empty() {
        let config = RetargetConfig::default();
        let s = retar_format_config(&config);
        assert!(!s.is_empty());
        assert!(s.contains("RetargetConfig"));
    }
    #[test]
    fn test_full_retarget_pipeline() {
        let dim = 6;
        let mut rng = 0xF00D_CAFE_u64;
        let config = RetargetConfig {
            expr_dim: dim,
            regularization: 1e-5,
            scale_by_variance: false,
            include_jaw: false,
            ..Default::default()
        };
        let pairs: Vec<RetargetPair> = (0..10)
            .map(|_| {
                let src = random_state(dim, &mut rng);
                let tgt = src.with_scale(2.0);
                RetargetPair {
                    source: src,
                    target: tgt,
                }
            })
            .collect();
        let retargeter = LinearExpressionRetargeter::fit(&pairs, config).unwrap();
        let test_src = random_state(dim, &mut rng);
        let test_tgt_expected = test_src.with_scale(2.0);
        let test_tgt_got = retargeter.retarget(&test_src).unwrap();
        let err: f32 = test_tgt_got
            .expression_params
            .iter()
            .zip(test_tgt_expected.expression_params.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(err < 0.5, "full pipeline error too large: {err}");
        let stats = retar_compute_stats(&retargeter, &pairs).unwrap();
        assert!(stats.mean_error < 0.5);
        assert!(!retar_format_stats(&stats).is_empty());
    }
}
