//! Test suite for `contrastive_learning` (split out of `mod.rs` to keep
//! that file under the workspace's 2000-line policy limit).

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    // Helper: roughly equal within tolerance
    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // Build a unit vector along axis `axis` of `dim` dimensions.
    fn axis_vec(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[axis] = 1.0;
        v
    }

    // Build orthogonal embeddings
    fn ortho_pair(dim: usize) -> (Vec<f32>, Vec<f32>) {
        (axis_vec(dim, 0), axis_vec(dim, 1))
    }

    // ── ContrastiveLearningConfig ────────────────────────────────────────────

    #[test]
    fn test_config_default_values() {
        let cfg = ContrastiveLearningConfig::default();
        assert!((cfg.temperature - 0.07).abs() < 1e-6);
        assert_eq!(cfg.embedding_dim, 256);
        assert_eq!(cfg.queue_size, 4096);
        assert!((cfg.momentum - 0.999).abs() < 1e-6);
        assert!(cfg.normalize_embeddings);
        assert!(!cfg.hard_negative_mining);
        assert!((cfg.margin - 0.5).abs() < 1e-6);
        assert_eq!(cfg.n_negatives, 0);
    }

    #[test]
    fn test_config_validate_valid() {
        assert!(ContrastiveLearningConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_temperature() {
        let cfg = ContrastiveLearningConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContrastiveError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn test_config_validate_negative_temperature() {
        let cfg = ContrastiveLearningConfig {
            temperature: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContrastiveError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn test_config_validate_zero_dim() {
        let cfg = ContrastiveLearningConfig {
            embedding_dim: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContrastiveError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_zero_queue_size() {
        // Regression: `queue_size` was declared and documented but never
        // validated, so `EmbeddingQueue::from_config` could be handed a
        // config that would panic at the queue layer instead of failing
        // config validation cleanly.
        let cfg = ContrastiveLearningConfig {
            queue_size: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContrastiveError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_momentum_out_of_range() {
        let cfg_too_high = ContrastiveLearningConfig {
            momentum: 1.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg_too_high.validate(),
            Err(ContrastiveError::InvalidConfig(_))
        ));
        let cfg_negative = ContrastiveLearningConfig {
            momentum: -0.1,
            ..Default::default()
        };
        assert!(matches!(
            cfg_negative.validate(),
            Err(ContrastiveError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_embedding_queue_from_config() {
        // Regression: `queue_size` and `embedding_dim` were declared on the
        // config and documented ("Memory-bank capacity (MoCo-style)") but
        // no function ever read `queue_size` to size a queue.
        let cfg = ContrastiveLearningConfig {
            queue_size: 8,
            embedding_dim: 3,
            ..Default::default()
        };
        let q = EmbeddingQueue::from_config(&cfg).expect("valid queue from config");
        assert_eq!(q.capacity(), 8);
        // Enqueue at the configured dim to confirm `dim` was wired too.
        q.clone()
            .enqueue(vec![1.0, 2.0, 3.0], 0)
            .expect("dim=3 embedding should be accepted");
    }

    #[test]
    fn test_embedding_queue_from_config_rejects_zero_queue_size() {
        let cfg = ContrastiveLearningConfig {
            queue_size: 0,
            ..Default::default()
        };
        assert!(EmbeddingQueue::from_config(&cfg).is_err());
    }

    #[test]
    fn test_new_momentum_encoder_uses_config_momentum() {
        // Regression: `momentum` was declared and documented ("Momentum for
        // key-encoder update") but nothing in the module ever constructed a
        // momentum encoder from it.
        let cfg = ContrastiveLearningConfig {
            momentum: 0.9,
            ..Default::default()
        };
        let mut enc = cfg
            .new_momentum_encoder(vec![0.0, 0.0])
            .expect("valid momentum encoder");
        assert!((enc.momentum - 0.9).abs() < 1e-6);
        enc.online_weights = vec![1.0, 1.0];
        enc.update();
        // m_weights = 0.9*0.0 + 0.1*1.0 = 0.1
        assert!((enc.momentum_weights[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_new_momentum_encoder_rejects_invalid_momentum() {
        let cfg = ContrastiveLearningConfig {
            momentum: 1.5,
            ..Default::default()
        };
        assert!(cfg.new_momentum_encoder(vec![0.0]).is_err());
    }

    // ── cl_normalize ─────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_unit_vector_unchanged() {
        let v = vec![1.0f32, 0.0, 0.0];
        let n = cl_normalize(&v);
        assert!(approx(n[0], 1.0, 1e-6));
        assert!(approx(n[1], 0.0, 1e-6));
    }

    #[test]
    fn test_normalize_zero_vector_returns_zeros() {
        let v = vec![0.0f32; 4];
        let n = cl_normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_normalize_arbitrary_vector() {
        let v = vec![3.0f32, 4.0];
        let n = cl_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx(norm, 1.0, 1e-6));
    }

    // ── cl_cosine_sim ─────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_sim_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        let s = cl_cosine_sim(&v, &v).unwrap();
        assert!(approx(s, 1.0, 1e-5));
    }

    #[test]
    fn test_cosine_sim_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let s = cl_cosine_sim(&a, &b).unwrap();
        assert!(approx(s, -1.0, 1e-5));
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let (a, b) = ortho_pair(4);
        let s = cl_cosine_sim(&a, &b).unwrap();
        assert!(approx(s, 0.0, 1e-5));
    }

    #[test]
    fn test_cosine_sim_empty_error() {
        let res = cl_cosine_sim(&[], &[]);
        assert!(matches!(
            res,
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_cosine_sim_mismatch_error() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32];
        assert!(matches!(
            cl_cosine_sim(&a, &b),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    // ── cl_dot ───────────────────────────────────────────────────────────────

    #[test]
    fn test_dot_correct() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let d = cl_dot(&a, &b).unwrap();
        assert!(approx(d, 32.0, 1e-5));
    }

    #[test]
    fn test_dot_mismatch_error() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        assert!(matches!(
            cl_dot(&a, &b),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    // ── cl_l2_distance ───────────────────────────────────────────────────────

    #[test]
    fn test_l2_distance_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!(approx(cl_l2_distance(&v, &v).unwrap(), 0.0, 1e-6));
    }

    #[test]
    fn test_l2_distance_known_pair() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        assert!(approx(cl_l2_distance(&a, &b).unwrap(), 5.0, 1e-5));
    }

    #[test]
    fn test_l2_distance_mismatch_error() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        assert!(matches!(
            cl_l2_distance(&a, &b),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    // ── cl_similarity_matrix ─────────────────────────────────────────────────

    #[test]
    fn test_similarity_matrix_1x1() {
        let v = vec![vec![1.0f32, 0.0, 0.0]];
        let m = cl_similarity_matrix(&v, true).unwrap();
        assert_eq!(m.len(), 1);
        assert!(approx(m[0], 1.0, 1e-5));
    }

    #[test]
    fn test_similarity_matrix_2x2_diagonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let vecs = vec![a, b];
        let m = cl_similarity_matrix(&vecs, true).unwrap();
        assert!(approx(m[0], 1.0, 1e-5)); // (0,0)
        assert!(approx(m[3], 1.0, 1e-5)); // (1,1)
        assert!(approx(m[1], 0.0, 1e-5)); // (0,1) orthogonal
    }

    #[test]
    fn test_similarity_matrix_empty_error() {
        assert!(matches!(
            cl_similarity_matrix(&[], true),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    #[test]
    fn test_similarity_matrix_dim_mismatch_error() {
        let vecs = vec![vec![1.0f32, 0.0], vec![1.0f32]];
        assert!(matches!(
            cl_similarity_matrix(&vecs, false),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    // ── log_sum_exp ──────────────────────────────────────────────────────────

    #[test]
    fn test_log_sum_exp_all_neg_infinity_is_neg_infinity() {
        // Empty softmax support: every logit is -inf.
        let v = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        assert_eq!(log_sum_exp(&v), f32::NEG_INFINITY);
    }

    #[test]
    fn test_log_sum_exp_with_positive_infinity_logit_is_positive_infinity() {
        // Regression: `is_infinite()` is true for BOTH +inf and -inf, so the
        // previous guard `if max.is_infinite() { return NEG_INFINITY }`
        // mapped a +inf logit (reachable via overflow with a small
        // temperature on unnormalized embeddings) to the WRONG sign
        // (-infinity) instead of the mathematically correct +infinity.
        let v = [1.0f32, f32::INFINITY, 3.0];
        assert_eq!(
            log_sum_exp(&v),
            f32::INFINITY,
            "a +inf logit must make log_sum_exp +inf, not -inf"
        );
    }

    #[test]
    fn test_log_sum_exp_finite_matches_naive_formula() {
        // Sanity: the fast-path additions for the +-inf cases must not
        // disturb the ordinary finite computation.
        let v = [1.0f32, 2.0, 3.0];
        let naive = (v.iter().map(|x| x.exp()).sum::<f32>()).ln();
        let got = log_sum_exp(&v);
        assert!((got - naive).abs() < 1e-4, "got={got} naive={naive}");
    }

    // ── cl_nt_xent_loss ──────────────────────────────────────────────────────

    #[test]
    fn test_nt_xent_odd_batch_error() {
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0]; 3];
        assert!(matches!(
            cl_nt_xent_loss(&embs, &cfg),
            Err(ContrastiveError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_nt_xent_too_small_error() {
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0]];
        assert!(matches!(
            cl_nt_xent_loss(&embs, &cfg),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    #[test]
    fn test_nt_xent_identical_pair_low_loss() {
        // When both embeddings in the only pair are identical (collinear),
        // the positive sim = 1, which is the maximum, so loss is minimised.
        let cfg = ContrastiveLearningConfig::default();
        let a = vec![1.0f32, 0.0];
        let embs = vec![a.clone(), a];
        let loss = cl_nt_xent_loss(&embs, &cfg).unwrap();
        // With only one negative (the pair partner), the denominator = just the pos itself,
        // so loss = -log(1) = 0. With 2 samples (one pair) there are no extra negatives.
        // The only non-self sample is the partner => loss → 0.
        assert!(loss.abs() < 1e-3, "loss={loss}");
    }

    #[test]
    fn test_nt_xent_orthogonal_pair_higher_loss() {
        let cfg = ContrastiveLearningConfig::default();
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        // orthogonal positive pair — positive sim = 0
        let embs = vec![a, b];
        let loss_orth = cl_nt_xent_loss(&embs, &cfg).unwrap();
        // identical pair has loss ≈ 0; orthogonal should be larger
        let a2 = vec![1.0f32, 0.0, 0.0, 0.0];
        let embs_ident = vec![a2.clone(), a2];
        let loss_ident = cl_nt_xent_loss(&embs_ident, &cfg).unwrap();
        assert!(
            loss_orth >= loss_ident,
            "orthogonal loss {loss_orth} should be >= identical loss {loss_ident}"
        );
    }

    #[test]
    fn test_nt_xent_larger_batch_positive() {
        let cfg = ContrastiveLearningConfig::default();
        // 4 embeddings = 2 positive pairs
        let embs = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.9f32, 0.1, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.9, 0.1],
        ];
        let loss = cl_nt_xent_loss(&embs, &cfg).unwrap();
        assert!(loss > 0.0, "loss={loss}");
    }

    #[test]
    fn test_nt_xent_temperature_effect() {
        // Higher temperature → softer distribution → lower raw loss magnitude
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![0.0f32, 1.0], // orthogonal — deliberately hard pair
        ];
        let cfg_low_t = ContrastiveLearningConfig {
            temperature: 0.01,
            ..Default::default()
        };
        let cfg_high_t = ContrastiveLearningConfig {
            temperature: 1.0,
            ..Default::default()
        };
        let loss_low = cl_nt_xent_loss(&embs, &cfg_low_t).unwrap();
        let loss_high = cl_nt_xent_loss(&embs, &cfg_high_t).unwrap();
        // Higher T → lower loss magnitude
        assert!(
            loss_high <= loss_low + 1e-3,
            "high-T loss {loss_high} should be <= low-T loss {loss_low}"
        );
    }

    #[test]
    fn test_nt_xent_honours_normalize_embeddings_false() {
        // Regression: `cl_nt_xent_loss` previously normalized every
        // embedding unconditionally, ignoring
        // `config.normalize_embeddings`, unlike its siblings
        // `cl_info_nce_loss` / `cl_supcon_loss` which both branch on it. A
        // caller working in an unnormalized space (magnitudes carrying
        // meaning) got silently forced into unit-normalized similarities.
        // Use pairs with matching direction but differing magnitude so
        // normalization measurably changes the similarity structure.
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![0.1f32, 0.0], // same direction as its pair partner, smaller magnitude
            vec![0.0f32, 5.0],
            vec![0.0f32, 0.6], // same direction as its pair partner, smaller magnitude
        ];
        let cfg_norm = ContrastiveLearningConfig {
            temperature: 1.0,
            normalize_embeddings: true,
            ..Default::default()
        };
        let cfg_unnorm = ContrastiveLearningConfig {
            temperature: 1.0,
            normalize_embeddings: false,
            ..Default::default()
        };
        let loss_norm = cl_nt_xent_loss(&embs, &cfg_norm).unwrap();
        let loss_unnorm = cl_nt_xent_loss(&embs, &cfg_unnorm).unwrap();
        assert!(
            (loss_norm - loss_unnorm).abs() > 1e-3,
            "normalize_embeddings must change the NT-Xent loss for these inputs: \
             normalized={loss_norm} unnormalized={loss_unnorm}"
        );
    }

    #[test]
    fn test_nt_xent_n_negatives_zero_matches_full_batch() {
        // Regression baseline: `n_negatives: 0` (the default) must behave
        // exactly as before this fix — every non-positive sample is used
        // as a negative.
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![-1.0f32, 0.0],
        ];
        let cfg_default = ContrastiveLearningConfig::default();
        let cfg_explicit_full = ContrastiveLearningConfig {
            n_negatives: 3, // equals the number of available negatives (total-2)
            ..Default::default()
        };
        let loss_default = cl_nt_xent_loss(&embs, &cfg_default).unwrap();
        let loss_explicit_full = cl_nt_xent_loss(&embs, &cfg_explicit_full).unwrap();
        assert!(
            (loss_default - loss_explicit_full).abs() < 1e-5,
            "n_negatives=0 and n_negatives=all-available should match: {loss_default} vs {loss_explicit_full}"
        );
    }

    #[test]
    fn test_nt_xent_n_negatives_caps_batch_and_changes_loss() {
        // Regression: `n_negatives` was documented ("Negatives per positive
        // (NT-Xent). Default: uses full batch") but never read by
        // `cl_nt_xent_loss`, which unconditionally used the full batch
        // regardless of this field. Capping the negative set must actually
        // change the computed loss (fewer negatives => an easier
        // discrimination problem => strictly lower loss here, since the
        // dropped negative sits at zero similarity to the anchor which
        // still slightly enlarges the softmax denominator).
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![-1.0f32, 0.0],
        ];
        let cfg_full = ContrastiveLearningConfig::default(); // n_negatives: 0 = full
        let cfg_capped = ContrastiveLearningConfig {
            n_negatives: 1,
            ..Default::default()
        };
        let loss_full = cl_nt_xent_loss(&embs, &cfg_full).unwrap();
        let loss_capped = cl_nt_xent_loss(&embs, &cfg_capped).unwrap();
        assert!(
            (loss_full - loss_capped).abs() > 1e-3,
            "capping n_negatives must change the loss: full={loss_full} capped={loss_capped}"
        );
    }

    // ── cl_info_nce_loss ─────────────────────────────────────────────────────

    #[test]
    fn test_info_nce_empty_anchors_error() {
        let cfg = ContrastiveLearningConfig::default();
        let res = cl_info_nce_loss(&[], &[], &[vec![1.0f32, 0.0]], &cfg);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_info_nce_empty_negatives_error() {
        let cfg = ContrastiveLearningConfig::default();
        let a = vec![vec![1.0f32, 0.0]];
        let p = vec![vec![1.0f32, 0.0]];
        let res = cl_info_nce_loss(&a, &p, &[], &cfg);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_info_nce_anchor_positive_mismatch_error() {
        let cfg = ContrastiveLearningConfig::default();
        let a = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let p = vec![vec![1.0f32, 0.0]];
        let n = vec![vec![-1.0f32, 0.0]];
        let res = cl_info_nce_loss(&a, &p, &n, &cfg);
        assert!(matches!(
            res,
            Err(ContrastiveError::LabelEmbedMismatch { .. })
        ));
    }

    #[test]
    fn test_info_nce_perfect_separation_low_loss() {
        // pos sim ~ 1, neg sim ~ -1 → near-zero loss
        let cfg = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            ..Default::default()
        };
        let a = vec![vec![1.0f32, 0.0]];
        let p = vec![vec![1.0f32, 0.0]]; // identical
        let n = vec![vec![-1.0f32, 0.0]]; // opposite
        let loss = cl_info_nce_loss(&a, &p, &n, &cfg).unwrap();
        assert!(loss < 0.1, "loss={loss}");
    }

    #[test]
    fn test_info_nce_worst_case_high_loss() {
        // pos sim == neg sim → high loss
        let cfg = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            ..Default::default()
        };
        let a = vec![vec![1.0f32, 0.0]];
        let p = vec![vec![0.0f32, 1.0]]; // orthogonal to anchor
        let n = vec![vec![0.0f32, 1.0]]; // same as positive
        let loss_hard = cl_info_nce_loss(&a, &p, &n, &cfg).unwrap();

        let p2 = vec![vec![1.0f32, 0.0]]; // identical to anchor
        let loss_easy = cl_info_nce_loss(&a, &p2, &n, &cfg).unwrap();
        assert!(loss_hard >= loss_easy, "hard={loss_hard} easy={loss_easy}");
    }

    #[test]
    fn test_info_nce_hard_negative_mining_matches_hardest_only_negatives() {
        // Regression: `hard_negative_mining` was documented ("Use only the
        // hardest negative per anchor") but never read by
        // `cl_info_nce_loss` -- every caller got the full negative set
        // regardless of the flag. With mining enabled, the loss over the
        // FULL negative set must equal the loss computed by manually
        // passing only the single most-similar negative (the config option
        // should be behaviourally identical to doing that reduction by
        // hand).
        let cfg_mining = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            hard_negative_mining: true,
            ..Default::default()
        };
        let cfg_no_mining = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            hard_negative_mining: false,
            ..Default::default()
        };
        let a = vec![vec![1.0f32, 0.0]];
        let p = vec![vec![1.0f32, 0.0]];
        // negatives[0] is the hardest (closest to the anchor).
        let negatives = vec![vec![0.99f32, 0.01], vec![0.0f32, 1.0], vec![-1.0f32, 0.0]];
        let hardest_only = vec![negatives[0].clone()];

        let loss_mining = cl_info_nce_loss(&a, &p, &negatives, &cfg_mining).unwrap();
        let loss_manual_hardest = cl_info_nce_loss(&a, &p, &hardest_only, &cfg_no_mining).unwrap();

        assert!(
            (loss_mining - loss_manual_hardest).abs() < 1e-5,
            "hard_negative_mining=true over the full set ({loss_mining}) should equal \
             mining=false over just the hardest negative ({loss_manual_hardest})"
        );

        // And it must differ from using the full negative set without mining
        // (otherwise the flag would still be a no-op).
        let loss_full_no_mining = cl_info_nce_loss(&a, &p, &negatives, &cfg_no_mining).unwrap();
        assert!(
            (loss_mining - loss_full_no_mining).abs() > 1e-3,
            "mining must actually change the result vs. using the full negative set: \
             mining={loss_mining} full={loss_full_no_mining}"
        );
    }

    // ── cl_triplet_loss ───────────────────────────────────────────────────────

    #[test]
    fn test_triplet_zero_when_margin_satisfied() {
        // d(a,p) << d(a,n) - margin
        let a = vec![vec![0.0f32, 0.0]];
        let p = vec![vec![0.1f32, 0.0]];
        let n = vec![vec![10.0f32, 0.0]];
        let loss = cl_triplet_loss(&a, &p, &n, 0.5).unwrap();
        assert!(approx(loss, 0.0, 1e-5));
    }

    #[test]
    fn test_triplet_positive_when_violated() {
        // d(a,p) > d(a,n) - margin
        let a = vec![vec![0.0f32]];
        let p = vec![vec![5.0f32]];
        let n = vec![vec![1.0f32]];
        let loss = cl_triplet_loss(&a, &p, &n, 0.5).unwrap();
        assert!(loss > 0.0, "loss={loss}");
    }

    #[test]
    fn test_triplet_dimension_mismatch_error() {
        let a = vec![vec![1.0f32, 0.0]];
        let p = vec![vec![1.0f32, 0.0]];
        let n = vec![vec![1.0f32]]; // wrong dim
        assert!(matches!(
            cl_triplet_loss(&a, &p, &n, 0.5),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_triplet_empty_error() {
        assert!(matches!(
            cl_triplet_loss(&[], &[], &[], 0.5),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    #[test]
    fn test_triplet_batch_mean() {
        // Two triplets — verify mean
        let a = vec![vec![0.0f32], vec![0.0f32]];
        let p = vec![vec![0.0f32], vec![5.0f32]]; // 2nd violated
        let n = vec![vec![10.0f32], vec![1.0f32]];
        let loss = cl_triplet_loss(&a, &p, &n, 0.5).unwrap();
        // first triplet: d_pos=0, d_neg=10, 0-10+0.5 < 0 → 0
        // second triplet: d_pos=5, d_neg=1, 5-1+0.5=4.5 > 0
        assert!(approx(loss, 2.25, 0.01), "loss={loss}");
    }

    // ── cl_supcon_loss ────────────────────────────────────────────────────────

    #[test]
    fn test_supcon_all_same_label_no_positives_after_self() {
        // With only 1 sample per label-group (n=1, all different labels), loss = 0
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let labels = vec![0usize, 1];
        // each has 0 positives → loss = 0
        let loss = cl_supcon_loss(&embs, &labels, &cfg).unwrap();
        assert!(approx(loss, 0.0, 1e-5));
    }

    #[test]
    fn test_supcon_unique_labels_identity() {
        // unique labels → 0 positives → graceful 0.0
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0], vec![1.0f32, 1.0]];
        let labels = vec![0usize, 1, 2];
        let loss = cl_supcon_loss(&embs, &labels, &cfg).unwrap();
        assert!(approx(loss, 0.0, 1e-5));
    }

    #[test]
    fn test_supcon_two_labels_two_samples_each() {
        let cfg = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            ..Default::default()
        };
        // 4 embeddings: 2 per label
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![0.9f32, 0.1],
            vec![0.0f32, 1.0],
            vec![0.1f32, 0.9],
        ];
        let labels = vec![0usize, 0, 1, 1];
        let loss = cl_supcon_loss(&embs, &labels, &cfg).unwrap();
        assert!(loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn test_supcon_label_embed_mismatch_error() {
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0]; 3];
        let labels = vec![0usize, 1];
        assert!(matches!(
            cl_supcon_loss(&embs, &labels, &cfg),
            Err(ContrastiveError::LabelEmbedMismatch { .. })
        ));
    }

    #[test]
    fn test_supcon_batch_too_small_error() {
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0]];
        let labels = vec![0usize];
        assert!(matches!(
            cl_supcon_loss(&embs, &labels, &cfg),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    // ── EmbeddingQueue ────────────────────────────────────────────────────────

    #[test]
    fn test_queue_zero_capacity_rejected() {
        // Regression: `EmbeddingQueue::new(0, dim)` previously succeeded
        // silently, and the first `enqueue` call would then panic — index
        // out of bounds on `self.entries[self.write_pos]` for an empty
        // `Vec` (the `entries.len() < capacity` branch is `0 < 0 = false`,
        // so it fell into the "overwrite in place" branch on an empty
        // buffer), or divide-by-zero on `(write_pos + 1) % capacity` if
        // that were reached first.
        let result = EmbeddingQueue::new(0, 4);
        assert!(
            matches!(result, Err(ContrastiveError::InvalidConfig(_))),
            "capacity=0 must be rejected at construction, not panic on first enqueue"
        );
    }

    #[test]
    fn test_queue_zero_dim_rejected() {
        let result = EmbeddingQueue::new(4, 0);
        assert!(matches!(result, Err(ContrastiveError::InvalidConfig(_))));
    }

    #[test]
    fn test_queue_enqueue_and_wrap() {
        let mut q = EmbeddingQueue::new(2, 2).expect("valid queue");
        q.enqueue(vec![1.0f32, 0.0], 0).unwrap();
        q.enqueue(vec![0.0f32, 1.0], 1).unwrap();
        assert!(q.is_full());
        // Wrap: overwrite slot 0
        q.enqueue(vec![0.5f32, 0.5], 2).unwrap();
        assert_eq!(q.len(), 2); // still 2
        assert_eq!(q.total_enqueued(), 3);
    }

    #[test]
    fn test_queue_dim_mismatch_error() {
        let mut q = EmbeddingQueue::new(4, 3).expect("valid queue");
        let res = q.enqueue(vec![1.0f32, 0.0], 0);
        assert!(matches!(
            res,
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_queue_get_negatives_excludes_same_label() {
        let mut q = EmbeddingQueue::new(10, 2).expect("valid queue");
        q.enqueue(vec![1.0f32, 0.0], 0).unwrap();
        q.enqueue(vec![0.0f32, 1.0], 1).unwrap();
        q.enqueue(vec![0.5f32, 0.5], 0).unwrap();

        let negs = q.get_negatives(0);
        assert_eq!(negs.len(), 1); // only label=1
    }

    #[test]
    fn test_queue_enqueue_batch_mismatch_error() {
        let mut q = EmbeddingQueue::new(10, 2).expect("valid queue");
        let embs = vec![vec![1.0f32, 0.0]; 3];
        let labels = vec![0usize, 1]; // len mismatch
        assert!(matches!(
            q.enqueue_batch(&embs, &labels),
            Err(ContrastiveError::LabelEmbedMismatch { .. })
        ));
    }

    #[test]
    fn test_queue_is_empty_initially() {
        let q = EmbeddingQueue::new(4, 2).expect("valid queue");
        assert!(q.is_empty());
        assert!(!q.is_full());
    }

    #[test]
    fn test_queue_enqueue_batch_ok() {
        let mut q = EmbeddingQueue::new(10, 2).expect("valid queue");
        let embs = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let labels = vec![0usize, 1];
        q.enqueue_batch(&embs, &labels).unwrap();
        assert_eq!(q.len(), 2);
    }

    // ── cl_mine_hard_negatives ────────────────────────────────────────────────

    #[test]
    fn test_mine_hard_negatives_count() {
        let anchors = vec![vec![1.0f32, 0.0]];
        let a_labels = vec![0usize];
        let negatives = vec![
            vec![0.9f32, 0.1],  // closest
            vec![0.0f32, 1.0],  // orthogonal
            vec![-1.0f32, 0.0], // farthest
        ];
        let n_labels = vec![1usize, 1, 1];
        let result = cl_mine_hard_negatives(&anchors, &a_labels, &negatives, &n_labels, 2).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn test_mine_hard_negatives_label_filtering() {
        let anchors = vec![vec![1.0f32, 0.0]];
        let a_labels = vec![0usize];
        let negatives = vec![
            vec![0.9f32, 0.1],  // label 0 — same as anchor → excluded
            vec![-1.0f32, 0.0], // label 1 — valid negative
        ];
        let n_labels = vec![0usize, 1];
        let result = cl_mine_hard_negatives(&anchors, &a_labels, &negatives, &n_labels, 5).unwrap();
        // only one valid negative (label 1)
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0], 1); // index of the label-1 entry
    }

    #[test]
    fn test_mine_hard_negatives_empty_anchors_error() {
        let res = cl_mine_hard_negatives(&[], &[], &[vec![1.0f32]], &[0usize], 1);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_mine_hard_negatives_hardest_first() {
        // Anchor = [1, 0]. Negatives: [0.9,0.1] sim≈0.99, [-1,0] sim=-1
        // With n_hard=1 → should pick index 0
        let anchors = vec![vec![1.0f32, 0.0]];
        let a_labels = vec![0usize];
        let negatives = vec![vec![-1.0f32, 0.0], vec![0.9f32, 0.1]];
        let n_labels = vec![1usize, 1];
        let result = cl_mine_hard_negatives(&anchors, &a_labels, &negatives, &n_labels, 1).unwrap();
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0], 1); // index 1 has the higher cosine similarity
    }

    #[test]
    fn test_mine_hard_negatives_unnormalized_inputs_still_rank_correctly() {
        // Regression guard for hoisting negative-normalization out of the
        // per-anchor loop: ranking must be unaffected by negatives' raw
        // magnitudes (cosine similarity is scale-invariant), across
        // multiple anchors, since a batched pre-normalization pass is only
        // equivalent to per-anchor normalization if every negative really
        // is normalized exactly once and reused unchanged for every
        // anchor.
        let anchors = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let a_labels = vec![0usize, 0usize];
        // Same directions as before but with very different (unnormalized)
        // magnitudes, to make a normalization bug (or a stale per-anchor
        // cache) visible.
        let negatives = vec![
            vec![9.0f32, 1.0],  // ~ [0.9, 0.1] direction, closest to anchor 0
            vec![0.0f32, 50.0], // ~ [0, 1] direction, closest to anchor 1
            vec![-3.0f32, 0.0], // ~ [-1, 0] direction, farthest from both
        ];
        let n_labels = vec![1usize, 1, 1];
        let result = cl_mine_hard_negatives(&anchors, &a_labels, &negatives, &n_labels, 1).unwrap();
        assert_eq!(
            result[0][0], 0,
            "anchor 0's hardest negative should be index 0"
        );
        assert_eq!(
            result[1][0], 1,
            "anchor 1's hardest negative should be index 1"
        );
    }

    // ── cl_mine_semi_hard_negatives ───────────────────────────────────────────

    #[test]
    fn test_mine_semi_hard_negatives_basic() {
        // anchor=[0], positive=[1] (d=1), negatives: [2] (d=2, semi-hard if margin>1), [5] (d=5, too far)
        let a = vec![vec![0.0f32]];
        let p = vec![vec![1.0f32]];
        let n = vec![vec![2.0f32], vec![5.0f32]];
        let margin = 2.0; // d_pos + margin = 3
        let result = cl_mine_semi_hard_negatives(&a, &p, &n, margin).unwrap();
        assert_eq!(result.len(), 1);
        // n[0]: d=2 > d_pos=1, 2 < 1+2=3 → semi-hard
        // n[1]: d=5, 5 >= 3 → not semi-hard
        assert!(result[0].contains(&0));
        assert!(!result[0].contains(&1));
    }

    #[test]
    fn test_mine_semi_hard_negatives_empty_error() {
        let res = cl_mine_semi_hard_negatives(&[], &[], &[vec![1.0f32]], 0.5);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_mine_semi_hard_negatives_none_qualify() {
        // All negatives too far
        let a = vec![vec![0.0f32]];
        let p = vec![vec![0.1f32]]; // d_pos = 0.1
        let n = vec![vec![100.0f32]]; // d = 100, far beyond d_pos + margin=0.6
        let result = cl_mine_semi_hard_negatives(&a, &p, &n, 0.5).unwrap();
        assert!(result[0].is_empty());
    }

    // ── cl_alignment ─────────────────────────────────────────────────────────

    #[test]
    fn test_alignment_identical_pairs() {
        let v = vec![1.0f32, 0.0, 0.0];
        let a = vec![v.clone(), v.clone()];
        let b = vec![v.clone(), v.clone()];
        let align = cl_alignment(&a, &b).unwrap();
        assert!(approx(align, 1.0, 1e-5));
    }

    #[test]
    fn test_alignment_orthogonal_pairs() {
        let e1 = vec![1.0f32, 0.0];
        let e2 = vec![0.0f32, 1.0];
        let a = vec![e1.clone()];
        let b = vec![e2.clone()];
        let align = cl_alignment(&a, &b).unwrap();
        assert!(approx(align, 0.0, 1e-5));
    }

    #[test]
    fn test_alignment_empty_error() {
        let res = cl_alignment(&[], &[]);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_alignment_mismatch_error() {
        let a = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let b = vec![vec![1.0f32, 0.0]];
        assert!(matches!(
            cl_alignment(&a, &b),
            Err(ContrastiveError::LabelEmbedMismatch { .. })
        ));
    }

    // ── cl_uniformity ─────────────────────────────────────────────────────────

    #[test]
    fn test_uniformity_single_embedding_zero() {
        let embs = vec![vec![1.0f32, 0.0]];
        let u = cl_uniformity(&embs).unwrap();
        assert!(approx(u, 0.0, 1e-5));
    }

    #[test]
    fn test_uniformity_many_embeddings_negative() {
        // Spread embeddings on the unit circle.
        // uniformity = -log(mean exp(-2||z_i - z_j||^2)).
        // For spread points, mean_exp < 1, so -log(mean_exp) > 0.
        // The metric is called "negative" because lower = more uniform.
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![-1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![0.0f32, -1.0],
        ];
        let u = cl_uniformity(&embs).unwrap();
        // Spread embeddings → large positive value (far from 0)
        assert!(u > 0.0, "uniformity={u}");
    }

    #[test]
    fn test_uniformity_empty_error() {
        assert!(matches!(
            cl_uniformity(&[]),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    #[test]
    fn test_uniformity_normalizes_unnormalized_inputs() {
        // Regression: the metric's own doc says "more uniformly
        // distributed on the hypersphere", which is only meaningful for
        // unit-norm inputs, but the implementation previously consumed
        // `embeddings` raw with no normalization step. Two batches that
        // share the same directions but differ in magnitude must now
        // produce the identical result.
        let embs_large_magnitude = vec![
            vec![10.0f32, 0.0],
            vec![-5.0f32, 0.0],
            vec![0.0f32, 20.0],
            vec![0.0f32, -1.0],
        ];
        let embs_unit = vec![
            vec![1.0f32, 0.0],
            vec![-1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![0.0f32, -1.0],
        ];
        let u_large = cl_uniformity(&embs_large_magnitude).unwrap();
        let u_unit = cl_uniformity(&embs_unit).unwrap();
        assert!(
            approx(u_large, u_unit, 1e-4),
            "uniformity must be direction-only (scale-invariant): {u_large} vs {u_unit}"
        );
    }

    // ── cl_uniformity_sampled ────────────────────────────────────────────────

    #[test]
    fn test_uniformity_sampled_covers_all_pairs_matches_exact() {
        // When max_pairs comfortably exceeds n*(n-1)/2, the sampled
        // estimate should closely track the exact value (not required to
        // be bit-identical, since sampling is still with replacement, but
        // close for a generous budget).
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![-1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![0.0f32, -1.0],
        ];
        let exact = cl_uniformity(&embs).unwrap();
        let sampled = cl_uniformity_sampled(&embs, 5000, 42).unwrap();
        // With only n=4 embeddings (6 distinct unordered pairs) the
        // with-replacement sampler cannot converge to zero variance no
        // matter how large `max_pairs` is, so this tolerance is generous
        // rather than tight (empirically the gap stays under ~0.02 for a
        // 5000-pair budget once index-pair sampling is unbiased).
        assert!(
            approx(exact, sampled, 0.1),
            "sampled estimate should approximate the exact value with a generous budget: \
             exact={exact} sampled={sampled}"
        );
    }

    #[test]
    fn test_uniformity_sampled_index_pairs_are_unbiased() {
        // Regression: an earlier version of this sampler drew `j`
        // uniformly in [0, n) and remapped `j == i` to `(i + 1) % n`,
        // which made `j = i + 1` roughly TWICE as likely as any other
        // index (reachable by both "drew i+1 directly" and "drew i,
        // bumped"). Verify the fix by checking that, holding one anchor
        // fixed, every other index is drawn with roughly equal frequency.
        // (Indirect test: `cl_uniformity_sampled` doesn't expose raw index
        // draws, so this measures the same distance-sample distribution
        // it would produce for a population whose distances make the bias
        // visible -- 4 embeddings at 3 distinct pairwise distances from a
        // "hub" at index 0.)
        let hub = vec![0.0f32, 0.0];
        let near = vec![0.1f32, 0.0]; // index 1: small distance from hub
        let mid = vec![1.0f32, 0.0]; // index 2: medium distance from hub
        let far = vec![5.0f32, 0.0]; // index 3: large distance from hub
        let embs = vec![hub, near, mid, far];

        // A biased sampler over-selects the pair (0, 1) [index i+1], which
        // has the SMALLEST pairwise distance among {(0,1),(0,2),(0,3)} and
        // among all 6 pairs overall, and therefore the LARGEST
        // exp(-2*dist^2) contribution -- so a bias toward it would push
        // the sampled uniformity value systematically HIGHER (more
        // "clumped") than the unbiased exact value, not just noisily off.
        // Assert magnitude of the gap stays small and unsigned-random
        // rather than large and one-directional, using a wide pair budget
        // to suppress ordinary sampling noise.
        let exact = cl_uniformity(&embs).unwrap();
        let mut max_abs_diff = 0.0f32;
        for seed in 1u64..=20 {
            let sampled = cl_uniformity_sampled(&embs, 20_000, seed).unwrap();
            max_abs_diff = max_abs_diff.max((exact - sampled).abs());
        }
        assert!(
            max_abs_diff < 0.05,
            "sampled estimates should track the exact value closely across seeds \
             once index-pair sampling is unbiased; max |diff| across 20 seeds = {max_abs_diff}"
        );
    }

    #[test]
    fn test_uniformity_sampled_bounds_pair_count_for_large_batch() {
        // The whole point of the sampled variant: it must not blow up to
        // O(n^2) for a large n. This is a smoke test that it completes and
        // returns a finite result promptly for a batch size where the
        // exact O(n^2) computation would be comparatively expensive.
        let n = 500usize;
        let mut state = 1u64;
        let embs: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..8)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state as f32 / u64::MAX as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect();
        let result = cl_uniformity_sampled(&embs, 1000, 7).unwrap();
        assert!(result.is_finite(), "result must be finite, got {result}");
    }

    #[test]
    fn test_uniformity_sampled_empty_error() {
        assert!(matches!(
            cl_uniformity_sampled(&[], 100, 1),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    // ── cl_update_stats ───────────────────────────────────────────────────────

    #[test]
    fn test_update_stats_ema_decay() {
        let mut stats = ContrastiveStats::default();
        cl_update_stats(&mut stats, 1.0, 0.8, 0.2);
        // After first update, ema = 0.99*0 + 0.01*1.0 = 0.01
        assert!(approx(stats.ema_loss, 0.01, 1e-5));
        cl_update_stats(&mut stats, 1.0, 0.8, 0.2);
        // ema = 0.99*0.01 + 0.01*1.0 = 0.0099 + 0.01 = 0.0199
        assert!(approx(stats.ema_loss, 0.0199, 1e-4));
    }

    #[test]
    fn test_update_stats_mean_loss() {
        let mut stats = ContrastiveStats::default();
        cl_update_stats(&mut stats, 2.0, 0.5, 0.1);
        cl_update_stats(&mut stats, 4.0, 0.5, 0.1);
        assert!(approx(stats.mean_loss, 3.0, 1e-5));
    }

    #[test]
    fn test_update_stats_n_pairs_increment() {
        let mut stats = ContrastiveStats::default();
        for _ in 0..5 {
            cl_update_stats(&mut stats, 1.0, 0.5, 0.3);
        }
        assert_eq!(stats.n_pairs, 5);
    }

    // ── cl_update_geometry_stats ─────────────────────────────────────────────

    #[test]
    fn test_update_geometry_stats_populates_alignment_and_uniformity() {
        // Regression: `cl_update_stats` never touched `alignment` /
        // `uniformity`, so they stayed at their `Default` value of 0.0
        // forever. `cl_alignment` and `cl_uniformity` existed but were
        // never wired to `ContrastiveStats`.
        let mut stats = ContrastiveStats::default();
        cl_update_stats(&mut stats, 1.0, 0.8, 0.1);

        let pos_a = vec![vec![1.0f32, 0.0]];
        let pos_b = vec![vec![1.0f32, 0.0]]; // identical → alignment = 1.0
        let all_embeddings = vec![
            vec![1.0f32, 0.0],
            vec![-1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![0.0f32, -1.0],
        ];
        cl_update_geometry_stats(&mut stats, &pos_a, &pos_b, &all_embeddings)
            .expect("geometry stats update");

        assert!(
            approx(stats.alignment, 1.0, 1e-5),
            "alignment should be 1.0 for identical pairs, got {}",
            stats.alignment
        );
        assert!(
            stats.uniformity > 0.0,
            "uniformity should be nonzero for spread embeddings, got {}",
            stats.uniformity
        );
    }

    #[test]
    fn test_update_geometry_stats_propagates_errors() {
        let mut stats = ContrastiveStats::default();
        let result = cl_update_geometry_stats(&mut stats, &[], &[], &[vec![1.0f32]]);
        assert!(matches!(result, Err(ContrastiveError::BatchTooSmall(_))));
    }

    // ── formatting ────────────────────────────────────────────────────────────

    #[test]
    fn test_format_stats_non_empty() {
        let stats = ContrastiveStats::default();
        let s = cl_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("ContrastiveStats"));
    }

    #[test]
    fn test_format_config_non_empty() {
        let cfg = ContrastiveLearningConfig::default();
        let s = cl_format_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("ContrastiveLearningConfig"));
    }

    // ── PRNG smoke test ───────────────────────────────────────────────────────

    #[test]
    fn test_xorshift_non_zero() {
        let mut state = 12345u64;
        for _ in 0..100 {
            assert_ne!(xorshift64(&mut state), 0);
        }
    }

    /// Regression: the module-private `xorshift_f32` was removed (nothing in
    /// this module draws a float), so the one property that still matters for
    /// the surviving `xorshift64` is that `cl_uniformity_sampled` stays
    /// reproducible for a given seed and varies with the seed.
    #[test]
    fn test_uniformity_sampled_is_seed_reproducible() {
        let embeddings = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.0, 1.0],
            vec![0.7f32, 0.7, 0.0],
            vec![-1.0f32, 0.0, 0.0],
        ];
        let a = cl_uniformity_sampled(&embeddings, 64, 42).expect("valid sampled uniformity");
        let b = cl_uniformity_sampled(&embeddings, 64, 42).expect("valid sampled uniformity");
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "same seed must give same estimate"
        );

        let c = cl_uniformity_sampled(&embeddings, 64, 1337).expect("valid sampled uniformity");
        assert_ne!(
            a.to_bits(),
            c.to_bits(),
            "a different seed must draw a different pair sample"
        );
    }

    // ── Integration: queue → info_nce ─────────────────────────────────────────

    #[test]
    fn test_queue_then_info_nce() {
        let mut q = EmbeddingQueue::new(16, 2).expect("valid queue");
        let neg_embs = vec![vec![-1.0f32, 0.0], vec![0.0f32, -1.0]];
        q.enqueue_batch(&neg_embs, &[1usize, 1]).unwrap();

        let anchors = vec![vec![1.0f32, 0.0]];
        let positives = vec![vec![1.0f32, 0.0]];
        let negatives: Vec<Vec<f32>> = q.as_slice().to_vec();

        let cfg = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            ..Default::default()
        };
        let loss = cl_info_nce_loss(&anchors, &positives, &negatives, &cfg).unwrap();
        assert!(loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn test_triplet_with_hard_negatives() {
        let anchors = vec![vec![1.0f32, 0.0]];
        let a_labels = vec![0usize];
        let neg_pool = vec![
            vec![0.9f32, 0.1],  // hardest
            vec![-1.0f32, 0.0], // easiest
        ];
        let n_labels = vec![1usize, 1];
        let hard_idxs =
            cl_mine_hard_negatives(&anchors, &a_labels, &neg_pool, &n_labels, 1).unwrap();
        let hard_neg = neg_pool[hard_idxs[0][0]].clone();

        let positives = vec![vec![1.0f32, 0.0]];
        let negatives = vec![hard_neg];
        // d(a,p)=0, d(a,n)≈0.14, margin=0.5 → loss > 0
        let loss = cl_triplet_loss(&anchors, &positives, &negatives, 0.5).unwrap();
        assert!(loss > 0.0, "loss={loss}");
    }
}
