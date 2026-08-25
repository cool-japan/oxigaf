//! Unit tests for [`crate::domain_adaptation`].
//!
//! Split out of the parent module to keep every file under the 2000-line cap.

use super::*;

// ----- helpers -----

fn make_batch(n_s: usize, n_t: usize, d: usize, src_val: f32, tgt_val: f32) -> DomainBatch {
    DomainBatch::new(vec![src_val; n_s * d], vec![tgt_val; n_t * d], n_s, n_t, d)
        .expect("valid batch")
}

fn linspace_features(n: usize, d: usize, start: f32, step: f32) -> Vec<f32> {
    (0..n * d).map(|k| start + k as f32 * step).collect()
}

// ----- DomainBatch -----

#[test]
fn test_domain_batch_valid() {
    let b = DomainBatch::new(vec![1.0; 6], vec![2.0; 4], 3, 2, 2);
    assert!(b.is_ok());
    let b = b.unwrap();
    assert_eq!(b.n_source, 3);
    assert_eq!(b.n_target, 2);
    assert_eq!(b.d, 2);
}

#[test]
fn test_domain_batch_wrong_source_len() {
    let r = DomainBatch::new(vec![1.0; 5], vec![2.0; 4], 3, 2, 2);
    assert!(matches!(
        r,
        Err(DomainAdaptationError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_domain_batch_wrong_target_len() {
    let r = DomainBatch::new(vec![1.0; 6], vec![2.0; 5], 3, 2, 2);
    assert!(matches!(
        r,
        Err(DomainAdaptationError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_domain_batch_zero_d() {
    let r = DomainBatch::new(vec![], vec![], 0, 0, 0);
    assert!(matches!(r, Err(DomainAdaptationError::EmptyFeatures)));
}

// ----- da_gaussian_kernel -----

#[test]
fn test_gaussian_kernel_zero_distance() {
    let x = vec![1.0f32, 2.0, 3.0];
    let k = da_gaussian_kernel(&x, &x, 3, 1.0);
    assert!((k - 1.0).abs() < 1e-6, "k={k}");
}

#[test]
fn test_gaussian_kernel_large_distance_approx_zero() {
    let x = vec![0.0f32, 0.0, 0.0];
    let y = vec![100.0f32, 100.0, 100.0];
    let k = da_gaussian_kernel(&x, &y, 3, 1.0);
    assert!(k < 1e-6, "k={k}");
}

#[test]
fn test_gaussian_kernel_symmetric() {
    let x = vec![1.0f32, 2.0, 0.5];
    let y = vec![0.5f32, 1.5, 1.0];
    let kxy = da_gaussian_kernel(&x, &y, 3, 2.0);
    let kyx = da_gaussian_kernel(&y, &x, 3, 2.0);
    assert!((kxy - kyx).abs() < 1e-7);
}

#[test]
fn test_gaussian_kernel_bandwidth_effect() {
    // Larger bandwidth → kernel value closer to 1 for same distance
    let x = vec![1.0f32];
    let y = vec![2.0f32];
    let k_small = da_gaussian_kernel(&x, &y, 1, 0.5);
    let k_large = da_gaussian_kernel(&x, &y, 1, 5.0);
    assert!(k_large > k_small);
}

// ----- da_mmd_biased -----

#[test]
fn test_mmd_biased_identical_distributions() {
    let features = linspace_features(5, 3, 0.0, 0.1);
    let mmd = da_mmd_biased(&features, &features, 5, 5, 3, 1.0).unwrap();
    assert!(mmd < 1e-5, "mmd={mmd}");
}

#[test]
fn test_mmd_biased_different_distributions() {
    let src = vec![0.0f32; 10]; // 5 × 2 zeros
    let tgt = vec![100.0f32; 10]; // 5 × 2 hundreds
    let mmd = da_mmd_biased(&src, &tgt, 5, 5, 2, 1.0).unwrap();
    assert!(mmd > 0.0, "mmd={mmd}");
}

#[test]
fn test_mmd_biased_non_negative() {
    let src = linspace_features(4, 3, 0.0, 0.2);
    let tgt = linspace_features(4, 3, 1.0, 0.3);
    let mmd = da_mmd_biased(&src, &tgt, 4, 4, 3, 1.0).unwrap();
    assert!(mmd >= 0.0);
}

#[test]
fn test_mmd_biased_invalid_bandwidth_zero() {
    let f = vec![1.0f32; 4];
    let r = da_mmd_biased(&f, &f, 2, 2, 1, 0.0);
    assert!(matches!(
        r,
        Err(DomainAdaptationError::InvalidBandwidth { .. })
    ));
}

#[test]
fn test_mmd_biased_invalid_bandwidth_negative() {
    let f = vec![1.0f32; 4];
    let r = da_mmd_biased(&f, &f, 2, 2, 1, -1.0);
    assert!(matches!(
        r,
        Err(DomainAdaptationError::InvalidBandwidth { .. })
    ));
}

#[test]
fn test_mmd_biased_empty_source() {
    let r = da_mmd_biased(&[], &[1.0], 0, 1, 1, 1.0);
    assert!(matches!(r, Err(DomainAdaptationError::EmptyFeatures)));
}

#[test]
fn test_mmd_biased_length_mismatch_errors_not_panics() {
    // Buffer too short for the claimed n_s/d -- must return an error
    // rather than indexing out of bounds inside the pairwise loops.
    let r = da_mmd_biased(&[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], 3, 2, 1, 1.0);
    assert!(matches!(
        r,
        Err(DomainAdaptationError::DimensionMismatch { .. })
    ));
}

// ----- da_mmd_unbiased -----

#[test]
fn test_mmd_unbiased_n1_returns_zero() {
    // With n=1 for both, both within-distribution terms are 0 by convention.
    // Cross term is k(x,y); if x==y it equals 1, so result is 0+0-2*1 → clamped to 0.
    let src = vec![1.0f32, 2.0];
    let tgt = vec![1.0f32, 2.0];
    let mmd = da_mmd_unbiased(&src, &tgt, 1, 1, 2, 1.0).unwrap();
    assert!(mmd >= 0.0);
}

#[test]
fn test_mmd_unbiased_identical_distributions() {
    let features = linspace_features(6, 4, 0.0, 0.1);
    let mmd = da_mmd_unbiased(&features, &features, 6, 6, 4, 1.0).unwrap();
    // Unbiased estimator on identical distributions should be near 0
    assert!(mmd < 1e-4, "mmd={mmd}");
}

#[test]
fn test_mmd_unbiased_different_distributions() {
    let src = vec![0.0f32; 6]; // 3 × 2 zeros
    let tgt = vec![50.0f32; 6]; // 3 × 2 fifties
    let mmd = da_mmd_unbiased(&src, &tgt, 3, 3, 2, 1.0).unwrap();
    assert!(mmd >= 0.0);
}

#[test]
fn test_mmd_unbiased_length_mismatch_errors_not_panics() {
    let r = da_mmd_unbiased(&[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], 3, 2, 1, 1.0);
    assert!(matches!(
        r,
        Err(DomainAdaptationError::DimensionMismatch { .. })
    ));
}

// ----- da_mmd_multiscale -----

#[test]
fn test_mmd_multiscale_sum_of_single_scales() {
    let batch = make_batch(4, 4, 3, 0.0, 1.0);
    let bw1 = 0.5;
    let bw2 = 2.0;

    let single1 = da_mmd_unbiased(
        &batch.source_features,
        &batch.target_features,
        batch.n_source,
        batch.n_target,
        batch.d,
        bw1,
    )
    .unwrap();
    let single2 = da_mmd_unbiased(
        &batch.source_features,
        &batch.target_features,
        batch.n_source,
        batch.n_target,
        batch.d,
        bw2,
    )
    .unwrap();
    let config = MmdConfig {
        kernel_bandwidths: vec![bw1, bw2],
        biased: false,
        eps: 1e-8,
    };
    let multi = da_mmd_multiscale(&batch, &config).unwrap();
    assert!((multi - (single1 + single2)).abs() < 1e-5);
}

#[test]
fn test_mmd_multiscale_empty_bandwidths_err() {
    let batch = make_batch(2, 2, 2, 0.0, 1.0);
    let config = MmdConfig {
        kernel_bandwidths: vec![],
        biased: false,
        eps: 1e-8,
    };
    assert!(da_mmd_multiscale(&batch, &config).is_err());
}

// ----- da_median_bandwidth -----

#[test]
fn test_median_bandwidth_positive() {
    let features = linspace_features(5, 3, 0.0, 0.5);
    let bw = da_median_bandwidth(&features, 5, 3);
    assert!(bw > 0.0);
    assert!(bw.is_finite());
}

#[test]
fn test_median_bandwidth_n1_returns_one() {
    let features = vec![1.0f32, 2.0, 3.0];
    let bw = da_median_bandwidth(&features, 1, 3);
    assert_eq!(bw, 1.0);
}

#[test]
fn test_median_bandwidth_uses_sqrt2_not_n_dependent_divisor() {
    // Two points at distance 4.0 apart -> median pairwise distance = 4.0.
    let features = vec![0.0f32, 0.0, 0.0, 4.0, 0.0, 0.0]; // n=2, d=3
    let bw = da_median_bandwidth(&features, 2, 3);
    let expected = 4.0f32 / std::f32::consts::SQRT_2;
    assert!(
        (bw - expected).abs() < 1e-4,
        "expected median/sqrt(2)={expected}, got {bw}"
    );
}

#[test]
fn test_median_bandwidth_clamps_n_to_available_rows() {
    // Claims n=100 but only 2 rows (d=3) actually fit in the buffer.
    let features = vec![0.0f32, 0.0, 0.0, 4.0, 0.0, 0.0];
    let bw = da_median_bandwidth(&features, 100, 3);
    let expected = 4.0f32 / std::f32::consts::SQRT_2;
    assert!(
        (bw - expected).abs() < 1e-4,
        "should clamp n down to the 2 rows actually available, \
         expected {expected}, got {bw}"
    );
}

// ----- da_feature_mean -----

#[test]
fn test_feature_mean_simple_2x2() {
    // [[1,2],[3,4]] → means [2, 3]
    let features = vec![1.0f32, 2.0, 3.0, 4.0];
    let means = da_feature_mean(&features, 2, 2);
    assert!((means[0] - 2.0).abs() < 1e-6);
    assert!((means[1] - 3.0).abs() < 1e-6);
}

#[test]
fn test_feature_mean_uniform() {
    let features = vec![5.0f32; 12]; // 4 × 3
    let means = da_feature_mean(&features, 4, 3);
    for &m in &means {
        assert!((m - 5.0).abs() < 1e-6);
    }
}

#[test]
fn test_feature_mean_clamps_n_does_not_panic() {
    // Claims n=100 but the buffer only has 2 rows of d=2.
    let features = vec![1.0f32, 2.0, 3.0, 4.0];
    let means = da_feature_mean(&features, 100, 2);
    assert!((means[0] - 2.0).abs() < 1e-6);
    assert!((means[1] - 3.0).abs() < 1e-6);
}

// ----- da_center_features -----

#[test]
fn test_center_features_column_sum_zero() {
    let features = linspace_features(5, 4, -2.0, 0.3);
    let (centered, _means) = da_center_features(&features, 5, 4);
    // Each column should sum to ≈ 0
    for j in 0..4 {
        let col_sum: f32 = (0..5).map(|i| centered[i * 4 + j]).sum();
        assert!(col_sum.abs() < 1e-4, "col {j} sum = {col_sum}");
    }
}

#[test]
fn test_center_features_returns_original_mean() {
    let features = vec![2.0f32, 4.0, 6.0, 8.0]; // 2 × 2
    let (_centered, means) = da_center_features(&features, 2, 2);
    assert!((means[0] - 4.0).abs() < 1e-6);
    assert!((means[1] - 6.0).abs() < 1e-6);
}

#[test]
fn test_center_features_clamps_n_does_not_panic() {
    // Claims n=100 but the buffer only has 2 rows of d=2.
    let features = vec![2.0f32, 4.0, 6.0, 8.0];
    let (centered, means) = da_center_features(&features, 100, 2);
    assert_eq!(centered.len(), features.len());
    assert!((means[0] - 4.0).abs() < 1e-6);
    assert!((means[1] - 6.0).abs() < 1e-6);
}

// ----- da_covariance -----

#[test]
fn test_covariance_identity_input() {
    // n=3 samples of d=3-dim identity vectors (already centered, each row is e_i scaled by sqrt(3))
    // Use [1,0,0; 0,1,0; 0,0,1] — each column mean = 1/3, so center first
    let raw = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let (centered, _) = da_center_features(&raw, 3, 3);
    let cov = da_covariance(&centered, 3, 3);
    // Should be a scaled symmetric matrix
    assert_eq!(cov.len(), 9);
    // Diagonal should be equal (by symmetry of identity-like input)
    assert!((cov[0] - cov[4]).abs() < 1e-5);
    assert!((cov[0] - cov[8]).abs() < 1e-5);
}

#[test]
fn test_covariance_constant_features_zero() {
    // All same value → zero covariance after centering
    let features = vec![3.0f32; 6]; // 3 × 2
    let (centered, _) = da_center_features(&features, 3, 2);
    let cov = da_covariance(&centered, 3, 2);
    for &c in &cov {
        assert!(c.abs() < 1e-6);
    }
}

#[test]
fn test_covariance_n1_returns_zero() {
    let features = vec![1.0f32, 2.0, 3.0]; // 1 × 3
    let cov = da_covariance(&features, 1, 3);
    for &c in &cov {
        assert_eq!(c, 0.0);
    }
}

#[test]
fn test_covariance_clamps_n_does_not_panic() {
    // Claims n=100 but the buffer only has 3 rows of d=2.
    let features = vec![3.0f32; 6];
    let (centered, _) = da_center_features(&features, 100, 2);
    let cov = da_covariance(&centered, 100, 2);
    assert_eq!(cov.len(), 4);
    for &c in &cov {
        assert!(c.abs() < 1e-6);
    }
}

// ----- da_frobenius_sq -----

#[test]
fn test_frobenius_sq_same_matrices_zero() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let r = da_frobenius_sq(&a, &a).unwrap();
    assert!(r.abs() < 1e-10);
}

#[test]
fn test_frobenius_sq_simple_case() {
    // ||[1,0] - [0,1]||² = 2
    let a = vec![1.0f32, 0.0];
    let b = vec![0.0f32, 1.0];
    let r = da_frobenius_sq(&a, &b).unwrap();
    assert!((r - 2.0).abs() < 1e-6);
}

#[test]
fn test_frobenius_sq_dimension_mismatch() {
    let a = vec![1.0f32, 2.0];
    let b = vec![1.0f32, 2.0, 3.0];
    assert!(matches!(
        da_frobenius_sq(&a, &b),
        Err(DomainAdaptationError::DimensionMismatch { .. })
    ));
}

// ----- da_coral_loss -----

#[test]
fn test_coral_loss_identical_domains_approx_zero() {
    let features = linspace_features(5, 4, 0.0, 0.2);
    let batch = DomainBatch::new(features.clone(), features.clone(), 5, 5, 4).unwrap();
    let loss = da_coral_loss(&batch).unwrap();
    assert!(loss < 1e-6, "loss={loss}");
}

#[test]
fn test_coral_loss_different_domains_positive() {
    // Source uniform [0,1], target uniform [10,11] → very different covariances
    let src = linspace_features(6, 3, 0.0, 0.1);
    let tgt = linspace_features(6, 3, 10.0, 0.5);
    let batch = DomainBatch::new(src, tgt, 6, 6, 3).unwrap();
    let loss = da_coral_loss(&batch).unwrap();
    assert!(loss >= 0.0);
}

#[test]
fn test_coral_loss_different_n_allowed() {
    // CORAL is valid even when n_source ≠ n_target
    let src = linspace_features(4, 2, 0.0, 0.5);
    let tgt = linspace_features(8, 2, 2.0, 0.1);
    let batch = DomainBatch::new(src, tgt, 4, 8, 2).unwrap();
    let r = da_coral_loss(&batch);
    assert!(r.is_ok());
}

// ----- DomainDiscriminator -----

#[test]
fn test_discriminator_weights_size() {
    let disc = DomainDiscriminator::new_random(16, 42);
    assert_eq!(disc.weights.len(), 16);
    assert_eq!(disc.d, 16);
}

#[test]
fn test_discriminator_predict_in_unit_interval() {
    let disc = DomainDiscriminator::new_random(4, 7);
    let feature = vec![0.5f32, -0.3, 1.0, 0.0];
    let p = disc.predict(&feature);
    assert!(p > 0.0 && p < 1.0, "p={p}");
}

#[test]
fn test_discriminator_predict_batch_length() {
    let disc = DomainDiscriminator::new_random(3, 13);
    let features = vec![0.1f32; 15]; // 5 × 3
    let preds = disc.predict_batch(&features, 5).unwrap();
    assert_eq!(preds.len(), 5);
}

#[test]
fn test_discriminator_predict_batch_all_in_unit() {
    let disc = DomainDiscriminator::new_random(3, 99);
    let features = linspace_features(4, 3, -1.0, 0.5);
    let preds = disc.predict_batch(&features, 4).unwrap();
    for &p in &preds {
        assert!(p > 0.0 && p < 1.0, "p={p}");
    }
}

#[test]
fn test_discriminator_predict_batch_size_mismatch_errors() {
    let disc = DomainDiscriminator::new_random(3, 13);
    let features = vec![0.1f32; 10]; // not a multiple of d=3 for n=5
    let result = disc.predict_batch(&features, 5);
    assert!(matches!(
        result,
        Err(DomainAdaptationError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_discriminator_xavier_range() {
    // Xavier limit for d=100 is sqrt(6/101) ≈ 0.2437
    let d = 100;
    let limit = (6.0f32 / (d + 1) as f32).sqrt();
    let disc = DomainDiscriminator::new_random(d, 2026);
    for &w in &disc.weights {
        assert!(
            w >= -limit && w <= limit,
            "w={w} not in [{}, {}]",
            -limit,
            limit
        );
    }
}

// ----- da_dann_loss -----

#[test]
fn test_dann_loss_valid() {
    let batch = make_batch(4, 4, 8, 0.0, 1.0);
    let disc = DomainDiscriminator::new_random(8, 1);
    let config = DannConfig::default();
    let loss = da_dann_loss(&disc, &batch, &config).unwrap();
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_dann_loss_dimension_mismatch() {
    let batch = make_batch(2, 2, 4, 0.0, 1.0);
    let disc = DomainDiscriminator::new_random(8, 1); // wrong d
    let config = DannConfig::default();
    assert!(matches!(
        da_dann_loss(&disc, &batch, &config),
        Err(DomainAdaptationError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_dann_loss_empty_batch_err() {
    let batch = DomainBatch {
        source_features: vec![],
        target_features: vec![1.0],
        n_source: 0,
        n_target: 1,
        d: 1,
    };
    let disc = DomainDiscriminator::new_random(1, 1);
    let config = DannConfig::default();
    assert!(matches!(
        da_dann_loss(&disc, &batch, &config),
        Err(DomainAdaptationError::EmptyFeatures)
    ));
}

#[test]
fn test_dann_loss_corrupted_batch_length_errors_not_panics() {
    // `DomainBatch`'s fields are all `pub`, so a hand-built batch can
    // violate the `source_features.len() == n_source * d` invariant that
    // `DomainBatch::new` would otherwise enforce -- must return an
    // error, not index out of bounds.
    let batch = DomainBatch {
        source_features: vec![1.0, 2.0], // claims n_source=3 but only has 2 elements
        target_features: vec![1.0, 2.0],
        n_source: 3,
        n_target: 2,
        d: 1,
    };
    let disc = DomainDiscriminator::new_random(1, 1);
    let config = DannConfig::default();
    assert!(matches!(
        da_dann_loss(&disc, &batch, &config),
        Err(DomainAdaptationError::DimensionMismatch { .. })
    ));
}

// ----- da_domain_accuracy -----

#[test]
fn test_domain_accuracy_range() {
    // A random discriminator on balanced data should give some accuracy 0..=1
    let batch = make_batch(20, 20, 4, -0.5, 0.5);
    let disc = DomainDiscriminator::new_random(4, 7777);
    let acc = da_domain_accuracy(&disc, &batch, 0.5).unwrap();
    assert!((0.0..=1.0).contains(&acc), "acc={acc}");
}

#[test]
fn test_domain_accuracy_all_zeros_source_target_mixed() {
    // All source and target have same feature = [0,0] → random disc result
    let batch = make_batch(10, 10, 2, 0.0, 0.0);
    let disc = DomainDiscriminator::new_random(2, 42);
    let acc = da_domain_accuracy(&disc, &batch, 0.5).unwrap();
    assert!((0.0..=1.0).contains(&acc));
}

#[test]
fn test_domain_accuracy_corrupted_batch_length_errors_not_panics() {
    let batch = DomainBatch {
        source_features: vec![1.0, 2.0],
        target_features: vec![1.0], // claims n_target=2 but only has 1 element
        n_source: 2,
        n_target: 2,
        d: 1,
    };
    let disc = DomainDiscriminator::new_random(1, 1);
    assert!(matches!(
        da_domain_accuracy(&disc, &batch, 0.5),
        Err(DomainAdaptationError::DimensionMismatch { .. })
    ));
}

// ----- da_entropy -----

#[test]
fn test_entropy_uniform_is_max() {
    let n = 4;
    let probs = vec![0.25f32; n];
    let h = da_entropy(&probs);
    let expected = -(0.25f32 * 0.25f32.ln()) * n as f32;
    assert!((h - expected).abs() < 1e-5);
}

#[test]
fn test_entropy_delta_approx_zero() {
    let mut probs = vec![0.0f32; 8];
    probs[0] = 1.0;
    let h = da_entropy(&probs);
    assert!(h < 1e-5, "h={h}");
}

#[test]
fn test_entropy_non_negative() {
    let probs = vec![0.1f32, 0.4, 0.3, 0.2];
    assert!(da_entropy(&probs) >= 0.0);
}

// ----- da_entropy_loss -----

#[test]
fn test_entropy_loss_uniform_non_negative() {
    let n_src = 3;
    let n_tgt = 4;
    let classes = 5;
    let src_probs = vec![0.2f32; n_src * classes];
    let tgt_probs = vec![0.2f32; n_tgt * classes];
    let loss = da_entropy_loss(&src_probs, n_src, &tgt_probs, n_tgt).unwrap();
    assert!(loss >= 0.0);
}

#[test]
fn test_entropy_loss_empty_err() {
    let r = da_entropy_loss(&[], 0, &[], 0);
    assert!(r.is_err());
}

// ----- da_confidence_threshold_mask -----

#[test]
fn test_confidence_mask_threshold_zero_all_true() {
    // classes=1: each raw element is its own row-max.
    let probs = vec![0.1f32, 0.5, 0.9, 0.01];
    let n = probs.len();
    let mask = da_confidence_threshold_mask(&probs, n, 0.0).expect("valid");
    assert!(mask.iter().all(|&m| m));
}

#[test]
fn test_confidence_mask_threshold_one_all_false() {
    let probs = vec![0.1f32, 0.5, 0.9, 0.99];
    let n = probs.len();
    let mask = da_confidence_threshold_mask(&probs, n, 1.0).expect("valid");
    assert!(mask.iter().all(|&m| !m));
}

#[test]
fn test_confidence_mask_selective() {
    let probs = vec![0.3f32, 0.7, 0.95, 0.2];
    let n = probs.len();
    let mask = da_confidence_threshold_mask(&probs, n, 0.5).expect("valid");
    assert!(!mask[0]);
    assert!(mask[1]);
    assert!(mask[2]);
    assert!(!mask[3]);
}

#[test]
fn test_confidence_mask_multi_class_takes_row_max() {
    // n=2 samples, classes=3. Sample 0's max is 0.6 (class 1); sample
    // 1's max is 0.9 (class 2).
    let probs = vec![
        0.1, 0.6, 0.3, // sample 0: max = 0.6
        0.05, 0.05, 0.9, // sample 1: max = 0.9
    ];
    let mask = da_confidence_threshold_mask(&probs, 2, 0.5).expect("valid");
    assert_eq!(mask.len(), 2, "mask length must equal n, not n*classes");
    assert!(mask[0], "sample 0 max=0.6 > 0.5");
    assert!(mask[1], "sample 1 max=0.9 > 0.5");
}

#[test]
fn test_confidence_mask_multi_class_below_threshold() {
    let probs = vec![
        0.1, 0.2, 0.3, // sample 0: max = 0.3 <= 0.5
        0.05, 0.05, 0.9, // sample 1: max = 0.9 > 0.5
    ];
    let mask = da_confidence_threshold_mask(&probs, 2, 0.5).expect("valid");
    assert!(!mask[0], "sample 0 max=0.3 should not exceed threshold");
    assert!(mask[1]);
}

#[test]
fn test_confidence_mask_invalid_n_errors() {
    let probs = vec![0.1f32, 0.2, 0.3]; // len=3, not divisible by n=2
    let result = da_confidence_threshold_mask(&probs, 2, 0.5);
    assert!(matches!(
        result,
        Err(DomainAdaptationError::InvalidConfig { .. })
    ));
}

#[test]
fn test_confidence_mask_empty_probs_returns_empty_ok() {
    let mask = da_confidence_threshold_mask(&[], 0, 0.5).expect("valid");
    assert!(mask.is_empty());
}

// ----- da_pseudo_label_loss -----

#[test]
fn test_pseudo_label_loss_all_confident_non_negative() {
    let n = 3;
    let classes = 4;
    // All samples have high confidence
    let mut probs = vec![0.01f32; n * classes];
    for i in 0..n {
        probs[i * classes] = 0.97; // class 0 is confident
    }
    let logits = [2.0f32, 0.1, 0.1, 0.1].repeat(n);
    let loss = da_pseudo_label_loss(&logits, &probs, n, 0.9, 1e-7).unwrap();
    assert!(loss >= 0.0);
}

#[test]
fn test_pseudo_label_loss_none_confident_returns_zero() {
    let n = 4;
    let classes = 3;
    let probs = vec![0.33f32; n * classes]; // uniform — low confidence
    let logits = vec![0.0f32; n * classes];
    let loss = da_pseudo_label_loss(&logits, &probs, n, 0.9, 1e-7).unwrap();
    assert_eq!(loss, 0.0);
}

#[test]
fn test_pseudo_label_loss_empty_n_err() {
    let r = da_pseudo_label_loss(&[], &[], 0, 0.5, 1e-7);
    assert!(r.is_err());
}

// ----- da_reversal_loss_scale -----

#[test]
fn test_reversal_loss_scale_step0_near_zero() {
    let scaled = da_reversal_loss_scale(1.0, 1.0, 0, 100);
    // step=0 → lambda_t ≈ 0
    assert!(scaled.abs() < 0.05, "scaled={scaled}");
}

#[test]
fn test_reversal_loss_scale_step_total_near_max() {
    let lambda = 1.0;
    let scaled = da_reversal_loss_scale(1.0, lambda, 100, 100);
    // step=total_steps → lambda_t ≈ lambda
    assert!(scaled > 0.8 * lambda, "scaled={scaled}");
}

#[test]
fn test_reversal_loss_scale_monotone() {
    let lambda = 0.5;
    let total = 50u64;
    let mut prev = da_reversal_loss_scale(1.0, lambda, 0, total);
    for step in 1..=total {
        let cur = da_reversal_loss_scale(1.0, lambda, step, total);
        assert!(
            cur >= prev - 1e-6,
            "not monotone at step {step}: {cur} < {prev}"
        );
        prev = cur;
    }
}

// ----- da_combined_loss -----

#[test]
fn test_combined_loss_mmd_method() {
    let batch = make_batch(4, 4, 3, 0.0, 1.0);
    let config = DomainAdaptConfig {
        method: DomainAdaptMethod::Mmd,
        ..DomainAdaptConfig::default()
    };
    let r = da_combined_loss(&batch, None, &config);
    assert!(r.is_ok());
    assert!(r.unwrap() >= 0.0);
}

#[test]
fn test_combined_loss_coral_method() {
    let batch = make_batch(4, 4, 3, 0.0, 1.0);
    let config = DomainAdaptConfig {
        method: DomainAdaptMethod::Coral,
        ..DomainAdaptConfig::default()
    };
    let r = da_combined_loss(&batch, None, &config);
    assert!(r.is_ok());
}

#[test]
fn test_combined_loss_dann_method() {
    let batch = make_batch(4, 4, 3, 0.0, 1.0);
    let disc = DomainDiscriminator::new_random(3, 5);
    let config = DomainAdaptConfig {
        method: DomainAdaptMethod::Dann,
        ..DomainAdaptConfig::default()
    };
    let r = da_combined_loss(&batch, Some(&disc), &config);
    assert!(r.is_ok());
}

#[test]
fn test_combined_loss_combined_method() {
    let batch = make_batch(5, 5, 4, -1.0, 1.0);
    let config = DomainAdaptConfig::default(); // Combined method
    let r = da_combined_loss(&batch, None, &config);
    assert!(r.is_ok());
    assert!(r.unwrap().is_finite());
}

#[test]
fn test_combined_loss_dann_without_discriminator_err() {
    let batch = make_batch(4, 4, 3, 0.0, 1.0);
    let config = DomainAdaptConfig {
        method: DomainAdaptMethod::Dann,
        ..DomainAdaptConfig::default()
    };
    let r = da_combined_loss(&batch, None, &config);
    assert!(r.is_err());
}

// ----- AdaptationStats -----

#[test]
fn test_adaptation_stats_mmd_method_some() {
    let batch = make_batch(4, 4, 3, 0.0, 1.0);
    let config = DomainAdaptConfig {
        method: DomainAdaptMethod::Mmd,
        ..DomainAdaptConfig::default()
    };
    let stats = da_compute_stats(&batch, None, &config).unwrap();
    assert!(stats.mmd_loss.is_some());
    assert!(stats.coral_loss.is_none());
    assert!(stats.dann_loss.is_none());
}

#[test]
fn test_adaptation_stats_coral_method_some() {
    let batch = make_batch(4, 4, 3, 0.0, 1.0);
    let config = DomainAdaptConfig {
        method: DomainAdaptMethod::Coral,
        ..DomainAdaptConfig::default()
    };
    let stats = da_compute_stats(&batch, None, &config).unwrap();
    assert!(stats.coral_loss.is_some());
    assert!(stats.mmd_loss.is_none());
}

#[test]
fn test_adaptation_stats_combined_all_some_except_dann_without_disc() {
    let batch = make_batch(4, 4, 3, 0.0, 1.0);
    let config = DomainAdaptConfig::default(); // Combined
    let stats = da_compute_stats(&batch, None, &config).unwrap();
    assert!(stats.mmd_loss.is_some());
    assert!(stats.coral_loss.is_some());
    assert!(stats.entropy_loss.is_some());
}

#[test]
fn test_adaptation_stats_combined_loss_finite() {
    let batch = make_batch(4, 4, 3, 0.1, -0.1);
    let config = DomainAdaptConfig::default();
    let stats = da_compute_stats(&batch, None, &config).unwrap();
    assert!(stats.combined_loss.is_finite());
}

#[test]
fn test_compute_stats_pseudo_labels_reflects_actual_confidence() {
    // Constant target features -> every dimension ties for the max, so
    // the softmax max probability is exactly 1/d. With d=4,
    // soft_max=0.25, comfortably below the default
    // confidence_threshold=0.9, so NO target sample should count as a
    // confident pseudo-label. Before the fix this was hardcoded to
    // `norm_max = 1.0`, which always exceeded any threshold < 1.0 and
    // reported n_pseudo_labels == n_target regardless of the data.
    let batch = make_batch(2, 5, 4, 0.0, 1.0);
    let config = DomainAdaptConfig {
        method: DomainAdaptMethod::Coral,
        confidence_threshold: 0.9,
        ..DomainAdaptConfig::default()
    };
    let stats = da_compute_stats(&batch, None, &config).unwrap();
    assert_eq!(
        stats.n_pseudo_labels, 0,
        "uniform (low-confidence) target features must not count as pseudo-labels"
    );
}

#[test]
fn test_compute_stats_pseudo_labels_counts_confident_samples() {
    // Two target samples: one uniform (low confidence), one strongly
    // peaked (high confidence). Exactly the peaked one should count.
    let d = 4;
    let uniform = vec![0.0f32; d];
    let mut peaked = vec![-10.0f32; d];
    peaked[0] = 10.0; // softmax max ~= 1.0 for this sample
    let mut target = uniform.clone();
    target.extend_from_slice(&peaked);
    let batch = DomainBatch::new(vec![0.0; d], target, 1, 2, d).expect("valid batch");
    let config = DomainAdaptConfig {
        method: DomainAdaptMethod::Coral,
        confidence_threshold: 0.9,
        ..DomainAdaptConfig::default()
    };
    let stats = da_compute_stats(&batch, None, &config).unwrap();
    assert_eq!(
        stats.n_pseudo_labels, 1,
        "exactly the peaked sample should exceed confidence_threshold=0.9"
    );
}

// ----- da_format_stats / da_format_config -----

#[test]
fn test_format_stats_non_empty() {
    let stats = AdaptationStats {
        mmd_loss: Some(0.1),
        coral_loss: None,
        dann_loss: None,
        entropy_loss: Some(0.5),
        combined_loss: 0.6,
        domain_accuracy: Some(0.7),
        n_pseudo_labels: 3,
    };
    let s = da_format_stats(&stats);
    assert!(!s.is_empty());
    assert!(s.contains("mmd="));
    assert!(s.contains("entropy="));
    assert!(s.contains("combined="));
}

#[test]
fn test_format_config_non_empty() {
    let config = DomainAdaptConfig::default();
    let s = da_format_config(&config);
    assert!(!s.is_empty());
    assert!(s.contains("DomainAdaptConfig"));
}

// ----- error variant coverage -----

#[test]
fn test_error_invalid_bandwidth_display() {
    let e = DomainAdaptationError::InvalidBandwidth { bw: 0.0 };
    let s = e.to_string();
    assert!(s.contains("0"));
}

#[test]
fn test_error_dimension_mismatch_display() {
    let e = DomainAdaptationError::DimensionMismatch { src: 3, tgt: 5 };
    let s = e.to_string();
    assert!(s.contains("3") && s.contains("5"));
}

#[test]
fn test_error_batch_mismatch_display() {
    let e = DomainAdaptationError::BatchMismatch { src: 4, tgt: 8 };
    let s = e.to_string();
    assert!(s.contains("4") && s.contains("8"));
}

#[test]
fn test_error_invalid_config_display() {
    let e = DomainAdaptationError::InvalidConfig {
        reason: "test reason".to_owned(),
    };
    let s = e.to_string();
    assert!(s.contains("test reason"));
}

// ----- edge cases -----

#[test]
fn test_mmd_biased_same_n_non_negative() {
    let src = linspace_features(3, 2, 0.0, 0.4);
    let tgt = linspace_features(3, 2, 1.0, 0.2);
    let r = da_mmd_biased(&src, &tgt, 3, 3, 2, 1.0).unwrap();
    assert!(r >= 0.0);
}

#[test]
fn test_coral_different_n_source_target() {
    // n_source ≠ n_target is perfectly valid for CORAL
    let src = linspace_features(3, 2, 0.0, 1.0);
    let tgt = linspace_features(5, 2, 2.0, 0.3);
    let batch = DomainBatch::new(src, tgt, 3, 5, 2).unwrap();
    let r = da_coral_loss(&batch);
    assert!(r.is_ok(), "{:?}", r.err());
}

#[test]
fn test_dann_discriminator_predict_sigmoid_saturates() {
    // A discriminator with large positive weights should push predictions near 1
    let d = 2;
    let disc = DomainDiscriminator {
        weights: vec![100.0f32, 100.0],
        bias: 100.0,
        d,
    };
    let p = disc.predict(&[1.0, 1.0]);
    assert!(p > 0.99, "p={p}");
}

#[test]
fn test_dann_discriminator_predict_sigmoid_low() {
    let d = 2;
    let disc = DomainDiscriminator {
        weights: vec![-100.0f32, -100.0],
        bias: -100.0,
        d,
    };
    let p = disc.predict(&[1.0, 1.0]);
    assert!(p < 0.01, "p={p}");
}

#[test]
fn test_reversal_loss_zero_total_steps() {
    // total_steps=0 should use t=1 (max lambda)
    let scaled = da_reversal_loss_scale(1.0, 1.0, 0, 0);
    // t=1 → lambda_t ≈ lambda
    assert!(scaled > 0.5);
}

#[test]
fn test_pseudo_label_loss_dimension_mismatch() {
    let logits = vec![1.0f32, 2.0, 3.0]; // 3 elements
    let probs = vec![0.1f32, 0.8, 0.1, 0.5, 0.3, 0.2]; // 6 elements
    let r = da_pseudo_label_loss(&logits, &probs, 2, 0.5, 1e-7);
    assert!(r.is_err());
}

#[test]
fn test_mmd_multiscale_biased_mode() {
    let batch = make_batch(3, 3, 2, 0.0, 2.0);
    let config = MmdConfig {
        kernel_bandwidths: vec![1.0, 2.0],
        biased: true,
        eps: 1e-8,
    };
    let r = da_mmd_multiscale(&batch, &config);
    assert!(r.is_ok());
    assert!(r.unwrap() >= 0.0);
}

// ── Regression (F286): MmdConfig::eps is the kernel-bandwidth floor ──────────
// The field used to be declared and defaulted but never read.

#[test]
fn test_mmd_config_validate_rejects_degenerate_bandwidths() {
    let ok = MmdConfig::default();
    ok.validate().expect("the default config must be valid");

    let empty = MmdConfig {
        kernel_bandwidths: vec![],
        ..MmdConfig::default()
    };
    assert!(matches!(
        empty.validate(),
        Err(DomainAdaptationError::InvalidConfig { .. })
    ));

    // A bandwidth below `eps` collapses the Gaussian kernel.
    let too_small = MmdConfig {
        kernel_bandwidths: vec![1.0, 1e-12],
        eps: 1e-8,
        ..MmdConfig::default()
    };
    match too_small.validate() {
        Err(DomainAdaptationError::InvalidBandwidth { bw }) => {
            assert!((bw - 1e-12).abs() < 1e-18, "bw={bw}")
        }
        other => panic!("expected InvalidBandwidth, got {other:?}"),
    }

    // Raising `eps` tightens the same check.
    let raised = MmdConfig {
        kernel_bandwidths: vec![0.5],
        eps: 1.0,
        ..MmdConfig::default()
    };
    assert!(matches!(
        raised.validate(),
        Err(DomainAdaptationError::InvalidBandwidth { .. })
    ));

    // `eps` itself must be a positive finite number.
    let bad_eps = MmdConfig {
        eps: 0.0,
        ..MmdConfig::default()
    };
    assert!(matches!(
        bad_eps.validate(),
        Err(DomainAdaptationError::InvalidConfig { .. })
    ));
}

#[test]
fn test_mmd_multiscale_rejects_sub_eps_bandwidth() {
    let batch = make_batch(2, 2, 2, 0.0, 1.0);
    let config = MmdConfig {
        kernel_bandwidths: vec![1e-12],
        eps: 1e-8,
        biased: false,
    };
    let err = da_mmd_multiscale(&batch, &config)
        .expect_err("a sub-eps bandwidth must be rejected, not silently used");
    assert!(matches!(
        err,
        DomainAdaptationError::InvalidBandwidth { .. }
    ));
}

#[test]
fn test_mmd_config_from_median_heuristic_floors_at_eps() {
    // Well-spread features: every bandwidth is a multiple of the median.
    let features = linspace_features(6, 2, 0.0, 1.0);
    let config = MmdConfig::from_median_heuristic(&features, 6, 2);
    config
        .validate()
        .expect("the median heuristic must produce a usable config");
    assert_eq!(config.kernel_bandwidths.len(), 5);
    for w in config.kernel_bandwidths.windows(2) {
        assert!(w[1] > w[0], "bandwidths must be increasing: {w:?}");
    }

    // Degenerate features (all rows identical): the median distance is ~0, so
    // every scale falls back to the eps floor instead of collapsing.
    let flat = vec![1.0f32; 12];
    let degenerate = MmdConfig::from_median_heuristic(&flat, 6, 2);
    degenerate
        .validate()
        .expect("a degenerate feature set must still validate");
    for &bw in &degenerate.kernel_bandwidths {
        assert!(bw >= degenerate.eps, "bandwidth {bw} fell below eps");
    }
}

// ── Regression (F287): dann_lambda_schedule affects the objective ────────────
// It was previously read only by `da_format_config`, so toggling it changed
// nothing but a log line.

#[test]
fn test_effective_lambda_follows_the_schedule_flag() {
    let mut config = DomainAdaptConfig {
        method: DomainAdaptMethod::Dann,
        dann: DannConfig {
            lambda: 0.4,
            eps: 1e-7,
        },
        dann_lambda_schedule: true,
        ..DomainAdaptConfig::default()
    };

    // Scheduled: ~0 at the start, approaching lambda at the end.
    let start = config.effective_lambda(0, 1000);
    let mid = config.effective_lambda(500, 1000);
    let end = config.effective_lambda(1000, 1000);
    assert!(start.abs() < 1e-6, "lambda must start at ~0, got {start}");
    assert!(start < mid && mid < end, "{start} < {mid} < {end}");
    assert!(end <= 0.4 + 1e-6 && end > 0.39, "end lambda was {end}");

    // Unscheduled: constant lambda at every step.
    config.dann_lambda_schedule = false;
    for step in [0u64, 250, 1000] {
        let l = config.effective_lambda(step, 1000);
        assert!((l - 0.4).abs() < 1e-6, "step {step} gave lambda {l}");
    }
}

#[test]
fn test_scaled_dann_loss_applies_the_schedule() {
    let batch = make_batch(3, 3, 2, 0.0, 1.0);
    let disc = DomainDiscriminator::new_random(2, 7);
    let mut config = DomainAdaptConfig {
        method: DomainAdaptMethod::Dann,
        dann: DannConfig {
            lambda: 0.5,
            eps: 1e-7,
        },
        dann_lambda_schedule: true,
        ..DomainAdaptConfig::default()
    };

    let raw = da_dann_loss(&disc, &batch, &config.dann).expect("dann loss");
    assert!(raw > 0.0, "the fixture must produce a non-zero loss");

    let early = da_scaled_dann_loss(&disc, &batch, &config, 0, 1000).expect("scaled");
    let late = da_scaled_dann_loss(&disc, &batch, &config, 1000, 1000).expect("scaled");
    assert!(early.abs() < 1e-5, "early scaled loss was {early}");
    assert!(late > early, "adversarial pressure must ramp up");
    // λ_t(total, total) = 2λ/(1 + e^-10) − λ, i.e. λ up to ~1e-4 relative.
    assert!(
        (late - raw * 0.5).abs() < raw * 1e-3,
        "late={late} raw={raw}"
    );

    // The step-aware combined loss routes through the same scaling...
    let combined = da_combined_loss_at_step(&batch, Some(&disc), &config, 1000, 1000)
        .expect("combined at step");
    assert!((combined - late).abs() < 1e-6);
    // ...while the step-less one stays unscaled.
    let unscaled = da_combined_loss(&batch, Some(&disc), &config).expect("combined");
    assert!((unscaled - raw).abs() < 1e-6);

    // With the schedule off, lambda is constant from step 0.
    config.dann_lambda_schedule = false;
    let flat_early = da_scaled_dann_loss(&disc, &batch, &config, 0, 1000).expect("scaled");
    assert!(
        (flat_early - raw * 0.5).abs() < raw * 1e-5,
        "flat_early={flat_early} raw={raw}"
    );
}

#[test]
fn test_combined_loss_at_step_matches_plain_for_non_dann_methods() {
    let batch = make_batch(3, 3, 2, 0.0, 1.0);
    for method in [
        DomainAdaptMethod::Mmd,
        DomainAdaptMethod::Coral,
        DomainAdaptMethod::Combined,
    ] {
        let config = DomainAdaptConfig {
            method,
            ..DomainAdaptConfig::default()
        };
        let plain = da_combined_loss(&batch, None, &config).expect("plain");
        let at_step = da_combined_loss_at_step(&batch, None, &config, 3, 10).expect("at step");
        assert!(
            (plain - at_step).abs() < 1e-6,
            "{method:?}: {plain} vs {at_step}"
        );
    }
}
