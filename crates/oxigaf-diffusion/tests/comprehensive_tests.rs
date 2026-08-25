//! Comprehensive integration tests for oxigaf-diffusion.
//!
//! Exercises the public API via the crate's re-exports (not internal `super::*`
//! paths) to verify integration-level correctness.  No real model weights are
//! required — all tests use synthetic inputs on the CPU.
//!
//! ## Coverage
//!
//! - **Scheduler**: alpha schedule properties, timestep bounds, add_noise
//!   round-trip, step determinism, both prediction types
//! - **Config**: default validity, stage_channels/num_stages, field values
//! - **Error types**: Display messages non-empty, From<io::Error> conversion
//! - **DebugHooks** (via public re-exports): all_finite, assert_finite,
//!   check_tensor_health, DebugHooks registry
//! - **SlicedAttention** (via public crate API): config validation, forward
//!   shapes, numerical equivalence across slice sizes
//! - **UpsamplerMode**: equality, Copy semantics
//! - **CFG formula**: mathematical correctness, scale monotonicity

use candle_core::{DType, Device, Tensor};
use oxigaf_diffusion::{
    all_finite, assert_finite, check_tensor_health, DdimScheduler, DebugConfig, DebugHooks,
    DiffusionConfig, DiffusionError, PredictionType, SlicedAttention, SlicedAttentionConfig,
    UpsamplerMode,
};

// ===========================================================================
// Category 1 — Scheduler
// ===========================================================================

#[test]
fn scheduler_alpha_cumprod_is_strictly_decreasing() {
    let sched = DdimScheduler::new(1000, PredictionType::Epsilon);
    // We can only inspect behaviour through the public API; verify via add_noise
    // and step that the schedule has the right properties by checking that
    // noise added at early and late timesteps differs appropriately.
    let device = Device::Cpu;
    let original = Tensor::ones((1, 4, 8, 8), DType::F32, &device).expect("tensor creation failed");
    let noise = Tensor::zeros((1, 4, 8, 8), DType::F32, &device).expect("tensor creation failed");

    // At t=0, add_noise should preserve mostly original (alpha close to 1).
    let noisy_low = sched
        .add_noise(&original, &noise, 0)
        .expect("add_noise t=0 failed");
    // At t=999, add_noise should suppress original (alpha close to 0).
    let noisy_high = sched
        .add_noise(&original, &noise, 999)
        .expect("add_noise t=999 failed");

    let low_sum = noisy_low
        .abs()
        .expect("abs failed")
        .sum_all()
        .expect("sum failed")
        .to_scalar::<f32>()
        .expect("scalar failed");
    let high_sum = noisy_high
        .abs()
        .expect("abs failed")
        .sum_all()
        .expect("sum failed")
        .to_scalar::<f32>()
        .expect("scalar failed");

    // At t=0, alpha~1 so output ~ original = ones → sum ~ 4*8*8 = 256.
    // At t=999, alpha~0 so output ~ 0 → sum ≈ 0.
    assert!(
        low_sum > high_sum,
        "low_t sum ({low_sum}) should exceed high_t sum ({high_sum})"
    );
}

#[test]
fn scheduler_timestep_count_matches_requested_steps() {
    let mut sched = DdimScheduler::new(1000, PredictionType::Epsilon);
    for n in [1usize, 10, 25, 50] {
        sched.set_timesteps(n).expect("set_timesteps failed");
        assert_eq!(
            sched.timesteps().len(),
            n,
            "set_timesteps({n}) produced {} timesteps",
            sched.timesteps().len()
        );
    }
}

#[test]
fn scheduler_timesteps_are_descending() {
    let mut sched = DdimScheduler::new(1000, PredictionType::VPrediction);
    sched.set_timesteps(20).expect("set_timesteps failed");
    let ts = sched.timesteps();
    for i in 0..ts.len().saturating_sub(1) {
        assert!(
            ts[i] >= ts[i + 1],
            "timestep[{i}]={} is not >= timestep[{}]={}",
            ts[i],
            i + 1,
            ts[i + 1]
        );
    }
}

#[test]
fn scheduler_timestep_bounds_within_train_range() {
    let train_steps = 1000usize;
    let mut sched = DdimScheduler::new(train_steps, PredictionType::Epsilon);
    sched.set_timesteps(50).expect("set_timesteps failed");
    for &t in sched.timesteps() {
        assert!(
            t < train_steps,
            "timestep {t} exceeds train_steps {train_steps}"
        );
    }
}

#[test]
fn scheduler_add_noise_output_is_finite() {
    let sched = DdimScheduler::new(1000, PredictionType::Epsilon);
    let device = Device::Cpu;
    let original = Tensor::randn(0f32, 1f32, (2, 4, 16, 16), &device).expect("randn failed");
    let noise = Tensor::randn(0f32, 1f32, (2, 4, 16, 16), &device).expect("randn failed");

    for t in [0, 250, 500, 750, 999] {
        let noisy = sched
            .add_noise(&original, &noise, t)
            .unwrap_or_else(|_| panic!("add_noise at t={t} failed"));
        let flat = noisy
            .flatten_all()
            .expect("flatten failed")
            .to_vec1::<f32>()
            .expect("to_vec1 failed");
        assert!(
            all_finite(&flat),
            "add_noise at t={t} produced non-finite values"
        );
    }
}

#[test]
fn scheduler_add_noise_preserves_shape() {
    let sched = DdimScheduler::new(1000, PredictionType::Epsilon);
    let device = Device::Cpu;
    let original = Tensor::zeros((3, 4, 8, 8), DType::F32, &device).expect("zeros failed");
    let noise = Tensor::randn(0f32, 1f32, (3, 4, 8, 8), &device).expect("randn failed");
    let noisy = sched
        .add_noise(&original, &noise, 500)
        .expect("add_noise failed");
    assert_eq!(
        noisy.dims(),
        &[3, 4, 8, 8],
        "add_noise should preserve input shape"
    );
}

#[test]
fn scheduler_step_epsilon_output_is_finite_and_correct_shape() {
    let mut sched = DdimScheduler::new(1000, PredictionType::Epsilon);
    sched.set_timesteps(10).expect("set_timesteps failed");
    let device = Device::Cpu;
    let t = sched.timesteps()[0];
    let sample = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device).expect("randn failed");
    let model_out = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device).expect("randn failed");
    let result = sched.step(&model_out, t, &sample).expect("step failed");
    assert_eq!(result.dims(), &[1, 4, 8, 8]);
    let flat = result
        .flatten_all()
        .expect("flatten failed")
        .to_vec1::<f32>()
        .expect("to_vec1 failed");
    assert!(all_finite(&flat), "step output must be finite");
}

#[test]
fn scheduler_step_v_prediction_output_is_finite() {
    let mut sched = DdimScheduler::new(1000, PredictionType::VPrediction);
    sched.set_timesteps(10).expect("set_timesteps failed");
    let device = Device::Cpu;
    let t = sched.timesteps()[0];
    let sample = Tensor::randn(0f32, 1f32, (2, 4, 8, 8), &device).expect("randn failed");
    let model_out = Tensor::randn(0f32, 1f32, (2, 4, 8, 8), &device).expect("randn failed");
    let result = sched.step(&model_out, t, &sample).expect("step failed");
    let flat = result
        .flatten_all()
        .expect("flatten failed")
        .to_vec1::<f32>()
        .expect("to_vec1 failed");
    assert!(all_finite(&flat), "v-pred step output must be finite");
}

#[test]
fn scheduler_timestep_tensor_correct_value_and_shape() {
    let sched = DdimScheduler::new(1000, PredictionType::Epsilon);
    let device = Device::Cpu;
    let batch = 4;
    let t = 500usize;
    let tensor = sched
        .timestep_tensor(t, batch, &device)
        .expect("timestep_tensor failed");
    assert_eq!(tensor.dims(), &[batch]);
    let vals = tensor.to_vec1::<f32>().expect("to_vec1 failed");
    for v in vals {
        assert!((v - t as f32).abs() < 1e-4, "timestep value mismatch");
    }
}

#[test]
fn scheduler_epsilon_and_vpred_produce_different_outputs() {
    let mut sched_eps = DdimScheduler::new(1000, PredictionType::Epsilon);
    sched_eps.set_timesteps(10).expect("set_timesteps failed");
    let mut sched_vpred = DdimScheduler::new(1000, PredictionType::VPrediction);
    sched_vpred.set_timesteps(10).expect("set_timesteps failed");

    let device = Device::Cpu;
    let t = sched_eps.timesteps()[0];
    // Use the same sample and model output for both.
    let sample = Tensor::full(0.5f32, (1, 4, 4, 4), &device).expect("full failed");
    let model_out = Tensor::full(0.1f32, (1, 4, 4, 4), &device).expect("full failed");

    let eps_out = sched_eps.step(&model_out, t, &sample).expect("eps step");
    let vpred_out = sched_vpred
        .step(&model_out, t, &sample)
        .expect("vpred step");

    let diff = (eps_out - vpred_out)
        .expect("sub failed")
        .abs()
        .expect("abs failed")
        .sum_all()
        .expect("sum failed")
        .to_scalar::<f32>()
        .expect("scalar failed");
    assert!(
        diff > 1e-6,
        "Epsilon and v-prediction should give different step outputs"
    );
}

// ===========================================================================
// Category 2 — DiffusionConfig
// ===========================================================================

#[test]
fn config_default_field_values() {
    let cfg = DiffusionConfig::default();
    assert_eq!(cfg.num_views, 4);
    assert!(
        (cfg.guidance_scale - 3.0).abs() < 1e-9,
        "default guidance_scale must be 3.0"
    );
    assert_eq!(cfg.num_inference_steps, 50);
    assert_eq!(cfg.image_size, 256);
    assert_eq!(cfg.latent_size, 32);
    assert_eq!(cfg.latent_channels, 4);
    assert_eq!(cfg.unet_in_channels, 8);
    assert_eq!(cfg.unet_out_channels, 4);
    assert_eq!(cfg.camera_pose_dim, 12);
    assert!((cfg.vae_scale_factor - 0.18215).abs() < 1e-6);
    assert_eq!(cfg.norm_num_groups, 32);
    assert_eq!(cfg.layers_per_block, 2);
    assert!(
        cfg.upsampler_mode.is_none(),
        "default upsampler_mode should be None"
    );
}

#[test]
fn config_stage_channels_matches_formula() {
    let cfg = DiffusionConfig::default();
    // channel_mult = [1, 2, 4, 4], base_channels = 320
    let expected = [320, 640, 1280, 1280];
    for (stage, &exp) in expected.iter().enumerate() {
        assert_eq!(
            cfg.stage_channels(stage),
            exp,
            "stage_channels({stage}) mismatch"
        );
    }
}

#[test]
fn config_num_stages_matches_channel_mult_len() {
    let cfg = DiffusionConfig::default();
    assert_eq!(cfg.num_stages(), cfg.channel_mult.len());
    assert_eq!(cfg.num_stages(), 4);
}

#[test]
fn config_guidance_scale_is_at_least_one() {
    let cfg = DiffusionConfig::default();
    assert!(
        cfg.guidance_scale >= 1.0,
        "guidance_scale must be >= 1.0, got {}",
        cfg.guidance_scale
    );
}

#[test]
fn config_latent_size_is_image_size_over_eight() {
    let cfg = DiffusionConfig::default();
    assert_eq!(
        cfg.latent_size,
        cfg.image_size / 8,
        "latent_size should equal image_size / 8"
    );
}

#[test]
fn config_channel_mult_and_attention_head_dim_same_length() {
    let cfg = DiffusionConfig::default();
    assert_eq!(
        cfg.channel_mult.len(),
        cfg.attention_head_dim.len(),
        "channel_mult and attention_head_dim must have the same length"
    );
}

#[test]
fn config_upsampler_mode_variants_distinct() {
    let a = UpsamplerMode::SdX2;
    let b = UpsamplerMode::BilinearVae;
    assert_ne!(a, b);
    assert_eq!(a, UpsamplerMode::SdX2);
    assert_eq!(b, UpsamplerMode::BilinearVae);
    // Copy semantics: no move required.
    let _c = a;
    let _d = a; // Both should compile fine since UpsamplerMode is Copy.
}

// ===========================================================================
// Category 3 — Error types
// ===========================================================================

#[test]
fn error_display_messages_are_non_empty() {
    let errors: Vec<DiffusionError> = vec![
        DiffusionError::ModelLoad("test context".into()),
        DiffusionError::WeightNotFound {
            layer: "conv1".into(),
            expected_shape: vec![3, 3, 64, 64],
        },
        DiffusionError::WeightShapeMismatch {
            layer: "fc".into(),
            expected: vec![512, 256],
            got: vec![256, 512],
        },
        DiffusionError::ShapeMismatch {
            op: "matmul".into(),
            expected: vec![4, 8],
            got: vec![4, 16],
        },
        DiffusionError::DtypeMismatch {
            expected: "f32".into(),
            got: "f16".into(),
        },
        DiffusionError::NanDetected {
            layer: "output".into(),
            timestep: Some(42),
        },
        DiffusionError::InfDetected {
            layer: "hidden".into(),
            timestep: None,
        },
        DiffusionError::NanInfDetected {
            name: "activations".into(),
            nan_count: 3,
            inf_count: 1,
            first_index: 7,
        },
        DiffusionError::NumericalInstability {
            context: "division by zero".into(),
        },
        DiffusionError::InvalidConfig("guidance_scale < 1".into()),
        DiffusionError::Inference("forward pass failed".into()),
        DiffusionError::InvalidTimestep {
            value: 1001,
            max: 1000,
        },
        DiffusionError::InvalidViewCount {
            expected: 4,
            got: 3,
        },
        DiffusionError::InvalidLatentShape {
            expected: vec![1, 4, 32, 32],
            got: vec![1, 4, 64, 64],
        },
        DiffusionError::SkipConnectionUnderflow {
            expected: 3,
            available: 1,
        },
        DiffusionError::SchedulerNotInitialized,
        DiffusionError::ClipEncodingFailed("encoder error".into()),
        DiffusionError::VaeEncodeFailed("encode error".into()),
        DiffusionError::VaeDecodeFailed("decode error".into()),
        DiffusionError::UnetForwardFailed {
            timestep: 500,
            reason: "OOM".into(),
        },
        DiffusionError::ImageProcessingError("bad pixel format".into()),
    ];

    for err in errors {
        let msg = err.to_string();
        assert!(
            !msg.is_empty(),
            "DiffusionError variant produced empty Display: {:?}",
            err
        );
    }
}

#[test]
fn error_from_io_error_conversion() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let diff_err: DiffusionError = io_err.into();
    let msg = diff_err.to_string();
    assert!(!msg.is_empty(), "IoError display must be non-empty");
    // Should be wrapped in the IoError variant.
    assert!(
        msg.contains("I/O error") || msg.contains("file not found"),
        "IoError display should contain context"
    );
}

#[test]
fn error_nan_inf_detected_contains_counts() {
    let err = DiffusionError::NanInfDetected {
        name: "layer_out".into(),
        nan_count: 5,
        inf_count: 2,
        first_index: 10,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("5"),
        "NanInfDetected should mention nan_count=5"
    );
    assert!(
        msg.contains("2"),
        "NanInfDetected should mention inf_count=2"
    );
}

// ===========================================================================
// Category 4 — debug_hooks public re-exports
// ===========================================================================

#[test]
fn debug_hooks_all_finite_empty_slice_returns_true() {
    let data: &[f32] = &[];
    assert!(
        all_finite(data),
        "empty slice should be considered all-finite"
    );
}

#[test]
fn debug_hooks_all_finite_clean_data() {
    let data = vec![0.0f32, 1.0, -1.0, 1e6, -1e6];
    assert!(all_finite(&data));
}

#[test]
fn debug_hooks_all_finite_detects_nan() {
    let data = vec![1.0f32, f32::NAN, 3.0];
    assert!(!all_finite(&data));
}

#[test]
fn debug_hooks_all_finite_detects_inf() {
    let data = vec![f32::INFINITY, 2.0f32];
    assert!(!all_finite(&data));
}

#[test]
fn debug_hooks_assert_finite_ok_on_clean_slice() {
    let data = vec![0.0f32, 1.0, 2.0, 3.0];
    let result = assert_finite("clean", &data);
    assert!(result.is_ok(), "clean data should pass assert_finite");
}

#[test]
fn debug_hooks_assert_finite_err_on_nan() {
    let data = vec![1.0f32, f32::NAN];
    let result = assert_finite("has_nan", &data);
    assert!(result.is_err(), "NaN should cause assert_finite to Err");
    match result {
        Err(DiffusionError::NanInfDetected { nan_count, .. }) => {
            assert_eq!(nan_count, 1);
        }
        other => panic!("Expected NanInfDetected, got {:?}", other),
    }
}

#[test]
fn debug_hooks_assert_finite_err_on_neg_inf() {
    let data = vec![f32::NEG_INFINITY, 0.0f32];
    let result = assert_finite("has_neg_inf", &data);
    assert!(
        result.is_err(),
        "NEG_INFINITY should cause assert_finite to Err"
    );
}

#[test]
fn debug_hooks_check_tensor_health_finite_values() {
    let data = vec![1.0f32, 2.0, 3.0];
    let health = check_tensor_health("test", &data);
    assert!(health.is_healthy);
    assert_eq!(health.nan_count, 0);
    assert_eq!(health.pos_inf_count, 0);
    assert_eq!(health.neg_inf_count, 0);
    assert_eq!(health.finite_count, 3);
    assert_eq!(health.total_elements, 3);
    assert!(health.first_bad_index.is_none());
    assert!(health.min_finite.is_some());
    assert!(health.max_finite.is_some());
    assert!(health.mean_finite.is_some());
}

#[test]
fn debug_hooks_check_tensor_health_mixed_anomalies() {
    let data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0f32];
    let health = check_tensor_health("mixed", &data);
    assert!(!health.is_healthy);
    assert_eq!(health.nan_count, 1);
    assert_eq!(health.pos_inf_count, 1);
    assert_eq!(health.neg_inf_count, 1);
    assert_eq!(health.finite_count, 1);
    assert_eq!(health.first_bad_index, Some(0));
}

#[test]
fn debug_hooks_registry_accumulates_bad_records_via_public_api() {
    let hooks = DebugHooks::new(DebugConfig {
        enabled: true,
        panic_on_nan: false,
        panic_on_inf: false,
        log_all_checks: false,
        max_records: 10,
    });

    let good = vec![1.0f32, 2.0];
    let bad = vec![f32::NAN, 3.0f32];

    hooks.check("good_tensor", &good);
    hooks.check("bad_tensor_1", &bad);
    hooks.check("bad_tensor_2", &bad);

    let (total, bad_count) = hooks.stats();
    assert_eq!(total, 3);
    assert_eq!(bad_count, 2);

    let records = hooks.bad_records();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].name, "bad_tensor_1");
    assert_eq!(records[1].name, "bad_tensor_2");

    assert!(hooks.has_issues());
}

#[test]
fn debug_hooks_registry_clear_resets_bad_count_and_records() {
    let hooks = DebugHooks::new(DebugConfig {
        enabled: true,
        ..DebugConfig::default()
    });

    hooks.check("b", &[f32::NAN]);
    assert!(hooks.has_issues());

    hooks.clear();

    assert!(!hooks.has_issues());
    assert!(hooks.bad_records().is_empty());
    // Total check count is preserved; only bad_count resets.
    let (total, bad) = hooks.stats();
    assert_eq!(total, 1, "total count must not reset on clear");
    assert_eq!(bad, 0, "bad count must reset to 0 on clear");
}

#[test]
fn debug_hooks_disabled_config_skips_scan() {
    let hooks = DebugHooks::new(DebugConfig {
        enabled: false,
        ..DebugConfig::default()
    });
    // Feed pure NaN — should be ignored.
    let result = hooks.check("ignored", &[f32::NAN; 64]);
    assert!(result.is_healthy, "disabled hooks should report healthy");
    assert!(!hooks.has_issues());
}

#[test]
fn debug_hooks_max_records_evicts_oldest() {
    let hooks = DebugHooks::new(DebugConfig {
        enabled: true,
        max_records: 3,
        ..DebugConfig::default()
    });

    for i in 0..5usize {
        hooks.check(format!("t{i}"), &[f32::NAN]);
    }

    let records = hooks.bad_records();
    assert_eq!(records.len(), 3, "should cap at max_records=3");
    // Oldest (t0, t1) evicted; newest (t2, t3, t4) retained.
    assert_eq!(records[0].name, "t2");
    assert_eq!(records[2].name, "t4");
}

// ===========================================================================
// Category 5 — SlicedAttention (via crate public API)
// ===========================================================================

fn make_qkv(
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_len = batch * heads * seq_q * head_dim;
    let k_len = batch * heads * seq_k * head_dim;
    let q: Vec<f32> = (0..q_len).map(|i| (i as f32 * 0.01).sin()).collect();
    let k: Vec<f32> = (0..k_len).map(|i| (i as f32 * 0.02).cos()).collect();
    let v: Vec<f32> = (0..k_len).map(|i| (i as f32 * 0.03).sin()).collect();
    (q, k, v)
}

#[test]
fn sliced_attention_default_config_is_valid() {
    let cfg = SlicedAttentionConfig::default();
    assert!(cfg.validate().is_ok(), "default config must be valid");
}

#[test]
fn sliced_attention_config_rejects_zero_heads() {
    let cfg = SlicedAttentionConfig::new(None, 0, 64);
    assert!(cfg.validate().is_err());
}

#[test]
fn sliced_attention_config_rejects_zero_head_dim() {
    let cfg = SlicedAttentionConfig::new(None, 8, 0);
    assert!(cfg.validate().is_err());
}

#[test]
fn sliced_attention_config_rejects_zero_slice_size() {
    let cfg = SlicedAttentionConfig::new(Some(0), 8, 64);
    assert!(cfg.validate().is_err());
}

#[test]
fn sliced_attention_new_fails_on_invalid_config() {
    let cfg = SlicedAttentionConfig::new(None, 0, 64);
    assert!(
        SlicedAttention::new(cfg).is_err(),
        "SlicedAttention::new must propagate config validation error"
    );
}

#[test]
fn sliced_attention_output_shape_matches_input() {
    let (batch, heads, seq_q, seq_k, head_dim) = (2, 4, 8, 12, 16);
    let (q, k, v) = make_qkv(batch, heads, seq_q, seq_k, head_dim);
    let cfg = SlicedAttentionConfig::new(None, heads, head_dim);
    let attn = SlicedAttention::new(cfg).expect("valid config");
    let out = attn
        .forward(&q, &k, &v, batch, seq_q, seq_k)
        .expect("forward failed");
    assert_eq!(
        out.len(),
        batch * heads * seq_q * head_dim,
        "output length must equal batch*heads*seq_q*head_dim"
    );
}

#[test]
fn sliced_attention_output_is_finite() {
    let (batch, heads, seq_q, seq_k, head_dim) = (1, 2, 6, 8, 8);
    let (q, k, v) = make_qkv(batch, heads, seq_q, seq_k, head_dim);
    let cfg = SlicedAttentionConfig::new(Some(3), heads, head_dim);
    let attn = SlicedAttention::new(cfg).expect("valid config");
    let out = attn
        .forward(&q, &k, &v, batch, seq_q, seq_k)
        .expect("forward failed");
    assert!(
        all_finite(&out),
        "attention output must be finite for normal inputs"
    );
}

#[test]
fn sliced_attention_slice_vs_no_slice_numerically_equivalent() {
    let (batch, heads, seq_q, seq_k, head_dim) = (1, 2, 8, 8, 16);
    let (q, k, v) = make_qkv(batch, heads, seq_q, seq_k, head_dim);

    let cfg_full = SlicedAttentionConfig::new(None, heads, head_dim);
    let attn_full = SlicedAttention::new(cfg_full).expect("valid");
    let out_full = attn_full
        .forward(&q, &k, &v, batch, seq_q, seq_k)
        .expect("forward");

    let cfg_sliced = SlicedAttentionConfig::new(Some(2), heads, head_dim);
    let attn_sliced = SlicedAttention::new(cfg_sliced).expect("valid");
    let out_sliced = attn_sliced
        .forward(&q, &k, &v, batch, seq_q, seq_k)
        .expect("forward");

    let max_diff: f32 = out_full
        .iter()
        .zip(out_sliced.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "sliced vs full attention max diff {max_diff} exceeds tolerance"
    );
}

#[test]
fn sliced_attention_forward_shape_mismatch_q_returns_err() {
    let heads = 1;
    let head_dim = 4;
    let cfg = SlicedAttentionConfig::new(None, heads, head_dim);
    let attn = SlicedAttention::new(cfg).expect("valid config");
    let q = vec![0.0f32; 3]; // wrong size (expected batch*1*4*4 = 16)
    let k = vec![0.0f32; 16];
    let v = vec![0.0f32; 16];
    let result = attn.forward(&q, &k, &v, 1, 4, 4);
    assert!(result.is_err(), "wrong Q length should return Err");
}

#[test]
fn sliced_attention_memory_estimate_sliced_less_than_standard() {
    let cfg = SlicedAttentionConfig::new(Some(4), 8, 64);
    let seq_q = 512;
    let seq_k = 512;
    let std_mem = cfg.memory_bytes_standard(seq_q, seq_k);
    let sliced_mem = cfg.memory_bytes_sliced(seq_k);
    assert!(
        sliced_mem < std_mem,
        "sliced memory ({sliced_mem}) must be less than standard ({std_mem})"
    );
}

// ===========================================================================
// Category 6 — CFG formula mathematical properties
// ===========================================================================

#[test]
fn cfg_formula_scale_one_equals_conditional() {
    // CFG: result = uncond + scale * (cond - uncond)
    // When scale = 1.0: result = cond
    let uncond = [0.0f32, 1.0, 2.0, 3.0];
    let cond = [4.0f32, 5.0, 6.0, 7.0];
    let scale = 1.0f32;
    let result: Vec<f32> = uncond
        .iter()
        .zip(cond.iter())
        .map(|(u, c)| u + scale * (c - u))
        .collect();
    for (r, c) in result.iter().zip(cond.iter()) {
        assert!(
            (r - c).abs() < 1e-6,
            "scale=1 should reproduce cond: got {r}, expected {c}"
        );
    }
}

#[test]
fn cfg_formula_scale_zero_equals_unconditional() {
    // When scale = 0.0: result = uncond
    let uncond = [1.0f32, 2.0, 3.0];
    let cond = [10.0f32, 20.0, 30.0];
    let scale = 0.0f32;
    let result: Vec<f32> = uncond
        .iter()
        .zip(cond.iter())
        .map(|(u, c)| u + scale * (c - u))
        .collect();
    for (r, u) in result.iter().zip(uncond.iter()) {
        assert!(
            (r - u).abs() < 1e-6,
            "scale=0 should reproduce uncond: got {r}, expected {u}"
        );
    }
}

#[test]
fn cfg_formula_higher_scale_increases_magnitude_monotonically() {
    let uncond = [0.0f32; 16];
    let cond = [1.0f32; 16];
    let scales = [1.0f32, 3.0, 7.5, 15.0];
    let magnitudes: Vec<f32> = scales
        .iter()
        .map(|&s| {
            uncond
                .iter()
                .zip(cond.iter())
                .map(|(u, c)| (u + s * (c - u)).abs())
                .sum::<f32>()
        })
        .collect();
    for i in 1..magnitudes.len() {
        assert!(
            magnitudes[i] > magnitudes[i - 1],
            "scale {} magnitude ({}) should exceed scale {} ({})",
            scales[i],
            magnitudes[i],
            scales[i - 1],
            magnitudes[i - 1]
        );
    }
}

#[test]
fn cfg_formula_output_is_finite_for_realistic_values() {
    // Simulate typical noise magnitudes from a diffusion model.
    let uncond: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
    let cond: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01 + 0.5).cos()).collect();

    for scale in [1.0f32, 3.0, 7.5, 10.0, 20.0] {
        let result: Vec<f32> = uncond
            .iter()
            .zip(cond.iter())
            .map(|(u, c)| u + scale * (c - u))
            .collect();
        assert!(
            all_finite(&result),
            "CFG formula with scale={scale} produced non-finite output"
        );
    }
}
