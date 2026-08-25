//! Integration tests for the training-loop components `TrainingConfig` wires in.
//!
//! Before this wave, `lr_scheduler`, `gradient_clipping`, `gradient_accumulation`
//! and `ema` were implemented, exported, tested in isolation — and unreachable
//! from `Trainer::train_step`.  These tests pin the *wiring*: that each one is
//! off by default, that a configured one is actually built, and that its effect
//! is observable through the public API rather than only inside the module.
//!
//! The parts that need a real `Trainer` need a GPU rasterizer, so the loop is
//! exercised here through the same primitives `Trainer::train_step` calls —
//! `GaussianOptimizer::set_lr_scale`, `GradientClipper::step`,
//! `GradientAccumulator`, `GaussianEma` — driven exactly as the loop drives
//! them.  Anything that genuinely needs a device lives in `end_to_end_tests.rs`
//! behind `#[ignore = "GPU test"]`.

use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

use oxigaf_trainer::config::{
    GradientClipConfig, LrScheduleConfig, OptimizerConfig, TrainingConfig,
};
use oxigaf_trainer::ema::GaussianEma;
use oxigaf_trainer::gradient_accumulation::{
    AccumulationConfig, GradNormalization, GradientAccumulator,
};
use oxigaf_trainer::gradient_clipping::{ClipMode, GradientClipper};
use oxigaf_trainer::lr_scheduler::LrSchedule;
use oxigaf_trainer::optimizer::{GaussianOptimizer, Gradients};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A minimal, deterministic model with `n` Gaussians and SH degree 0.
fn test_model(n: usize) -> GaussianModel {
    let attr = GaussianAttributes {
        position: [0.1, -0.2, 0.3],
        _pad0: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [-3.0, -3.0, -3.0],
        opacity: 0.25,
    };
    GaussianModel {
        gaussians: vec![attr; n],
        sh_coeffs: vec![0.05; n * 3],
        sh_degree: 0,
        face_indices: vec![0; n],
        barycentric: vec![[1.0, 0.0, 0.0]; n],
        local_offsets: vec![[0.01, 0.02, -0.03]; n],
        is_rigid: vec![false; n],
    }
}

/// A gradient buffer whose every group is filled with `value`.
fn filled_gradients(n: usize, value: f32) -> Gradients {
    let mut grads = Gradients::zeros(n, 3);
    for group in [
        &mut grads.position,
        &mut grads.rotation,
        &mut grads.scale,
        &mut grads.opacity,
        &mut grads.sh,
        &mut grads.offset,
    ] {
        group.iter_mut().for_each(|g| *g = value);
    }
    grads
}

// ---------------------------------------------------------------------------
// F130 — the learning rate actually changes across steps per the schedule
// ---------------------------------------------------------------------------

#[test]
fn configured_lr_schedule_changes_the_learning_rate_across_steps() {
    let config = TrainingConfig {
        total_iterations: 1_000,
        lr_schedule: LrScheduleConfig::WarmupCosine {
            warmup_steps: 100,
            total_steps: 1_000,
            min_factor: 0.05,
        },
        ..Default::default()
    };
    config.validate().expect("schedule config must validate");

    let scheduler = config
        .lr_schedule
        .build(config.total_iterations)
        .expect("valid schedule builds")
        .expect("WarmupCosine is not Fixed");

    let model = test_model(2);
    let mut optimizer = GaussianOptimizer::new(&config.optimizer, &model);

    // Drive the multiplier exactly as `Trainer::train_step` does.
    let mut observed = Vec::new();
    for iteration in [1_u32, 50, 100, 500, 999] {
        let scale = scheduler.lr_at(iteration as usize) as f32;
        optimizer
            .set_lr_scale(scale)
            .expect("schedule values are finite and non-negative");
        observed.push(optimizer.position_lr(iteration));
    }

    // Warmup ramps up...
    assert!(
        observed[0] < observed[2],
        "warmup must raise the LR: {:?}",
        observed
    );
    // ...and cosine decay brings it back down.
    assert!(
        observed[4] < observed[2],
        "cosine must lower the LR: {:?}",
        observed
    );
    // Every value must be a usable rate.
    assert!(observed.iter().all(|lr| lr.is_finite() && *lr >= 0.0));
    // At minimum, the rate is not constant — the regression this pins is a
    // schedule that was configured but never consulted.
    let first = observed[0];
    assert!(
        observed.iter().any(|lr| (lr - first).abs() > 1e-12),
        "the LR never changed: {observed:?}"
    );
}

#[test]
fn fixed_schedule_leaves_the_learning_rate_untouched() {
    let config = TrainingConfig::default();
    assert_eq!(config.lr_schedule, LrScheduleConfig::Fixed);
    assert!(config
        .lr_schedule
        .build(config.total_iterations)
        .expect("Fixed builds")
        .is_none());

    let model = test_model(1);
    let optimizer = GaussianOptimizer::new(&OptimizerConfig::default(), &model);
    assert_eq!(optimizer.lr_scale(), 1.0);
}

// ---------------------------------------------------------------------------
// F131 — a synthetic exploding gradient gets clipped
// ---------------------------------------------------------------------------

#[test]
fn exploding_gradients_are_clipped_before_the_optimizer_step() {
    let config = TrainingConfig {
        gradient_clip: GradientClipConfig::GlobalNorm { max_norm: 1.0 },
        ..Default::default()
    };
    config.validate().expect("clip config must validate");
    let mode = config
        .gradient_clip
        .clip_mode()
        .expect("GlobalNorm is not Disabled");
    let mut clipper = GradientClipper::new(mode).expect("valid threshold");

    // A gradient far above the threshold, as an exploding step would produce.
    let exploding = filled_gradients(8, 1_000.0);
    let mut groups = exploding.to_group_vecs();
    let stats = clipper.step(&mut groups).expect("non-empty gradients");

    assert!(stats.was_clipped, "the norm must have exceeded 1.0");
    assert!(
        stats.original_norm > 1_000.0,
        "fixture should explode: {}",
        stats.original_norm
    );
    assert!(
        (stats.clipped_norm - 1.0).abs() < 1e-3,
        "clipped norm should sit at the threshold, got {}",
        stats.clipped_norm
    );

    // The clipped values must survive the round-trip back into `Gradients`.
    let mut clipped = exploding.clone();
    clipped
        .set_from_group_vecs(&groups)
        .expect("shapes are unchanged by clipping");
    assert!(
        clipped.position.iter().all(|g| g.abs() < 1_000.0),
        "clipping must actually shrink the stored gradients"
    );
    assert!(
        clipped.offset.iter().all(|g| g.abs() < 1_000.0),
        "the offset group must be clipped too"
    );

    // A well-behaved gradient passes through untouched.
    let calm = filled_gradients(8, 1e-3);
    let mut calm_groups = calm.to_group_vecs();
    let calm_stats = clipper.step(&mut calm_groups).expect("non-empty gradients");
    assert!(!calm_stats.was_clipped);
    assert_eq!(calm_groups, calm.to_group_vecs());
}

#[test]
fn clipping_is_disabled_by_default() {
    let config = TrainingConfig::default();
    assert_eq!(config.gradient_clip, GradientClipConfig::Disabled);
    assert!(config.gradient_clip.clip_mode().is_none());
}

// ---------------------------------------------------------------------------
// F132 — accumulation batches N steps into one update
// ---------------------------------------------------------------------------

#[test]
fn gradient_accumulation_defers_the_update_and_averages_the_window() {
    let config = TrainingConfig {
        gradient_accumulation_steps: 3,
        ..Default::default()
    };
    config
        .validate()
        .expect("accumulation config must validate");

    let mut accumulator = GradientAccumulator::new(AccumulationConfig {
        accumulation_steps: config.gradient_accumulation_steps as usize,
        normalization: GradNormalization::MeanOverSteps,
        auto_clear: true,
    })
    .expect("3 is a valid window");

    let sizes = Gradients::zeros(4, 3).group_sizes();
    accumulator.initialize(&sizes).expect("sizes are valid");

    // Two micro-batches: not ready.
    for value in [1.0_f32, 2.0] {
        accumulator
            .accumulate(&filled_gradients(4, value).to_group_vecs(), 1)
            .expect("shapes match");
    }
    assert!(
        !accumulator.should_update(),
        "the optimizer must not step mid-window"
    );

    // Third micro-batch completes the window.
    accumulator
        .accumulate(&filled_gradients(4, 3.0).to_group_vecs(), 1)
        .expect("shapes match");
    assert!(accumulator.should_update());

    let averaged = accumulator.apply().expect("window is full");
    let mut effective = Gradients::zeros(4, 3);
    effective
        .set_from_group_vecs(&averaged)
        .expect("shapes are unchanged");
    // mean(1, 2, 3) == 2
    for g in effective.position.iter() {
        assert!((g - 2.0).abs() < 1e-6, "expected the window mean, got {g}");
    }
    for g in effective.offset.iter() {
        assert!((g - 2.0).abs() < 1e-6, "offset group must average too: {g}");
    }

    // `auto_clear` starts the next window from zero.
    assert!(!accumulator.should_update());
}

#[test]
fn accumulation_is_off_by_default() {
    assert_eq!(TrainingConfig::default().gradient_accumulation_steps, 1);
}

// ---------------------------------------------------------------------------
// F133 — EMA weights diverge from the raw weights after N steps
// ---------------------------------------------------------------------------

#[test]
fn ema_weights_diverge_from_raw_weights_after_several_steps() {
    let config = TrainingConfig {
        ema_decay: Some(0.9),
        ..Default::default()
    };
    config.validate().expect("EMA config must validate");
    let decay = config.ema_decay.expect("just set");

    let mut model = test_model(3);
    let mut ema = GaussianEma::new(&model, decay);

    // A real optimizer step, repeated, exactly as the loop drives it.
    let mut optimizer = GaussianOptimizer::new(&config.optimizer, &model);
    let gradients = filled_gradients(model.len(), 1.0);
    for iteration in 1..=10_u32 {
        optimizer
            .step(&mut model, &gradients, iteration)
            .expect("shapes match the model");
        ema.update(&model);
    }

    let mut averaged = model.clone();
    ema.apply_to(&mut averaged);

    // The shadow lags the live weights, so the two must differ.
    let raw_opacity = model.gaussians[0].opacity;
    let ema_opacity = averaged.gaussians[0].opacity;
    assert!(
        (raw_opacity - ema_opacity).abs() > 1e-6,
        "EMA {ema_opacity} should lag raw {raw_opacity}"
    );

    // ...and it must lag *behind*, i.e. sit between the start and the live
    // value, not somewhere unrelated.
    let start_opacity = test_model(1).gaussians[0].opacity;
    let lo = raw_opacity.min(start_opacity);
    let hi = raw_opacity.max(start_opacity);
    assert!(
        ema_opacity >= lo - 1e-6 && ema_opacity <= hi + 1e-6,
        "EMA {ema_opacity} left the [{lo}, {hi}] interval"
    );

    assert_eq!(ema.step(), 10);
    assert!(ema.effective_decay() <= decay);
}

#[test]
fn ema_is_off_by_default() {
    assert!(TrainingConfig::default().ema_decay.is_none());
}

// ---------------------------------------------------------------------------
// Config-level guards: a misconfigured component is reported, never dropped
// ---------------------------------------------------------------------------

#[test]
fn misconfigured_components_are_rejected_by_validate() {
    let cases: [(&str, TrainingConfig); 4] = [
        (
            "zero accumulation window",
            TrainingConfig {
                gradient_accumulation_steps: 0,
                ..Default::default()
            },
        ),
        (
            "out-of-range EMA decay",
            TrainingConfig {
                ema_decay: Some(1.5),
                ..Default::default()
            },
        ),
        (
            "non-positive clip threshold",
            TrainingConfig {
                gradient_clip: GradientClipConfig::GlobalNorm { max_norm: 0.0 },
                ..Default::default()
            },
        ),
        (
            "warmup longer than the schedule",
            TrainingConfig {
                total_iterations: 100,
                lr_schedule: LrScheduleConfig::WarmupCosine {
                    warmup_steps: 500,
                    total_steps: 100,
                    min_factor: 0.0,
                },
                ..Default::default()
            },
        ),
    ];

    for (name, config) in cases {
        assert!(
            config.validate().is_err(),
            "{name} should have been rejected"
        );
    }
}

#[test]
fn clip_mode_round_trips_through_the_serialisable_config() {
    let cases = [
        (
            GradientClipConfig::GlobalNorm { max_norm: 2.0 },
            ClipMode::GlobalNorm { max_norm: 2.0 },
        ),
        (
            GradientClipConfig::PerGroupNorm { max_norm: 3.0 },
            ClipMode::PerGroupNorm { max_norm: 3.0 },
        ),
        (
            GradientClipConfig::Value { max_value: 0.5 },
            ClipMode::ValueClip { max_val: 0.5 },
        ),
    ];
    for (config, expected) in cases {
        assert_eq!(config.clip_mode(), Some(expected));
        config.validate().expect("all thresholds are positive");
        let json = serde_json::to_string(&config).expect("config serialises");
        let restored: GradientClipConfig = serde_json::from_str(&json).expect("config round-trips");
        assert_eq!(restored, config);
    }
}

// ---------------------------------------------------------------------------
// Density control invalidates per-Gaussian state (accumulation window + EMA)
// ---------------------------------------------------------------------------

#[test]
fn accumulation_rejects_a_window_whose_model_was_resized() {
    // Density control runs *after* the optimizer step, so it can resize the
    // model while an accumulation window is still filling.  The accumulator
    // must refuse the mismatched micro-batch rather than silently accumulate
    // into buffers sized for a model that no longer exists.
    let mut accumulator = GradientAccumulator::new(AccumulationConfig {
        accumulation_steps: 4,
        normalization: GradNormalization::MeanOverSteps,
        auto_clear: true,
    })
    .expect("4 is a valid window");

    accumulator
        .initialize(&Gradients::zeros(4, 3).group_sizes())
        .expect("sizes are valid");
    accumulator
        .accumulate(&filled_gradients(4, 1.0).to_group_vecs(), 1)
        .expect("first micro-batch matches");

    // Densification added Gaussians mid-window.
    let grown = filled_gradients(7, 1.0);
    assert!(
        accumulator.accumulate(&grown.to_group_vecs(), 1).is_err(),
        "a resized model must not be folded into the open window"
    );

    // `clear()` (what the trainer calls on a resize) then a re-`initialize`
    // at the new sizes restores a usable accumulator.
    accumulator.clear();
    accumulator
        .initialize(&grown.group_sizes())
        .expect("new sizes are valid");
    accumulator
        .accumulate(&grown.to_group_vecs(), 1)
        .expect("the new window accepts the new shape");
    assert_eq!(accumulator.steps_accumulated, 1);
}

#[test]
fn ema_shadow_is_meaningless_across_a_prune_and_must_be_rebuilt() {
    // Pruning K Gaussians and appending K leaves the *count* identical while
    // changing which Gaussian sits at every index — so keying invalidation on
    // the length alone would blend unrelated parameters forever.
    let decay = 0.9_f32;
    let mut model = test_model(4);
    let mut ema = GaussianEma::new(&model, decay);
    for _ in 0..5 {
        for g in model.gaussians.iter_mut() {
            g.opacity += 0.1;
        }
        ema.update(&model);
    }

    // Simulate a prune+clone that keeps the count: index 0 now holds a
    // completely different Gaussian.
    let mut resized = model.clone();
    resized.gaussians[0].opacity = -5.0;

    // Carrying the old shadow forward would average -5.0 against the history
    // of the Gaussian that used to live there.
    let mut stale = resized.clone();
    ema.apply_to(&mut stale);
    assert!(
        (stale.gaussians[0].opacity - resized.gaussians[0].opacity).abs() > 1e-3,
        "the stale shadow visibly contaminates the replaced Gaussian"
    );

    // Rebuilding restarts the average from the live weights, which is what
    // `Trainer::reset_size_dependent_state` does.
    let rebuilt = GaussianEma::new(&resized, decay);
    let mut fresh = resized.clone();
    rebuilt.apply_to(&mut fresh);
    assert!(
        (fresh.gaussians[0].opacity - resized.gaussians[0].opacity).abs() < 1e-6,
        "a rebuilt shadow starts at the live weights"
    );
    assert_eq!(rebuilt.step(), 0);
}
