//! Regression tests: the `oxigaf` meta-crate's Cargo features must actually
//! reach the sub-crate they claim to forward to.
//!
//! Every feature on `oxigaf` is a pure forwarding alias (`gpu_debug =
//! ["oxigaf-render/gpu_debug", "oxigaf-diffusion/gpu_debug"]`, `npz =
//! ["oxigaf-flame/npz"]`, …). Nothing in the type system enforces that, so a
//! dropped or mistyped forwarding entry silently turns `--features X` into a
//! no-op: the crate still builds, the flag still "works", and the behaviour it
//! was supposed to switch on never appears. That is exactly how
//! `--features gpu_debug` once left `oxigaf-diffusion`'s NaN/Inf hooks off.
//!
//! Each test below therefore observes a *behavioural* marker owned by the
//! downstream crate, and asserts both directions:
//!
//! - with the feature on, the downstream marker must be on;
//! - with the feature off, it must be off.
//!
//! Both halves only compile in their own configuration, so this file must be
//! run twice to cover everything:
//!
//! ```text
//! cargo test -p oxigaf --test feature_forwarding_tests
//! cargo test -p oxigaf --all-features --test feature_forwarding_tests
//! ```
//!
//! Two features have no assertion here, deliberately:
//!
//! - `simd` — `oxigaf-flame` exposes `pub mod simd` unconditionally and has no
//!   `cfg!(feature = "simd")` marker anywhere in its public surface (the
//!   feature is nightly-only and compiles out on stable), so there is nothing
//!   observable to assert without inventing a marker in a crate this test does
//!   not own.
//! - `gpu_debug` → `oxigaf-render` — the render half of that forwarding only
//!   changes `wgpu::InstanceFlags` inside `Rasterizer::new` and adds
//!   `validate_no_nan_inf` calls in the backward pass, neither of which is
//!   observable without a live GPU adapter. The diffusion half *is* asserted
//!   below, which is enough to catch a dropped forwarding entry.

// ---------------------------------------------------------------------------
// gpu_debug -> oxigaf-diffusion/gpu_debug
// ---------------------------------------------------------------------------

/// `DebugConfig::default().enabled` is literally `cfg!(feature = "gpu_debug")`
/// inside `oxigaf-diffusion`, so it reports whether the forwarding landed.
#[cfg(feature = "gpu_debug")]
#[test]
fn gpu_debug_reaches_diffusion_debug_hooks() {
    let config = oxigaf::diffusion::debug_hooks::DebugConfig::default();
    assert!(
        config.enabled,
        "oxigaf's `gpu_debug` must forward to oxigaf-diffusion/gpu_debug; \
         DebugConfig::default().enabled was false, meaning the NaN/Inf hooks \
         are inert despite the feature being requested"
    );
}

#[cfg(not(feature = "gpu_debug"))]
#[test]
fn without_gpu_debug_diffusion_debug_hooks_are_inert() {
    let config = oxigaf::diffusion::debug_hooks::DebugConfig::default();
    assert!(
        !config.enabled,
        "a default build must not pay for NaN/Inf scanning"
    );
}

/// Beyond the config flag, the hooks must actually *behave* differently: an
/// enabled registry scans and reports the anomaly, a disabled one short-
/// circuits and reports the tensor as healthy without looking at it.
#[test]
fn debug_hooks_default_behaviour_matches_the_feature() {
    use oxigaf::diffusion::debug_hooks::{DebugConfig, DebugHooks};

    let hooks = DebugHooks::new(DebugConfig::default());
    let health = hooks.check("forwarding_probe", &[0.0f32, 1.0, f32::NAN]);

    if cfg!(feature = "gpu_debug") {
        assert!(
            !health.is_healthy,
            "with gpu_debug on, the NaN must be detected"
        );
        assert_eq!(health.nan_count, 1);
    } else {
        assert!(
            health.is_healthy,
            "with gpu_debug off, check() must short-circuit without scanning"
        );
    }
}

// ---------------------------------------------------------------------------
// flash_attention -> oxigaf-diffusion/flash_attention
// ---------------------------------------------------------------------------

/// `DiffusionConfig::default().use_flash_attention` is `#[cfg]`-selected in
/// `oxigaf-diffusion`. `oxigaf-diffusion`'s own `default` feature set is empty,
/// so this is `false` unless the forwarding works.
#[cfg(feature = "flash_attention")]
#[test]
fn flash_attention_reaches_diffusion_config() {
    let config = oxigaf::diffusion::DiffusionConfig::default();
    assert!(
        config.use_flash_attention,
        "oxigaf's `flash_attention` must forward to \
         oxigaf-diffusion/flash_attention"
    );
}

#[cfg(not(feature = "flash_attention"))]
#[test]
fn without_flash_attention_diffusion_config_uses_standard_attention() {
    let config = oxigaf::diffusion::DiffusionConfig::default();
    assert!(
        !config.use_flash_attention,
        "flash attention must be opt-in; oxigaf-diffusion's `default` is empty"
    );
}

// ---------------------------------------------------------------------------
// mixed_precision -> oxigaf-diffusion/mixed_precision
// ---------------------------------------------------------------------------

#[cfg(feature = "mixed_precision")]
#[test]
fn mixed_precision_reaches_diffusion_precision_config() {
    use oxigaf::diffusion::mixed_precision::{MixedPrecisionConfig, PrecisionMode};

    assert_eq!(
        MixedPrecisionConfig::default().mode,
        PrecisionMode::BFloat16,
        "oxigaf's `mixed_precision` must forward to \
         oxigaf-diffusion/mixed_precision"
    );
}

#[cfg(not(feature = "mixed_precision"))]
#[test]
fn without_mixed_precision_diffusion_stays_fp32() {
    use oxigaf::diffusion::mixed_precision::{MixedPrecisionConfig, PrecisionMode};

    assert_eq!(
        MixedPrecisionConfig::default().mode,
        PrecisionMode::Float32,
        "a default build must stay in FP32"
    );
}

// ---------------------------------------------------------------------------
// parallel -> oxigaf-flame/parallel
// ---------------------------------------------------------------------------

/// `oxigaf-flame` gates `compute_normals_batch_par` / `recompute_batch_normals_par`
/// behind its `parallel` feature, so merely *naming* them proves the forwarding
/// landed. This is a compile-time assertion; the runtime call keeps it honest.
#[cfg(feature = "parallel")]
#[test]
fn parallel_reaches_flame_rayon_entry_points() {
    use nalgebra::{Point3, Vector3};

    // Naming the item is already the assertion: this line does not compile
    // unless `oxigaf-flame/parallel` is active. Calling it keeps the test
    // honest about the rayon path actually running.
    let parallel_entry_point = oxigaf::flame::compute_normals_batch_par;

    // A single triangle in the z = 0 plane, batched twice.
    let vertices = vec![
        Point3::new(0.0f32, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let vertices_batch = vec![vertices.clone(), vertices];
    let faces = [[0u32, 1, 2]];
    let mut normals_batch = vec![vec![Vector3::zeros(); 3], vec![Vector3::zeros(); 3]];

    parallel_entry_point(&vertices_batch, &faces, &mut normals_batch);

    for normals in &normals_batch {
        assert_eq!(normals.len(), 3, "one normal per vertex");
        for normal in normals {
            assert!(
                (normal.z.abs() - 1.0).abs() < 1e-5,
                "a triangle in the z = 0 plane must have ±Z normals, got {normal:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// npz -> oxigaf-flame/npz
// ---------------------------------------------------------------------------

/// With `npz` off, `oxigaf-flame` compiles a stub `FlameSequence::from_npz`
/// that unconditionally reports the feature is disabled — it never touches the
/// filesystem. With `npz` on, the real reader runs and fails for the ordinary
/// reason (the file is not there). Probing a path that cannot exist therefore
/// distinguishes the two builds without needing a fixture archive.
#[test]
fn npz_forwarding_selects_the_real_reader() {
    use oxigaf::flame::FlameSequence;

    let missing = std::env::temp_dir().join("oxigaf_feature_forwarding_absent.npz");
    // Guard against a stray file from an earlier run making this vacuous.
    assert!(
        !missing.exists(),
        "test fixture path must not exist: {}",
        missing.display()
    );

    let err = match FlameSequence::from_npz(&missing) {
        Ok(_) => panic!("loading a non-existent .npz must fail"),
        Err(err) => err.to_string(),
    };

    if cfg!(feature = "npz") {
        assert!(
            !err.contains("NPZ support not enabled"),
            "oxigaf's `npz` must forward to oxigaf-flame/npz, but the \
             feature-disabled stub answered instead: {err}"
        );
    } else {
        assert!(
            err.contains("NPZ support not enabled"),
            "without `npz`, from_npz must report the disabled feature rather \
             than an I/O error: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// Bundles
// ---------------------------------------------------------------------------

/// `all_features` is documented as the maximal bundle. If a feature is ever
/// added to `[features]` without being added to the bundle, the bundle stops
/// being maximal — and, because `--all-features` and `--features all_features`
/// then differ, CI can pass while the documented bundle is broken.
#[cfg(feature = "all_features")]
#[test]
fn all_features_bundle_enables_every_forwarded_feature() {
    for (name, enabled) in [
        ("simd", cfg!(feature = "simd")),
        ("parallel", cfg!(feature = "parallel")),
        ("flash_attention", cfg!(feature = "flash_attention")),
        ("mixed_precision", cfg!(feature = "mixed_precision")),
        ("gpu_debug", cfg!(feature = "gpu_debug")),
        ("npz", cfg!(feature = "npz")),
    ] {
        assert!(
            enabled,
            "`all_features` must enable `{name}`; add it to the bundle in \
             crates/oxigaf/Cargo.toml"
        );
    }
}

/// `full_performance` is the *speed* bundle: it must pull in all three
/// performance features and nothing else.
///
/// Only the positive half is asserted. The negative half — "`full_performance`
/// must not enable `gpu_debug` / `npz`" — is deliberately absent: `cfg!` cannot
/// tell "the bundle turned this on" from "the user asked for it alongside", so
/// `--features full_performance,npz` would fail such an assertion spuriously.
/// That the bundle's own list is exclusive is visible in Cargo.toml and
/// enforced by Cargo, not by this crate.
#[cfg(feature = "full_performance")]
#[test]
fn full_performance_bundle_enables_every_performance_feature() {
    for (name, enabled) in [
        ("simd", cfg!(feature = "simd")),
        ("parallel", cfg!(feature = "parallel")),
        ("flash_attention", cfg!(feature = "flash_attention")),
    ] {
        assert!(enabled, "`full_performance` must enable `{name}`");
    }
}
