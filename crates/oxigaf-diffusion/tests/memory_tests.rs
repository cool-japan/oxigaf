//! Tests for MemoryBudget, OffloadSchedule, SequentialVaeConfig::validate,
//! and DiffusionConfig defaults related to memory / offload settings.

use oxigaf_diffusion::{
    config::DiffusionConfig,
    sequential_vae::SequentialVaeConfig,
    weight_offload::{MemoryBudget, OffloadSchedule, OffloadStrategy},
};

// ---------------------------------------------------------------------------
// SequentialVaeConfig::validate edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_vae_validate_zero_chunk_size() {
    let cfg = SequentialVaeConfig::new(0, 4, 64, 64, 0.18215);
    let err = cfg.validate().expect_err("chunk_size=0 should fail");
    assert!(
        err.to_string().contains("chunk_size"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_sequential_vae_validate_zero_latent_channels() {
    let cfg = SequentialVaeConfig::new(1, 0, 64, 64, 0.18215);
    let err = cfg.validate().expect_err("latent_channels=0 should fail");
    assert!(
        err.to_string().contains("latent_channels"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_sequential_vae_validate_zero_image_height() {
    let cfg = SequentialVaeConfig::new(1, 4, 0, 64, 0.18215);
    let err = cfg.validate().expect_err("image_height=0 should fail");
    assert!(
        err.to_string().contains("image_height"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_sequential_vae_validate_zero_image_width() {
    let cfg = SequentialVaeConfig::new(1, 4, 64, 0, 0.18215);
    let err = cfg.validate().expect_err("image_width=0 should fail");
    assert!(
        err.to_string().contains("image_width"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_sequential_vae_validate_zero_latent_scale() {
    let cfg = SequentialVaeConfig::new(1, 4, 64, 64, 0.0);
    let err = cfg.validate().expect_err("latent_scale=0.0 should fail");
    assert!(
        err.to_string().contains("latent_scale"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_sequential_vae_validate_negative_latent_scale() {
    let cfg = SequentialVaeConfig::new(1, 4, 64, 64, -0.5);
    let err = cfg
        .validate()
        .expect_err("negative latent_scale should fail");
    assert!(
        err.to_string().contains("latent_scale"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_sequential_vae_validate_valid_config() {
    let cfg = SequentialVaeConfig::default();
    cfg.validate().expect("default config should be valid");
}

// ---------------------------------------------------------------------------
// MemoryBudget::can_load / load / unload
// ---------------------------------------------------------------------------

#[test]
fn test_memory_budget_can_load_when_empty() {
    let budget = MemoryBudget::new(8192.0, 0.1);
    // Available ≈ 7372.8 MB — can load 1000 MB.
    assert!(budget.can_load(1000.0));
}

#[test]
fn test_memory_budget_cannot_load_beyond_available() {
    let budget = MemoryBudget::new(4096.0, 0.1);
    // Available ≈ 3686.4 MB — cannot load 4000 MB.
    assert!(!budget.can_load(4000.0));
}

#[test]
fn test_memory_budget_load_success() {
    let mut budget = MemoryBudget::new(4096.0, 0.1);
    budget.load(1000.0).expect("load should succeed");
    assert!((budget.currently_loaded_mb - 1000.0).abs() < 0.01);
}

#[test]
fn test_memory_budget_load_failure_oom() {
    let mut budget = MemoryBudget::new(1024.0, 0.1);
    // Available ≈ 921.6 MB — cannot load 1000 MB.
    let err = budget.load(1000.0).expect_err("should fail OOM");
    assert!(
        err.to_string().contains("VRAM"),
        "expected VRAM error, got: {err}"
    );
}

#[test]
fn test_memory_budget_unload_reduces_loaded() {
    let mut budget = MemoryBudget::new(4096.0, 0.1);
    budget.load(500.0).expect("load 500 MB");
    budget.unload(200.0);
    assert!((budget.currently_loaded_mb - 300.0).abs() < 0.01);
}

#[test]
fn test_memory_budget_unload_clamps_to_zero() {
    let mut budget = MemoryBudget::new(4096.0, 0.1);
    budget.unload(9999.0); // unload more than is loaded
    assert_eq!(budget.currently_loaded_mb, 0.0);
}

// ---------------------------------------------------------------------------
// OffloadSchedule phase ordering for Sequential strategy
// ---------------------------------------------------------------------------

#[test]
fn test_offload_schedule_sequential_phase_count() {
    let sched = OffloadSchedule::for_strategy(OffloadStrategy::Sequential);
    // One phase per component (5 total).
    assert_eq!(sched.total_phases(), 5);
}

#[test]
fn test_offload_schedule_sequential_single_component_per_phase() {
    let sched = OffloadSchedule::for_strategy(OffloadStrategy::Sequential);
    for phase in &sched.phases {
        assert_eq!(
            phase.required_components.len(),
            1,
            "Sequential strategy must have exactly 1 component per phase"
        );
    }
}

#[test]
fn test_offload_schedule_all_in_memory_single_phase() {
    let sched = OffloadSchedule::for_strategy(OffloadStrategy::AllInMemory);
    assert_eq!(sched.total_phases(), 1);
    assert_eq!(sched.phases[0].required_components.len(), 5);
}

#[test]
fn test_offload_schedule_peak_memory_positive() {
    for strategy in [
        OffloadStrategy::AllInMemory,
        OffloadStrategy::Sequential,
        OffloadStrategy::CacheOne,
    ] {
        let sched = OffloadSchedule::for_strategy(strategy);
        assert!(
            sched.peak_memory_mb() > 0.0,
            "peak memory must be > 0 for {strategy:?}"
        );
    }
}

#[test]
fn test_offload_schedule_format_schedule_non_empty() {
    let sched = OffloadSchedule::for_strategy(OffloadStrategy::Sequential);
    let formatted = sched.format_schedule();
    assert!(!formatted.is_empty());
    // Should mention the strategy name.
    assert!(formatted.contains("Sequential"));
}

// ---------------------------------------------------------------------------
// DiffusionConfig default values
// ---------------------------------------------------------------------------

#[test]
fn test_diffusion_config_default_sequential_vae_false() {
    let cfg = DiffusionConfig::default();
    assert!(
        !cfg.sequential_vae,
        "sequential_vae should default to false"
    );
}

#[test]
fn test_diffusion_config_default_vae_chunk_size_one() {
    let cfg = DiffusionConfig::default();
    assert_eq!(cfg.vae_chunk_size, 1, "vae_chunk_size should default to 1");
}

#[test]
fn test_diffusion_config_default_offload_strategy_all_in_memory() {
    let cfg = DiffusionConfig::default();
    assert_eq!(
        cfg.offload_strategy,
        OffloadStrategy::AllInMemory,
        "offload_strategy should default to AllInMemory"
    );
}

#[test]
fn test_diffusion_config_default_num_views() {
    let cfg = DiffusionConfig::default();
    assert_eq!(cfg.num_views, 4);
}
