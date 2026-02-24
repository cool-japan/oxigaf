//! Integration tests for checkpoint save/load.
//!
//! Uses std::env::temp_dir() for all file operations.

use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf_trainer::checkpoint::{
    build_checkpoint, load_checkpoint, save_checkpoint, CheckpointData,
};
use oxigaf_trainer::config::OptimizerConfig;
use oxigaf_trainer::metrics::MetricTracker;
use oxigaf_trainer::optimizer::GaussianOptimizer;
use oxigaf_trainer::{TrainerError, CHECKPOINT_VERSION};
use std::fs;

/// Create a test model with specified Gaussians.
fn make_model(n: usize) -> GaussianModel {
    GaussianModel {
        gaussians: (0..n)
            .map(|i| GaussianAttributes {
                position: [i as f32, (i * 2) as f32, (i * 3) as f32],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-3.0, -4.0, -5.0],
                opacity: -1.0,
            })
            .collect(),
        sh_coeffs: vec![0.5; n * 3], // sh_degree 0 → 3 coeffs per Gaussian
        sh_degree: 0,
        face_indices: vec![0; n],
        barycentric: vec![[1.0, 0.0, 0.0]; n],
        local_offsets: vec![[0.0; 3]; n],
        is_rigid: vec![true; n],
    }
}

// ============================================================================
// Round-trip Tests
// ============================================================================

#[test]
fn checkpoint_file_roundtrip() -> Result<(), TrainerError> {
    let model = make_model(5);
    let opt = GaussianOptimizer::new(&OptimizerConfig::default(), &model);
    let tracker = MetricTracker::new();
    let ckpt = build_checkpoint(&model, &opt, 100, &tracker);

    // Use temp directory
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_ckpt_roundtrip.json");

    // Save
    save_checkpoint(&path, &ckpt)?;

    // Load
    let loaded = load_checkpoint(&path)?;

    // Verify
    assert_eq!(loaded.version, CHECKPOINT_VERSION);
    assert_eq!(loaded.iteration, 100);
    assert_eq!(loaded.positions.len(), 5);
    assert_eq!(loaded.positions[2], [2.0, 4.0, 6.0]);
    assert_eq!(loaded.sh_coeffs.len(), 15);

    // Cleanup
    let _ = fs::remove_file(&path);

    Ok(())
}

#[test]
fn checkpoint_preserves_model_data() -> Result<(), TrainerError> {
    let mut model = make_model(3);
    model.gaussians[1].rotation = [0.1, 0.2, 0.3, 0.94];
    model.gaussians[2].scale = [-1.0, -2.0, -3.0];
    model.local_offsets[0] = [0.1, 0.2, 0.3];
    model.is_rigid[1] = false;

    let opt = GaussianOptimizer::new(&OptimizerConfig::default(), &model);
    let tracker = MetricTracker::new();
    let ckpt = build_checkpoint(&model, &opt, 50, &tracker);

    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_ckpt_model_data.json");

    save_checkpoint(&path, &ckpt)?;
    let loaded = load_checkpoint(&path)?;

    assert_eq!(loaded.rotations[1], [0.1, 0.2, 0.3, 0.94]);
    assert_eq!(loaded.scales[2], [-1.0, -2.0, -3.0]);
    assert_eq!(loaded.local_offsets[0], [0.1, 0.2, 0.3]);
    assert!(!loaded.is_rigid[1]);

    let _ = fs::remove_file(&path);
    Ok(())
}

#[test]
fn checkpoint_preserves_optimizer_state() -> Result<(), TrainerError> {
    let model = make_model(2);
    let mut opt = GaussianOptimizer::new(&OptimizerConfig::default(), &model);

    // Modify optimizer state
    opt.position.m[0] = 1.5;
    opt.position.v[0] = 0.25;
    opt.position.t = 42;

    let tracker = MetricTracker::new();
    let ckpt = build_checkpoint(&model, &opt, 42, &tracker);

    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_ckpt_opt_state.json");

    save_checkpoint(&path, &ckpt)?;
    let loaded = load_checkpoint(&path)?;

    // Find and verify position group
    let pos_group = loaded
        .optimizer_groups
        .iter()
        .find(|g| g.name == "position");
    assert!(pos_group.is_some());

    if let Some(pg) = pos_group {
        assert!((pg.m[0] - 1.5).abs() < 1e-8);
        assert!((pg.v[0] - 0.25).abs() < 1e-8);
        assert_eq!(pg.t, 42);
    }

    let _ = fs::remove_file(&path);
    Ok(())
}

// ============================================================================
// Validation Tests
// ============================================================================

#[test]
fn validate_detects_array_length_mismatch() {
    let data = CheckpointData {
        version: CHECKPOINT_VERSION,
        iteration: 0,
        positions: vec![[0.0; 3]; 3],
        rotations: vec![[0.0, 0.0, 0.0, 1.0]; 3],
        scales: vec![[0.0; 3]; 3],
        opacities: vec![0.0; 2], // Wrong length!
        sh_coeffs: vec![0.0; 9],
        sh_degree: 0,
        face_indices: vec![0; 3],
        barycentric: vec![[1.0, 0.0, 0.0]; 3],
        local_offsets: vec![[0.0; 3]; 3],
        is_rigid: vec![true; 3],
        optimizer_groups: vec![],
        metrics_history: vec![],
    };

    let result = data.validate();
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(TrainerError::CheckpointDataMismatch { .. })
    ));
}

#[test]
fn validate_detects_nan_in_positions() {
    let data = CheckpointData {
        version: CHECKPOINT_VERSION,
        iteration: 0,
        positions: vec![[f32::NAN, 0.0, 0.0]],
        rotations: vec![[0.0, 0.0, 0.0, 1.0]],
        scales: vec![[0.0; 3]],
        opacities: vec![0.0],
        sh_coeffs: vec![0.0; 3],
        sh_degree: 0,
        face_indices: vec![0],
        barycentric: vec![[1.0, 0.0, 0.0]],
        local_offsets: vec![[0.0; 3]],
        is_rigid: vec![true],
        optimizer_groups: vec![],
        metrics_history: vec![],
    };

    let result = data.validate();
    assert!(matches!(result, Err(TrainerError::NanDetected { .. })));
}

#[test]
fn validate_detects_inf_in_scales() {
    let data = CheckpointData {
        version: CHECKPOINT_VERSION,
        iteration: 0,
        positions: vec![[0.0; 3]],
        rotations: vec![[0.0, 0.0, 0.0, 1.0]],
        scales: vec![[f32::INFINITY, 0.0, 0.0]],
        opacities: vec![0.0],
        sh_coeffs: vec![0.0; 3],
        sh_degree: 0,
        face_indices: vec![0],
        barycentric: vec![[1.0, 0.0, 0.0]],
        local_offsets: vec![[0.0; 3]],
        is_rigid: vec![true],
        optimizer_groups: vec![],
        metrics_history: vec![],
    };

    let result = data.validate();
    assert!(matches!(result, Err(TrainerError::InfDetected { .. })));
}

#[test]
fn validate_detects_version_mismatch() {
    let data = CheckpointData {
        version: CHECKPOINT_VERSION + 100, // Future version
        iteration: 0,
        positions: vec![[0.0; 3]],
        rotations: vec![[0.0, 0.0, 0.0, 1.0]],
        scales: vec![[0.0; 3]],
        opacities: vec![0.0],
        sh_coeffs: vec![0.0; 3],
        sh_degree: 0,
        face_indices: vec![0],
        barycentric: vec![[1.0, 0.0, 0.0]],
        local_offsets: vec![[0.0; 3]],
        is_rigid: vec![true],
        optimizer_groups: vec![],
        metrics_history: vec![],
    };

    let result = data.validate();
    assert!(matches!(
        result,
        Err(TrainerError::CheckpointVersionMismatch { .. })
    ));
}

#[test]
fn validate_detects_sh_length_mismatch() {
    let data = CheckpointData {
        version: CHECKPOINT_VERSION,
        iteration: 0,
        positions: vec![[0.0; 3]; 2],
        rotations: vec![[0.0, 0.0, 0.0, 1.0]; 2],
        scales: vec![[0.0; 3]; 2],
        opacities: vec![0.0; 2],
        sh_coeffs: vec![0.0; 3], // Should be 6 for 2 Gaussians with degree 0
        sh_degree: 0,
        face_indices: vec![0; 2],
        barycentric: vec![[1.0, 0.0, 0.0]; 2],
        local_offsets: vec![[0.0; 3]; 2],
        is_rigid: vec![true; 2],
        optimizer_groups: vec![],
        metrics_history: vec![],
    };

    let result = data.validate();
    assert!(matches!(
        result,
        Err(TrainerError::CheckpointDataMismatch { .. })
    ));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn load_nonexistent_file_returns_error() {
    let path = std::env::temp_dir().join("nonexistent_checkpoint_xyz123.json");
    let result = load_checkpoint(&path);
    assert!(result.is_err());
}

#[test]
fn load_corrupted_json_returns_error() -> Result<(), std::io::Error> {
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_corrupted_ckpt.json");

    // Write invalid JSON
    fs::write(&path, "{ this is not valid json }")?;

    let result = load_checkpoint(&path);
    assert!(result.is_err());
    assert!(matches!(result, Err(TrainerError::CheckpointCorrupted(_))));

    let _ = fs::remove_file(&path);
    Ok(())
}

// ============================================================================
// Large Model Tests
// ============================================================================

#[test]
fn checkpoint_large_model() -> Result<(), TrainerError> {
    let model = make_model(1000);
    let opt = GaussianOptimizer::new(&OptimizerConfig::default(), &model);
    let tracker = MetricTracker::new();
    let ckpt = build_checkpoint(&model, &opt, 500, &tracker);

    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_large_ckpt.json");

    save_checkpoint(&path, &ckpt)?;
    let loaded = load_checkpoint(&path)?;

    assert_eq!(loaded.positions.len(), 1000);
    assert_eq!(loaded.sh_coeffs.len(), 3000);

    let _ = fs::remove_file(&path);
    Ok(())
}
