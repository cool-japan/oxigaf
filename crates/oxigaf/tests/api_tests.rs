//! Integration tests for the OxiGAF meta-crate public API.

use oxigaf::{
    check_gpu, detect_best_backend, export, render_from_file, validate_config, verify_assets,
    ExportFormat, OxigafError, PipelineBuilder, PipelineConfig,
};
use std::fs;

// ============================================================================
// PipelineBuilder tests
// ============================================================================

#[test]
fn test_pipeline_builder_default_values() {
    // Builder should have sensible defaults for num_views and iterations.
    // Without required fields it should fail on build().
    let err = PipelineBuilder::new()
        .build()
        .expect_err("should fail without required fields");
    assert!(
        matches!(err, OxigafError::InvalidConfig(_)),
        "expected InvalidConfig, got {err:?}"
    );
}

#[test]
fn test_pipeline_builder_roundtrip() {
    let tmp = std::env::temp_dir().join("oxigaf_builder_test");
    fs::create_dir_all(&tmp).expect("create temp dir");

    let config = PipelineBuilder::new()
        .flame_model_path(&tmp)
        .output_dir(&tmp)
        .num_views(4)
        .iterations(1000)
        .build()
        .expect("builder should succeed");

    assert_eq!(config.flame_model_path, tmp);
    assert_eq!(config.output_dir, tmp);
    assert_eq!(config.num_views, 4);
    assert_eq!(config.iterations, 1000);

    fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_pipeline_builder_zero_num_views_rejected() {
    let tmp = std::env::temp_dir().join("oxigaf_zero_views_test");
    fs::create_dir_all(&tmp).expect("create temp dir");

    let err = PipelineBuilder::new()
        .flame_model_path(&tmp)
        .output_dir(&tmp)
        .num_views(0)
        .build()
        .expect_err("zero num_views should fail");

    assert!(matches!(err, OxigafError::InvalidConfig(_)));
    fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_pipeline_builder_zero_iterations_rejected() {
    let tmp = std::env::temp_dir().join("oxigaf_zero_iter_test");
    fs::create_dir_all(&tmp).expect("create temp dir");

    let err = PipelineBuilder::new()
        .flame_model_path(&tmp)
        .output_dir(&tmp)
        .iterations(0)
        .build()
        .expect_err("zero iterations should fail");

    assert!(matches!(err, OxigafError::InvalidConfig(_)));
    fs::remove_dir_all(&tmp).ok();
}

// ============================================================================
// validate_config tests
// ============================================================================

#[test]
fn test_validate_config_catches_nonexistent_paths() {
    let config = PipelineConfig {
        flame_model_path: std::path::PathBuf::from("/nonexistent/flame/path/that/cannot/exist"),
        output_dir: std::env::temp_dir().join("oxigaf_validate_nonexistent_out"),
        num_views: 8,
        iterations: 1000,
    };

    let err = validate_config(&config).expect_err("nonexistent path should fail validation");
    assert!(
        matches!(err, OxigafError::PathNotFound(_)),
        "expected PathNotFound, got {err:?}"
    );
}

#[test]
fn test_validate_config_passes_for_existing_path() {
    let tmp = std::env::temp_dir().join("oxigaf_validate_test");
    fs::create_dir_all(&tmp).expect("create temp dir");

    let config = PipelineConfig {
        flame_model_path: tmp.clone(),
        output_dir: tmp.clone(),
        num_views: 4,
        iterations: 500,
    };

    validate_config(&config).expect("valid config should pass");
    fs::remove_dir_all(&tmp).ok();
}

// ============================================================================
// check_gpu tests
// ============================================================================

#[test]
fn test_check_gpu_returns_result() {
    // Should not panic; empty vec is acceptable on headless CI.
    let result = check_gpu();
    assert!(
        result.is_ok(),
        "check_gpu() should not return an error; got {result:?}"
    );
    let gpus = result.expect("ok above");
    // If any adapters were found, their fields should be non-empty.
    for gpu in &gpus {
        assert!(!gpu.backend.is_empty());
        assert!(!gpu.device_type.is_empty());
    }
}

// ============================================================================
// verify_assets tests
// ============================================================================

#[test]
fn test_verify_assets_returns_missing_list() {
    let tmp = std::env::temp_dir().join("oxigaf_assets_test");
    fs::create_dir_all(&tmp).expect("create temp dir");

    // Create some but not all expected assets.
    fs::write(tmp.join("shape_dirs.npy"), b"dummy").expect("write");
    fs::write(tmp.join("exp_dirs.npy"), b"dummy").expect("write");

    let missing = verify_assets(&tmp);

    // shape_dirs and exp_dirs are present; others should be missing.
    assert!(
        !missing.contains(&"shape_dirs.npy".to_string()),
        "shape_dirs.npy should not be in missing list"
    );
    assert!(
        !missing.contains(&"exp_dirs.npy".to_string()),
        "exp_dirs.npy should not be in missing list"
    );
    assert!(
        missing.contains(&"v_template.npy".to_string()),
        "v_template.npy should be reported missing"
    );

    fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_verify_assets_all_present() {
    const EXPECTED: &[&str] = &[
        "shape_dirs.npy",
        "exp_dirs.npy",
        "posedirs.npy",
        "v_template.npy",
        "J_regressor.npy",
        "kintree_table.npy",
        "faces.npy",
    ];

    let tmp = std::env::temp_dir().join("oxigaf_assets_all_test");
    fs::create_dir_all(&tmp).expect("create temp dir");
    for name in EXPECTED {
        fs::write(tmp.join(name), b"dummy").expect("write");
    }

    let missing = verify_assets(&tmp);
    assert!(
        missing.is_empty(),
        "no files should be missing; got {missing:?}"
    );

    fs::remove_dir_all(&tmp).ok();
}

// ============================================================================
// detect_best_backend tests
// ============================================================================

#[test]
fn test_detect_best_backend_non_empty() {
    let backend = detect_best_backend();
    assert!(!backend.is_empty(), "backend string must not be empty");
    // Should be one of the known backends.
    let known = ["Metal", "Vulkan", "Dx12", "Gl", "Unknown"];
    assert!(
        known.contains(&backend.as_str()),
        "unexpected backend string: {backend}"
    );
}

#[test]
#[cfg(target_os = "macos")]
fn test_detect_best_backend_is_metal_on_macos() {
    assert_eq!(detect_best_backend(), "Metal");
}

// ============================================================================
// ExportFormat tests
// ============================================================================

#[test]
fn test_export_format_debug() {
    assert_eq!(format!("{:?}", ExportFormat::Ply), "Ply");
    assert_eq!(format!("{:?}", ExportFormat::Gltf), "Gltf");
    assert_eq!(format!("{:?}", ExportFormat::Obj), "Obj");
}

#[test]
fn test_export_format_equality() {
    assert_eq!(ExportFormat::Ply, ExportFormat::Ply);
    assert_ne!(ExportFormat::Ply, ExportFormat::Gltf);
    let fmt = ExportFormat::Gltf;
    assert_eq!(fmt, ExportFormat::Gltf);
}

// ============================================================================
// render_from_file and export stub tests
// ============================================================================

#[test]
fn test_render_from_file_missing_path_returns_err() {
    let err = render_from_file(
        "/nonexistent/model.ply",
        std::env::temp_dir().join("oxigaf_render_missing_out.png"),
        512,
        512,
    )
    .expect_err("missing model path should fail");
    assert!(matches!(err, OxigafError::PathNotFound(_)));
}

#[test]
fn test_render_from_file_zero_dimensions_returns_err() {
    let tmp = std::env::temp_dir().join("oxigaf_render_test.ply");
    fs::write(&tmp, b"ply").expect("write dummy ply");

    let out = std::env::temp_dir().join("oxigaf_render_zero_dim_out.png");
    let err =
        render_from_file(tmp.as_path(), out.as_path(), 0, 512).expect_err("zero width should fail");
    assert!(matches!(err, OxigafError::InvalidConfig(_)));

    fs::remove_file(&tmp).ok();
}

#[test]
fn test_export_missing_path_returns_err() {
    let err = export(
        "/nonexistent/model.ply",
        std::env::temp_dir().join("oxigaf_export_missing_out.gltf"),
        ExportFormat::Gltf,
    )
    .expect_err("missing model path should fail");
    assert!(matches!(err, OxigafError::PathNotFound(_)));
}
