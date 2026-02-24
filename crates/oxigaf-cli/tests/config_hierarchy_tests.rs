//! Integration tests for hierarchical configuration loading.
//!
//! Tests the priority order: CLI args > env vars > project config > user config > defaults

use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use serial_test::serial;

// Import the config module (assuming it's publicly accessible)
use oxigaf_cli::config::{load_hierarchical_config, ProjectConfig};

/// Get a temporary directory for test files
fn get_temp_test_dir(test_name: &str) -> PathBuf {
    let mut path = env::temp_dir();
    path.push(format!("oxigaf_test_{}", test_name));
    if path.exists() {
        fs::remove_dir_all(&path).ok();
    }
    fs::create_dir_all(&path).expect("Failed to create test directory");
    path
}

/// Clean up test directory
fn cleanup_test_dir(path: &PathBuf) {
    if path.exists() {
        fs::remove_dir_all(path).ok();
    }
}

/// Clean up environment variables used in tests
fn cleanup_env_vars() {
    env::remove_var("OXIGAF_TOTAL_ITERATIONS");
    env::remove_var("OXIGAF_IMAGE_SIZE");
    env::remove_var("OXIGAF_POSITION_LR");
    env::remove_var("OXIGAF_SCALING_LR");
    env::remove_var("OXIGAF_DEVICE_GPU_INDEX");
    env::remove_var("OXIGAF_OUTPUT_CHECKPOINT_INTERVAL");
    env::remove_var("OXIGAF_OUTPUT_EXPORT_FORMAT");
    env::remove_var("OXIGAF_SH_DEGREE");
}

#[test]
#[serial]
fn test_default_config_loads() -> Result<()> {
    cleanup_env_vars();

    let config = load_hierarchical_config(None, None)?;

    // Should use default values
    assert_eq!(config.training.total_iterations, 15_000);
    assert_eq!(config.training.image_size, 512);
    assert_eq!(config.training.views_per_step, 4);
    assert_eq!(config.device.gpu_index, 0);
    assert_eq!(config.output.checkpoint_interval, 1_000);

    Ok(())
}

#[test]
#[serial]
fn test_env_var_overrides_defaults() -> Result<()> {
    cleanup_env_vars();

    env::set_var("OXIGAF_TOTAL_ITERATIONS", "5000");
    env::set_var("OXIGAF_IMAGE_SIZE", "1024");
    env::set_var("OXIGAF_DEVICE_GPU_INDEX", "1");

    let config = load_hierarchical_config(None, None)?;

    assert_eq!(config.training.total_iterations, 5000);
    assert_eq!(config.training.image_size, 1024);
    assert_eq!(config.device.gpu_index, 1);

    cleanup_env_vars();
    Ok(())
}

#[test]
#[serial]
fn test_config_file_overrides_defaults() -> Result<()> {
    cleanup_env_vars();
    let test_dir = get_temp_test_dir("config_file_test");

    let config_path = test_dir.join("test_config.toml");
    let config_content = r#"
[training]
total_iterations = 8000
image_size = 768

[output]
checkpoint_interval = 500
"#;
    fs::write(&config_path, config_content)?;

    let config = load_hierarchical_config(Some(&config_path), None)?;

    assert_eq!(config.training.total_iterations, 8000);
    assert_eq!(config.training.image_size, 768);
    assert_eq!(config.output.checkpoint_interval, 500);
    // Other values should be defaults
    assert_eq!(config.training.views_per_step, 4);

    cleanup_test_dir(&test_dir);
    Ok(())
}

#[test]
#[serial]
fn test_env_vars_override_config_file() -> Result<()> {
    cleanup_env_vars();
    let test_dir = get_temp_test_dir("env_override_file_test");

    let config_path = test_dir.join("test_config.toml");
    let config_content = r#"
[training]
total_iterations = 8000
image_size = 768
"#;
    fs::write(&config_path, config_content)?;

    // Set env var to override config file
    env::set_var("OXIGAF_TOTAL_ITERATIONS", "10000");

    let config = load_hierarchical_config(Some(&config_path), None)?;

    // Env var should win
    assert_eq!(config.training.total_iterations, 10000);
    // File value should still apply for non-overridden fields
    assert_eq!(config.training.image_size, 768);

    cleanup_env_vars();
    cleanup_test_dir(&test_dir);
    Ok(())
}

#[test]
#[serial]
fn test_cli_args_override_env_vars() -> Result<()> {
    cleanup_env_vars();

    env::set_var("OXIGAF_TOTAL_ITERATIONS", "5000");
    env::set_var("OXIGAF_IMAGE_SIZE", "1024");

    let mut cli_override = ProjectConfig::default();
    cli_override.training.total_iterations = 12000;

    let config = load_hierarchical_config(None, Some(&cli_override))?;

    // CLI arg should win
    assert_eq!(config.training.total_iterations, 12000);
    // Env var should still apply for non-overridden fields
    assert_eq!(config.training.image_size, 1024);

    cleanup_env_vars();
    Ok(())
}

#[test]
#[serial]
fn test_cli_args_override_everything() -> Result<()> {
    cleanup_env_vars();
    let test_dir = get_temp_test_dir("cli_override_all_test");

    let config_path = test_dir.join("test_config.toml");
    let config_content = r#"
[training]
total_iterations = 8000
image_size = 768
"#;
    fs::write(&config_path, config_content)?;

    env::set_var("OXIGAF_TOTAL_ITERATIONS", "10000");
    env::set_var("OXIGAF_IMAGE_SIZE", "1024");

    let mut cli_override = ProjectConfig::default();
    // Use values different from defaults to ensure merge works correctly
    cli_override.training.total_iterations = 20000; // default is 15000
    cli_override.training.image_size = 2048; // default is 512

    let config = load_hierarchical_config(Some(&config_path), Some(&cli_override))?;

    // CLI args should have highest priority
    assert_eq!(config.training.total_iterations, 20000);
    assert_eq!(config.training.image_size, 2048);

    cleanup_env_vars();
    cleanup_test_dir(&test_dir);
    Ok(())
}

#[test]
#[serial]
fn test_invalid_env_var_returns_error() {
    cleanup_env_vars();

    env::set_var("OXIGAF_TOTAL_ITERATIONS", "not_a_number");

    let result = load_hierarchical_config(None, None);
    assert!(result.is_err());

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Invalid OXIGAF_TOTAL_ITERATIONS"));

    cleanup_env_vars();
}

#[test]
#[serial]
fn test_multiple_env_vars() -> Result<()> {
    cleanup_env_vars();

    env::set_var("OXIGAF_TOTAL_ITERATIONS", "6000");
    env::set_var("OXIGAF_IMAGE_SIZE", "640");
    env::set_var("OXIGAF_POSITION_LR", "0.0003");
    env::set_var("OXIGAF_DEVICE_GPU_INDEX", "2");
    env::set_var("OXIGAF_OUTPUT_CHECKPOINT_INTERVAL", "250");
    env::set_var("OXIGAF_SH_DEGREE", "2");

    let config = load_hierarchical_config(None, None)?;

    assert_eq!(config.training.total_iterations, 6000);
    assert_eq!(config.training.image_size, 640);
    assert!((config.training.optimizer.position_lr - 0.0003).abs() < f32::EPSILON);
    assert_eq!(config.device.gpu_index, 2);
    assert_eq!(config.output.checkpoint_interval, 250);
    assert_eq!(config.training.init.sh_degree, 2);

    cleanup_env_vars();
    Ok(())
}

#[test]
#[serial]
fn test_partial_config_file_merges_with_defaults() -> Result<()> {
    cleanup_env_vars();
    let test_dir = get_temp_test_dir("partial_config_test");

    let config_path = test_dir.join("test_config.toml");
    // Only specify a few fields
    let config_content = r#"
[training]
total_iterations = 9000

[device]
gpu_index = 3
"#;
    fs::write(&config_path, config_content)?;

    let config = load_hierarchical_config(Some(&config_path), None)?;

    // Specified values should be used
    assert_eq!(config.training.total_iterations, 9000);
    assert_eq!(config.device.gpu_index, 3);
    // Unspecified values should be defaults
    assert_eq!(config.training.image_size, 512);
    assert_eq!(config.training.views_per_step, 4);
    assert_eq!(config.output.checkpoint_interval, 1_000);

    cleanup_test_dir(&test_dir);
    Ok(())
}

#[test]
#[serial]
fn test_nonexistent_config_file_uses_defaults() -> Result<()> {
    cleanup_env_vars();
    let test_dir = get_temp_test_dir("nonexistent_config_test");

    let config_path = test_dir.join("nonexistent.toml");

    // Should fail because the file doesn't exist and it's not "oxigaf.toml"
    let result = load_hierarchical_config(Some(&config_path), None);
    assert!(result.is_err());

    cleanup_test_dir(&test_dir);
    Ok(())
}

#[test]
#[serial]
fn test_string_env_vars() -> Result<()> {
    cleanup_env_vars();

    env::set_var("OXIGAF_OUTPUT_EXPORT_FORMAT", "safetensors");
    env::set_var("OXIGAF_DEVICE_BACKEND", "metal");

    let config = load_hierarchical_config(None, None)?;

    assert_eq!(config.output.export_format, "safetensors");
    assert_eq!(config.device.backend, "metal");

    cleanup_env_vars();
    Ok(())
}

#[test]
#[serial]
fn test_validation_catches_invalid_config() {
    cleanup_env_vars();

    env::set_var("OXIGAF_TOTAL_ITERATIONS", "0");

    let result = load_hierarchical_config(None, None);
    assert!(result.is_err());

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("total_iterations must be > 0"));

    cleanup_env_vars();
}

#[test]
#[serial]
fn test_learning_rate_env_vars() -> Result<()> {
    cleanup_env_vars();

    env::set_var("OXIGAF_POSITION_LR", "0.0005");
    env::set_var("OXIGAF_SCALING_LR", "0.01");
    env::set_var("OXIGAF_ROTATION_LR", "0.002");
    env::set_var("OXIGAF_OPACITY_LR", "0.1");
    env::set_var("OXIGAF_SH_LR", "0.005");

    let config = load_hierarchical_config(None, None)?;

    assert!((config.training.optimizer.position_lr - 0.0005).abs() < f32::EPSILON);
    assert!((config.training.optimizer.scale_lr - 0.01).abs() < f32::EPSILON);
    assert!((config.training.optimizer.rotation_lr - 0.002).abs() < f32::EPSILON);
    assert!((config.training.optimizer.opacity_lr - 0.1).abs() < f32::EPSILON);
    assert!((config.training.optimizer.sh_lr - 0.005).abs() < f32::EPSILON);

    cleanup_env_vars();
    Ok(())
}
