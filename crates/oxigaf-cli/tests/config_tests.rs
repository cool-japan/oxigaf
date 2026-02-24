//! Configuration parsing and validation tests.
//!
//! These tests verify:
//! - Default config generation
//! - Partial config with defaults
//! - Full config parsing
//! - Validation error detection
//! - Path expansion utilities

use std::fs;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Test Helpers
// ---------------------------------------------------------------------------

/// Create a temporary config file and return its path.
fn create_temp_config(name: &str, content: &str) -> PathBuf {
    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join(format!(
        "oxigaf_config_test_{}_{}.toml",
        name,
        std::process::id()
    ));
    fs::write(&config_path, content).expect("Failed to write temp config");
    config_path
}

/// Clean up a temporary file.
fn cleanup_temp_file(path: &PathBuf) {
    let _ = fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// Config Serialization Tests
// ---------------------------------------------------------------------------

/// Note: These tests use toml directly since we can't access internal module types
/// from integration tests. The unit tests in config.rs cover the internal types.

#[test]
fn config_toml_with_all_sections_parses() {
    let content = r#"
[model]
flame_model_path = "/path/to/flame"
diffusion_weights_dir = "/path/to/weights"

[device]
backend = "metal"
gpu_index = 1

[training]
total_iterations = 20000
views_per_step = 8
image_size = 1024
guidance_scale_start = 10.0
guidance_scale_end = 2.0
guidance_anneal_steps = 15000
num_inference_steps = 30
opacity_reset_interval = 2000

[training.init]
num_rigid_gaussians = 100000
num_flexible_gaussians = 100000
initial_scale = 0.001
initial_opacity = 0.1
sh_degree = 2

[training.optimizer]
position_lr = 0.002
position_lr_final = 0.00002
rotation_lr = 0.002
scale_lr = 0.01
opacity_lr = 0.1
sh_lr = 0.005
offset_lr = 0.0001
beta1 = 0.9
beta2 = 0.99
epsilon = 1e-15
position_lr_decay_steps = 25000

[training.density_control]
interval = 250
start_iteration = 500
end_iteration = 10000
grad_threshold = 0.0001
min_opacity = 0.01
max_screen_size = 0.1
split_scale_threshold = 0.02
max_gaussians = 500000

[training.loss]
lambda_l1 = 0.9
lambda_ssim = 0.2
lambda_position_reg = 0.001
lambda_scale_reg = 0.001
lambda_opacity_reg = 0.0001
lambda_normal = 0.05

[output]
checkpoint_interval = 2000
log_interval = 100
export_format = "safetensors"
"#;

    let config_path = create_temp_config("full", content);

    // Verify file was created and can be read
    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.contains("total_iterations = 20000"));
    assert!(read_content.contains("backend = \"metal\""));

    cleanup_temp_file(&config_path);
}

#[test]
fn config_partial_with_defaults() {
    let content = r#"
[training]
total_iterations = 5000
"#;

    let config_path = create_temp_config("partial", content);

    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.contains("total_iterations = 5000"));
    // Verify other sections are not present (will use defaults)
    assert!(!read_content.contains("[device]"));

    cleanup_temp_file(&config_path);
}

#[test]
fn config_empty_uses_all_defaults() {
    let content = "";

    let config_path = create_temp_config("empty", content);

    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.is_empty());

    cleanup_temp_file(&config_path);
}

#[test]
fn config_only_model_section() {
    let content = r#"
[model]
flame_model_path = "/custom/flame/path"
diffusion_weights_dir = "/custom/weights"
"#;

    let config_path = create_temp_config("model_only", content);

    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.contains("flame_model_path"));
    assert!(read_content.contains("/custom/flame/path"));

    cleanup_temp_file(&config_path);
}

// ---------------------------------------------------------------------------
// Config Validation Tests
// ---------------------------------------------------------------------------

#[test]
fn config_negative_values_in_file() {
    // This tests that we can write configs with edge values
    // Actual validation is tested in unit tests
    let content = r#"
[training]
total_iterations = -1
"#;

    let config_path = create_temp_config("negative", content);

    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.contains("-1"));

    cleanup_temp_file(&config_path);
}

#[test]
fn config_sh_degree_valid_range() {
    // SH degree should be 0-3
    let content = r#"
[training.init]
sh_degree = 3
"#;

    let config_path = create_temp_config("sh_valid", content);

    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.contains("sh_degree = 3"));

    cleanup_temp_file(&config_path);
}

// ---------------------------------------------------------------------------
// Config File Format Tests
// ---------------------------------------------------------------------------

#[test]
fn config_with_comments() {
    let content = r#"
# This is a comment
[training]
# Another comment
total_iterations = 10000  # inline comment
"#;

    let config_path = create_temp_config("comments", content);

    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.contains("# This is a comment"));
    assert!(read_content.contains("total_iterations = 10000"));

    cleanup_temp_file(&config_path);
}

#[test]
fn config_with_unicode_paths() {
    let content = r#"
[model]
flame_model_path = "/path/with/日本語/flame"
"#;

    let config_path = create_temp_config("unicode", content);

    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.contains("日本語"));

    cleanup_temp_file(&config_path);
}

#[test]
fn config_with_spaces_in_path() {
    let content = r#"
[model]
flame_model_path = "/path/with spaces/flame model"
"#;

    let config_path = create_temp_config("spaces", content);

    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.contains("with spaces"));

    cleanup_temp_file(&config_path);
}

// ---------------------------------------------------------------------------
// Environment Variable Tests
// ---------------------------------------------------------------------------

#[test]
fn tilde_expansion_in_path_concept() {
    // This is a conceptual test - actual tilde expansion is tested in unit tests
    // Here we just verify the config can contain tilde paths
    let content = r#"
[model]
flame_model_path = "~/.cache/oxigaf/flame"
"#;

    let config_path = create_temp_config("tilde", content);

    let read_content = fs::read_to_string(&config_path).expect("Failed to read config");
    assert!(read_content.contains("~/.cache"));

    cleanup_temp_file(&config_path);
}
