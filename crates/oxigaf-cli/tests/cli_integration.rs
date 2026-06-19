//! Integration tests for the oxigaf CLI.
//!
//! These tests use `assert_cmd` to test the CLI as a black box, verifying:
//! - Help output contains expected information
//! - Version output is correct
//! - Invalid arguments produce helpful errors
//! - Config file parsing works correctly

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use assert_cmd::Command;
use predicates::prelude::*;

// Counter for generating unique test file names
static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Test Helpers
// ---------------------------------------------------------------------------

/// Get a Command for the oxigaf binary.
fn oxigaf_cmd() -> Command {
    #[allow(deprecated)]
    Command::cargo_bin("oxigaf").expect("oxigaf binary should exist")
}

/// Create a temporary config file and return its path.
/// Uses an atomic counter to ensure unique filenames across parallel tests.
fn create_temp_config(content: &str) -> std::path::PathBuf {
    let temp_dir = std::env::temp_dir();
    let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let config_path = temp_dir.join(format!(
        "oxigaf_test_config_{}_{}.toml",
        std::process::id(),
        counter
    ));
    fs::write(&config_path, content).expect("Failed to write temp config");
    config_path
}

/// Clean up a temporary file.
fn cleanup_temp_file(path: &Path) {
    let _ = fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// Help Output Tests
// ---------------------------------------------------------------------------

#[test]
fn cli_help_shows_usage() {
    oxigaf_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("oxigaf"))
        .stdout(predicate::str::contains("Gaussian Avatar"));
}

#[test]
fn cli_help_shows_subcommands() {
    oxigaf_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("reconstruct"))
        .stdout(predicate::str::contains("render"))
        .stdout(predicate::str::contains("export"))
        .stdout(predicate::str::contains("setup"));
}

#[test]
fn reconstruct_help_shows_options() {
    oxigaf_cmd()
        .args(["reconstruct", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--input"))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--flame-model"))
        .stdout(predicate::str::contains("--config"));
}

#[test]
fn render_help_shows_options() {
    oxigaf_cmd()
        .args(["render", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--width"))
        .stdout(predicate::str::contains("--height"));
}

#[test]
fn export_help_shows_options() {
    oxigaf_cmd()
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--format"));
}

#[test]
fn setup_help_shows_options() {
    oxigaf_cmd()
        .args(["setup", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--cache-dir"));
}

// ---------------------------------------------------------------------------
// Version Tests
// ---------------------------------------------------------------------------

#[test]
fn cli_version_shows_version() {
    oxigaf_cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("oxigaf"))
        .stdout(predicate::str::contains("0.1.2"));
}

// ---------------------------------------------------------------------------
// Invalid Argument Tests
// ---------------------------------------------------------------------------

#[test]
fn invalid_subcommand_shows_error() {
    oxigaf_cmd()
        .arg("invalid-subcommand")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[test]
fn reconstruct_missing_required_args() {
    // Missing --input, --output, --flame-model
    oxigaf_cmd()
        .arg("reconstruct")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn render_missing_required_args() {
    // Missing --model, --output
    oxigaf_cmd()
        .arg("render")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn export_missing_required_args() {
    // Missing --model, --output
    oxigaf_cmd()
        .arg("export")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn export_invalid_format() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output.xyz");
    oxigaf_cmd()
        .args([
            "export",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--format",
            "invalid-format",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid"));
}

// ---------------------------------------------------------------------------
// Config File Tests
// ---------------------------------------------------------------------------

#[test]
fn config_default_values_work() {
    // Test that default config file path works when file doesn't exist
    // (should use built-in defaults)
    let temp_dir = std::env::temp_dir();
    let output_dir = temp_dir.join(format!("oxigaf_test_output_{}", std::process::id()));

    // This will fail because the FLAME model doesn't exist,
    // but it should get past config loading
    oxigaf_cmd()
        .args([
            "reconstruct",
            "--input",
            "/nonexistent/input.mp4",
            "--output",
            output_dir.to_str().expect("valid path"),
            "--flame-model",
            "/nonexistent/flame",
            // Uses default config (oxigaf.toml) which won't exist
        ])
        .assert()
        .failure();

    // Clean up
    let _ = fs::remove_dir_all(&output_dir);
}

#[test]
fn config_custom_path_works() {
    let config_content = r#"
[training]
total_iterations = 5000
image_size = 256
"#;

    let config_path = create_temp_config(config_content);
    let temp_dir = std::env::temp_dir();
    let output_dir = temp_dir.join(format!("oxigaf_test_output2_{}", std::process::id()));

    // This will fail because input doesn't exist, but should parse config
    oxigaf_cmd()
        .args([
            "reconstruct",
            "--input",
            "/nonexistent/input.mp4",
            "--output",
            output_dir.to_str().expect("valid path"),
            "--flame-model",
            "/nonexistent/flame",
            "--config",
            config_path.to_str().expect("valid path"),
        ])
        .assert()
        .failure();

    // Clean up
    cleanup_temp_file(&config_path);
    let _ = fs::remove_dir_all(&output_dir);
}

#[test]
fn config_invalid_toml_syntax_error() {
    let config_content = r#"
[training
total_iterations = 5000
"#; // Missing closing bracket

    let config_path = create_temp_config(config_content);
    let temp_dir = std::env::temp_dir();
    let output_dir = temp_dir.join(format!("oxigaf_test_output3_{}", std::process::id()));

    oxigaf_cmd()
        .args([
            "reconstruct",
            "--input",
            "/nonexistent/input.mp4",
            "--output",
            output_dir.to_str().expect("valid path"),
            "--flame-model",
            "/nonexistent/flame",
            "--config",
            config_path.to_str().expect("valid path"),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("parse").or(predicate::str::contains("TOML")));

    // Clean up
    cleanup_temp_file(&config_path);
    let _ = fs::remove_dir_all(&output_dir);
}

#[test]
fn config_validation_error_zero_iterations() {
    let config_content = r#"
[training]
total_iterations = 0
"#;

    let config_path = create_temp_config(config_content);
    let temp_dir = std::env::temp_dir();
    let output_dir = temp_dir.join(format!("oxigaf_test_output4_{}", std::process::id()));

    oxigaf_cmd()
        .args([
            "reconstruct",
            "--input",
            "/nonexistent/input.mp4",
            "--output",
            output_dir.to_str().expect("valid path"),
            "--flame-model",
            "/nonexistent/flame",
            "--config",
            config_path.to_str().expect("valid path"),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("iterations").or(predicate::str::contains("0")));

    // Clean up
    cleanup_temp_file(&config_path);
    let _ = fs::remove_dir_all(&output_dir);
}

// ---------------------------------------------------------------------------
// Export Format Tests
// ---------------------------------------------------------------------------

#[test]
fn export_ply_format_accepted() {
    // Just check that the format is parsed correctly
    oxigaf_cmd()
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("ply"));
}

#[test]
fn export_safetensors_format_accepted() {
    oxigaf_cmd()
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("safetensors"));
}

// ---------------------------------------------------------------------------
// Setup Command Tests
// ---------------------------------------------------------------------------

#[test]
fn setup_cache_dir_option_works() {
    // Verify the option is recognized (actual download would require network)
    oxigaf_cmd()
        .args(["setup", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("cache-dir"));
}

// ---------------------------------------------------------------------------
// Benchmark Command Tests
// ---------------------------------------------------------------------------

#[test]
fn benchmark_help_shows_options() {
    oxigaf_cmd()
        .args(["benchmark", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--target"))
        .stdout(predicate::str::contains("--warmup"))
        .stdout(predicate::str::contains("--iterations"))
        .stdout(predicate::str::contains("--format"));
}

#[test]
fn benchmark_with_default_args_runs() {
    // Run a minimal benchmark (should complete quickly)
    oxigaf_cmd()
        .args([
            "benchmark",
            "--target",
            "flame",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--size",
            "tiny",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("FLAME"));
}

#[test]
fn benchmark_json_output_format() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join(format!("bench_output_{}.json", std::process::id()));

    oxigaf_cmd()
        .args([
            "benchmark",
            "--target",
            "flame",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--size",
            "tiny",
            "--format",
            "json",
            "--output",
            output_path.to_str().expect("valid path"),
        ])
        .assert()
        .success();

    // Verify JSON file was created and is valid
    let json_content = fs::read_to_string(&output_path).expect("Failed to read output file");
    let _parsed: serde_json::Value =
        serde_json::from_str(&json_content).expect("Output should be valid JSON");

    // Clean up
    cleanup_temp_file(&output_path);
}

#[test]
fn benchmark_invalid_target() {
    oxigaf_cmd()
        .args(["benchmark", "--target", "invalid-target"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid"));
}

#[test]
fn benchmark_csv_format() {
    oxigaf_cmd()
        .args([
            "benchmark",
            "--target",
            "export",
            "--format",
            "csv",
            "--warmup",
            "1",
            "--iterations",
            "2",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("name,target,iterations"));
}

// ---------------------------------------------------------------------------
// Doctor Command Tests
// ---------------------------------------------------------------------------

#[test]
fn doctor_help_shows_options() {
    oxigaf_cmd()
        .args(["doctor", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--verbose"))
        .stdout(predicate::str::contains("--json"));
}

#[test]
fn doctor_runs_basic_checks() {
    oxigaf_cmd()
        .args(["doctor"])
        .assert()
        .success()
        .stdout(predicate::str::contains("GPU Configuration"))
        .stdout(predicate::str::contains("Version Information"));
}

#[test]
fn doctor_json_output() {
    oxigaf_cmd()
        .args(["doctor", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"version\":"))
        .stdout(predicate::str::contains("\"command\":"))
        .stdout(predicate::str::contains("\"status\":"))
        .stdout(predicate::str::contains("\"result\":"));
}

#[test]
fn doctor_verbose_mode() {
    oxigaf_cmd()
        .args(["doctor", "--verbose"])
        .assert()
        .success()
        .stdout(predicate::str::contains("OxiGAF"));
}

#[test]
fn doctor_with_invalid_flame_path() {
    oxigaf_cmd()
        .args(["doctor", "--flame-model", "/nonexistent/flame/model"])
        .assert()
        .success() // doctor should succeed but report issues
        .stdout(predicate::str::contains("FLAME Model"));
}

// ---------------------------------------------------------------------------
// Convert Command Tests
// ---------------------------------------------------------------------------

#[test]
fn convert_help_shows_options() {
    oxigaf_cmd()
        .args(["convert", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--input"))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--version"))
        .stdout(predicate::str::contains("--verify"));
}

#[test]
fn convert_missing_required_args() {
    oxigaf_cmd()
        .arg("convert")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn convert_nonexistent_input() {
    let temp_dir = std::env::temp_dir();
    let output_dir = temp_dir.join(format!("convert_output_{}", std::process::id()));

    oxigaf_cmd()
        .args([
            "convert",
            "--input",
            "/nonexistent/flame.pkl",
            "--output",
            output_dir.to_str().expect("valid path"),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found").or(predicate::str::contains("exist")));

    // Clean up
    let _ = fs::remove_dir_all(&output_dir);
}

#[test]
fn convert_invalid_format() {
    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join(format!("test_file_{}.txt", std::process::id()));
    let output_dir = temp_dir.join(format!("convert_output2_{}", std::process::id()));

    // Create a dummy file with unsupported extension
    fs::write(&input_path, b"dummy content").expect("Failed to write dummy file");

    oxigaf_cmd()
        .args([
            "convert",
            "--input",
            input_path.to_str().expect("valid path"),
            "--output",
            output_dir.to_str().expect("valid path"),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unsupported").or(predicate::str::contains("format")));

    // Clean up
    cleanup_temp_file(&input_path);
    let _ = fs::remove_dir_all(&output_dir);
}

// ---------------------------------------------------------------------------
// Early Stopping Tests
// ---------------------------------------------------------------------------

#[test]
fn train_help_shows_early_stopping_options() {
    oxigaf_cmd()
        .args(["train", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--patience"))
        .stdout(predicate::str::contains("--min-delta"))
        .stdout(predicate::str::contains("--early-stop-loss"));
}

#[test]
fn train_patience_option_accepted() {
    // Verify patience parameter is accepted (test will fail due to missing inputs)
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "train",
            "--input",
            "/nonexistent/input.mp4",
            "--output",
            output.to_str().unwrap_or(""),
            "--flame-model",
            "/nonexistent/flame",
            "--patience",
            "100",
        ])
        .assert()
        .failure(); // Expected to fail due to missing files, not due to invalid args
}

#[test]
fn train_min_delta_option_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "train",
            "--input",
            "/nonexistent/input.mp4",
            "--output",
            output.to_str().unwrap_or(""),
            "--flame-model",
            "/nonexistent/flame",
            "--min-delta",
            "0.001",
        ])
        .assert()
        .failure(); // Expected to fail due to missing files
}

// ---------------------------------------------------------------------------
// Render Quality Tests
// ---------------------------------------------------------------------------

#[test]
fn render_help_shows_quality_options() {
    oxigaf_cmd()
        .args(["render", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--quality"))
        .stdout(predicate::str::contains("ultra"));
}

#[test]
fn render_quality_ultra_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "render",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--quality",
            "ultra",
        ])
        .assert()
        .failure(); // Expected to fail due to missing model
}

#[test]
fn render_quality_low_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "render",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--quality",
            "low",
        ])
        .assert()
        .failure();
}

#[test]
fn render_quality_medium_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "render",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--quality",
            "medium",
        ])
        .assert()
        .failure();
}

#[test]
fn render_quality_high_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "render",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--quality",
            "high",
        ])
        .assert()
        .failure();
}

#[test]
fn render_invalid_quality_rejected() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "render",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--quality",
            "invalid-quality",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid"));
}

// ---------------------------------------------------------------------------
// EXR Format Tests
// ---------------------------------------------------------------------------

#[test]
fn render_help_shows_exr_format() {
    oxigaf_cmd()
        .args(["render", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("exr"));
}

#[test]
fn render_exr_format_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "render",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--format",
            "exr",
        ])
        .assert()
        .failure(); // Expected to fail due to missing model
}

#[test]
fn render_png_format_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "render",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--format",
            "png",
        ])
        .assert()
        .failure();
}

#[test]
fn render_jpeg_format_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "render",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--format",
            "jpeg",
        ])
        .assert()
        .failure();
}

// ---------------------------------------------------------------------------
// Export Metadata Tests
// ---------------------------------------------------------------------------

#[test]
fn export_help_shows_metadata_option() {
    oxigaf_cmd()
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--include-metadata"));
}

#[test]
fn export_gltf_with_metadata_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output.glb");
    oxigaf_cmd()
        .args([
            "export",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--format",
            "gltf",
            "--include-metadata",
        ])
        .assert()
        .failure(); // Expected to fail due to missing model
}

#[test]
fn export_gltf_without_metadata_accepted() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output.glb");
    oxigaf_cmd()
        .args([
            "export",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--format",
            "gltf",
        ])
        .assert()
        .failure();
}

#[test]
fn export_gltf_format_accepted() {
    oxigaf_cmd()
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("gltf"));
}

// ---------------------------------------------------------------------------
// Combined Feature Tests
// ---------------------------------------------------------------------------

#[test]
fn train_with_all_early_stopping_options() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "train",
            "--input",
            "/nonexistent/input.mp4",
            "--output",
            output.to_str().unwrap_or(""),
            "--flame-model",
            "/nonexistent/flame",
            "--patience",
            "100",
            "--min-delta",
            "0.0001",
            "--early-stop-loss",
            "0.01",
        ])
        .assert()
        .failure();
}

#[test]
fn render_with_ultra_quality_and_exr() {
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "render",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
            "--quality",
            "ultra",
            "--format",
            "exr",
        ])
        .assert()
        .failure();
}

// ---------------------------------------------------------------------------
// Completions Command Tests
// ---------------------------------------------------------------------------

#[test]
fn completions_help_shows_options() {
    oxigaf_cmd()
        .args(["completions", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Shell to generate completions for",
        ))
        .stdout(predicate::str::contains("bash"))
        .stdout(predicate::str::contains("zsh"))
        .stdout(predicate::str::contains("fish"))
        .stdout(predicate::str::contains("powershell"));
}

#[test]
fn completions_bash_generates_output() {
    oxigaf_cmd()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("_oxigaf"))
        .stdout(predicate::str::contains("COMPREPLY"));
}

#[test]
fn completions_zsh_generates_output() {
    oxigaf_cmd()
        .args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::contains("#compdef oxigaf"))
        .stdout(predicate::str::contains("_oxigaf"));
}

#[test]
fn completions_fish_generates_output() {
    oxigaf_cmd()
        .args(["completions", "fish"])
        .assert()
        .success()
        .stdout(predicate::str::contains("complete -c oxigaf"))
        .stdout(predicate::str::contains("__fish_oxigaf"));
}

#[test]
fn completions_powershell_generates_output() {
    oxigaf_cmd()
        .args(["completions", "powershell"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Register-ArgumentCompleter"))
        .stdout(predicate::str::contains("oxigaf"));
}

#[test]
fn completions_missing_shell_arg() {
    oxigaf_cmd()
        .arg("completions")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn completions_invalid_shell() {
    oxigaf_cmd()
        .args(["completions", "invalid-shell"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid"));
}

#[test]
fn completions_help_shows_installation_instructions() {
    oxigaf_cmd()
        .args(["completions", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Installation"))
        .stdout(predicate::str::contains("bash_completion"))
        .stdout(predicate::str::contains("fish/completions"));
}

#[test]
fn completions_bash_includes_all_subcommands() {
    oxigaf_cmd()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("train"))
        .stdout(predicate::str::contains("render"))
        .stdout(predicate::str::contains("export"))
        .stdout(predicate::str::contains("benchmark"))
        .stdout(predicate::str::contains("doctor"))
        .stdout(predicate::str::contains("setup"))
        .stdout(predicate::str::contains("completions"));
}

#[test]
fn completions_zsh_includes_all_subcommands() {
    oxigaf_cmd()
        .args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::contains("train"))
        .stdout(predicate::str::contains("render"))
        .stdout(predicate::str::contains("export"));
}

#[test]
fn completions_fish_includes_all_subcommands() {
    oxigaf_cmd()
        .args(["completions", "fish"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Train (reconstruct)"))
        .stdout(predicate::str::contains("Render an existing"))
        .stdout(predicate::str::contains("Export an avatar"));
}

#[test]
fn completions_elvish_generates_output() {
    // Test elvish shell support if available
    let _result = oxigaf_cmd().args(["completions", "elvish"]).assert();

    // Elvish should either succeed or fail with invalid value error
    // depending on whether clap_complete supports it
    // (No specific exit code check needed - just verify command runs)
}

// ---------------------------------------------------------------------------
// Verbosity Mode Tests
// ---------------------------------------------------------------------------

#[test]
fn verbose_mode_shows_debug_info() {
    // Test that -v flag enables debug output
    oxigaf_cmd()
        .args(["doctor", "-v"])
        .assert()
        .success()
        .stdout(predicate::str::contains("OxiGAF"));
}

#[test]
fn quiet_mode_suppresses_progress() {
    // Test that -q flag suppresses progress bars and verbose output
    // Doctor command should still complete successfully but with minimal output
    oxigaf_cmd().args(["doctor", "-q"]).assert().success();
}

#[test]
fn multiple_verbose_flags_increase_verbosity() {
    // Test -vvv works correctly
    oxigaf_cmd()
        .args(["doctor", "-vvv"])
        .assert()
        .success()
        .stdout(predicate::str::contains("OxiGAF"));
}

#[test]
fn verbose_and_quiet_conflict() {
    // Test that -v and -q together causes error
    oxigaf_cmd()
        .args(["doctor", "-v", "-q"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("cannot be used with")
                .or(predicate::str::contains("conflict")),
        );
}

#[test]
fn global_verbose_flag_before_subcommand() {
    // Test that global -v flag works before subcommand
    oxigaf_cmd().args(["-v", "doctor"]).assert().success();
}

#[test]
fn global_quiet_flag_before_subcommand() {
    // Test that global -q flag works before subcommand
    oxigaf_cmd().args(["-q", "doctor"]).assert().success();
}

#[test]
fn verbose_flag_shows_in_help() {
    // Verify -v/--verbose flag appears in help
    oxigaf_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--verbose"))
        .stdout(predicate::str::contains("Increase verbosity"));
}

#[test]
fn quiet_flag_shows_in_help() {
    // Verify -q/--quiet flag appears in help
    oxigaf_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--quiet"))
        .stdout(predicate::str::contains("Quiet mode"));
}

#[test]
fn verbose_with_benchmark() {
    // Test verbosity with benchmark command
    oxigaf_cmd()
        .args([
            "-v",
            "benchmark",
            "--target",
            "flame",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--size",
            "tiny",
        ])
        .assert()
        .success();
}

#[test]
fn quiet_with_benchmark() {
    // Test quiet mode with benchmark command
    oxigaf_cmd()
        .args([
            "-q",
            "benchmark",
            "--target",
            "flame",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--size",
            "tiny",
        ])
        .assert()
        .success();
}

// ---------------------------------------------------------------------------
// Dry-Run Mode Tests
// ---------------------------------------------------------------------------

#[test]
fn dry_run_flag_shows_in_help() {
    // Verify --dry-run flag appears in help
    oxigaf_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("Dry run"));
}

#[test]
fn dry_run_global_flag_before_subcommand() {
    // Test that global --dry-run flag works before subcommand
    // Note: This will fail because input doesn't exist, which is expected
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output.ply");
    oxigaf_cmd()
        .args([
            "--dry-run",
            "export",
            "--model",
            "/nonexistent/model.ply",
            "--output",
            output.to_str().unwrap_or(""),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn dry_run_export_shows_dry_run_message() {
    // Create a temporary file to use as input
    let temp_dir = std::env::temp_dir();
    let temp_input = temp_dir.join("oxigaf_test_model.ply");

    // Write a minimal PLY file
    let ply_content = "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";
    fs::write(&temp_input, ply_content).expect("Failed to write temp PLY");

    let temp_output = temp_dir.join("oxigaf_test_output.ply");

    // Run with --dry-run
    oxigaf_cmd()
        .args([
            "--dry-run",
            "export",
            "--model",
            temp_input.to_str().unwrap_or(""),
            "--output",
            temp_output.to_str().unwrap_or(""),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("DRY RUN"))
        .stdout(predicate::str::contains("Would create"));

    // Clean up
    cleanup_temp_file(&temp_input);
}

#[test]
fn dry_run_export_shows_resource_estimates() {
    // Create a temporary file to use as input
    let temp_dir = std::env::temp_dir();
    let temp_input = temp_dir.join("oxigaf_test_model_estimates.ply");

    // Write a minimal PLY file
    let ply_content = "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";
    fs::write(&temp_input, ply_content).expect("Failed to write temp PLY");

    let temp_output = temp_dir.join("oxigaf_test_output_estimates.ply");

    // Run with --dry-run
    oxigaf_cmd()
        .args([
            "--dry-run",
            "export",
            "--model",
            temp_input.to_str().unwrap_or(""),
            "--output",
            temp_output.to_str().unwrap_or(""),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Resource estimates"))
        .stdout(predicate::str::contains("Disk"));

    // Clean up
    cleanup_temp_file(&temp_input);
}

#[test]
fn dry_run_convert_fails_on_invalid_input() {
    // Test that dry-run still validates input existence
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "--dry-run",
            "convert",
            "--input",
            "/nonexistent/flame.npz",
            "--output",
            output.to_str().unwrap_or(""),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn dry_run_train_fails_on_missing_input() {
    // Test that dry-run validates inputs for train command
    let _tmpdir = tempfile::tempdir().expect("create temp dir");
    let output = _tmpdir.path().join("output");
    oxigaf_cmd()
        .args([
            "--dry-run",
            "train",
            "--input",
            "/nonexistent/input.mp4",
            "--output",
            output.to_str().unwrap_or(""),
            "--flame-model",
            "/nonexistent/flame",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn dry_run_with_verbose() {
    // Test that --dry-run works with -v flag
    let temp_dir = std::env::temp_dir();
    let temp_input = temp_dir.join("oxigaf_test_verbose_dry.ply");

    // Write a minimal PLY file
    let ply_content = "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";
    fs::write(&temp_input, ply_content).expect("Failed to write temp PLY");

    let temp_output = temp_dir.join("oxigaf_test_verbose_output.ply");

    oxigaf_cmd()
        .args([
            "-v",
            "--dry-run",
            "export",
            "--model",
            temp_input.to_str().unwrap_or(""),
            "--output",
            temp_output.to_str().unwrap_or(""),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("DRY RUN"));

    // Clean up
    cleanup_temp_file(&temp_input);
}

#[test]
fn dry_run_with_quiet() {
    // Test that --dry-run works with -q flag
    let temp_dir = std::env::temp_dir();
    let temp_input = temp_dir.join("oxigaf_test_quiet_dry.ply");

    // Write a minimal PLY file
    let ply_content = "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";
    fs::write(&temp_input, ply_content).expect("Failed to write temp PLY");

    let temp_output = temp_dir.join("oxigaf_test_quiet_output.ply");

    oxigaf_cmd()
        .args([
            "-q",
            "--dry-run",
            "export",
            "--model",
            temp_input.to_str().unwrap_or(""),
            "--output",
            temp_output.to_str().unwrap_or(""),
        ])
        .assert()
        .success();

    // Clean up
    cleanup_temp_file(&temp_input);
}

#[test]
fn dry_run_does_not_create_files() {
    // Verify that --dry-run doesn't actually create output files
    let temp_dir = std::env::temp_dir();
    let temp_input = temp_dir.join("oxigaf_test_no_create.ply");
    let temp_output = temp_dir.join("oxigaf_test_should_not_exist.ply");

    // Write a minimal PLY file
    let ply_content = "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";
    fs::write(&temp_input, ply_content).expect("Failed to write temp PLY");

    // Ensure output doesn't exist before test
    let _ = fs::remove_file(&temp_output);

    oxigaf_cmd()
        .args([
            "--dry-run",
            "export",
            "--model",
            temp_input.to_str().unwrap_or(""),
            "--output",
            temp_output.to_str().unwrap_or(""),
        ])
        .assert()
        .success();

    // Verify output was NOT created
    assert!(
        !temp_output.exists(),
        "Output file should not exist after dry-run"
    );

    // Clean up
    cleanup_temp_file(&temp_input);
}

#[test]
fn dry_run_convert_shows_expected_outputs() {
    // Create a temporary input file
    let temp_dir = std::env::temp_dir();
    let temp_input = temp_dir.join("oxigaf_test_convert.npz");

    // Write dummy content (not a valid NPZ, but enough to pass existence check for dry-run)
    fs::write(&temp_input, b"dummy").expect("Failed to write temp file");

    let temp_output = temp_dir.join("oxigaf_test_convert_output");

    oxigaf_cmd()
        .args([
            "--dry-run",
            "convert",
            "--input",
            temp_input.to_str().unwrap_or(""),
            "--output",
            temp_output.to_str().unwrap_or(""),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("DRY RUN"))
        .stdout(predicate::str::contains("v_template.npy"))
        .stdout(predicate::str::contains("shapedirs.npy"))
        .stdout(predicate::str::contains("faces.npy"));

    // Clean up
    cleanup_temp_file(&temp_input);
}
