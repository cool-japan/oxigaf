//! Integration tests for JSON output mode.
//!
//! These tests verify that the `--json` flag produces valid, machine-readable
//! JSON output with no extraneous text, suitable for scripting and automation.

use assert_cmd::Command;
use predicates::prelude::*;

// ---------------------------------------------------------------------------
// Test Helpers
// ---------------------------------------------------------------------------

/// Get a Command for the oxigaf binary.
fn oxigaf_cmd() -> Command {
    #[allow(deprecated)]
    Command::cargo_bin("oxigaf").expect("oxigaf binary should exist")
}

/// Parse JSON from stdout and verify it's valid.
fn parse_json_output(output: &[u8]) -> serde_json::Value {
    serde_json::from_slice(output).expect("Output should be valid JSON")
}

// ---------------------------------------------------------------------------
// JSON Mode Tests
// ---------------------------------------------------------------------------

#[test]
fn json_flag_conflicts_with_verbose() {
    // --json and --verbose should conflict
    oxigaf_cmd()
        .args(["--json", "--verbose", "doctor"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
}

#[test]
fn json_doctor_valid_structure() {
    let output = oxigaf_cmd()
        .args(["--json", "doctor"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = parse_json_output(&output);

    // Verify required top-level fields
    assert!(json["version"].is_string(), "version should be a string");
    assert_eq!(json["command"], "doctor", "command should be 'doctor'");
    assert!(json["status"].is_string(), "status should be a string");

    // Verify status is one of the valid values
    let status = json["status"].as_str().unwrap();
    assert!(
        status == "success" || status == "error" || status == "warning",
        "status should be success, error, or warning"
    );

    // Result should be present (doctor returns diagnostics)
    assert!(json["result"].is_object(), "result should be an object");
}

#[test]
fn json_output_no_extraneous_text() {
    let output = oxigaf_cmd()
        .args(["--json", "doctor"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    // Ensure output is pure JSON (no extra text before or after)
    let output_str = String::from_utf8_lossy(&output);

    // Should start with '{' and end with '}'
    let trimmed = output_str.trim();
    assert!(trimmed.starts_with('{'), "JSON output should start with {{");
    assert!(trimmed.ends_with('}'), "JSON output should end with }}");

    // Should be valid JSON
    assert!(
        serde_json::from_str::<serde_json::Value>(trimmed).is_ok(),
        "Output should be valid JSON"
    );
}

#[test]
fn json_doctor_no_progress_output() {
    let output = oxigaf_cmd()
        .args(["--json", "doctor"])
        .assert()
        .success()
        .get_output()
        .clone();

    // Stderr should be empty (no progress bars or info messages)
    let stderr_str = String::from_utf8_lossy(&output.stderr);

    // Should not contain progress indicators
    assert!(
        !stderr_str.contains("✓") && !stderr_str.contains("✅"),
        "stderr should not contain status symbols in JSON mode"
    );

    // Should not contain section headers
    assert!(
        !stderr_str.contains("GPU Configuration") && !stderr_str.contains("Version Information"),
        "stderr should not contain section headers in JSON mode"
    );
}

#[test]
fn json_output_includes_version() {
    let output = oxigaf_cmd()
        .args(["--json", "doctor"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = parse_json_output(&output);

    // Version should match CARGO_PKG_VERSION
    let version = json["version"]
        .as_str()
        .expect("version should be a string");
    assert!(!version.is_empty(), "version should not be empty");

    // Version should follow semver format (basic check)
    assert!(
        version.contains('.'),
        "version should contain dots (semver format)"
    );
}

#[test]
fn json_output_has_correct_command_name() {
    let output = oxigaf_cmd()
        .args(["--json", "doctor"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = parse_json_output(&output);
    assert_eq!(json["command"], "doctor");
}

#[test]
fn json_output_omits_empty_fields() {
    let output = oxigaf_cmd()
        .args(["--json", "doctor"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = parse_json_output(&output);
    let json_str = serde_json::to_string(&json).unwrap();

    // If there are no errors, the errors field should be omitted (not present or empty)
    if json["status"] == "success" {
        // Errors array should either not exist or be omitted due to skip_serializing_if
        assert!(
            !json_str.contains("\"errors\"")
                || json["errors"].as_array().is_none_or(|a| a.is_empty()),
            "errors field should be omitted when empty"
        );
    }
}

#[test]
fn json_completions_command_works_without_json() {
    // Completions command should work normally without --json
    // (it doesn't support JSON output as it outputs shell scripts)
    oxigaf_cmd()
        .args(["completions", "bash"])
        .assert()
        .success();
}

// ---------------------------------------------------------------------------
// Error Handling Tests
// ---------------------------------------------------------------------------

#[test]
fn json_error_output_on_failure() {
    // Try to export a non-existent model file
    let temp_dir = std::env::temp_dir();
    let nonexistent = temp_dir.join("nonexistent_model.ply");
    let output_path = temp_dir.join("output.ply");

    let output = oxigaf_cmd()
        .args([
            "--json",
            "export",
            "--model",
            nonexistent.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .get_output()
        .stdout
        .clone();

    // Should still output valid JSON even on error
    let json = parse_json_output(&output);

    assert_eq!(json["status"], "error", "status should be error on failure");

    // Errors array should be present and non-empty
    let errors = json["errors"]
        .as_array()
        .expect("errors should be an array");
    assert!(
        !errors.is_empty(),
        "errors array should not be empty on failure"
    );

    // At least one error message should be a string
    assert!(errors[0].is_string(), "error messages should be strings");
}

#[test]
fn json_error_contains_useful_message() {
    let temp_dir = std::env::temp_dir();
    let nonexistent = temp_dir.join("nonexistent_model.ply");
    let output_path = temp_dir.join("output.ply");

    let output = oxigaf_cmd()
        .args([
            "--json",
            "export",
            "--model",
            nonexistent.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .get_output()
        .stdout
        .clone();

    let json = parse_json_output(&output);
    let errors = json["errors"]
        .as_array()
        .expect("errors should be an array");
    let error_msg = errors[0].as_str().expect("error should be a string");

    // Error message should mention the problem
    assert!(
        error_msg.contains("not found")
            || error_msg.contains("does not exist")
            || error_msg.contains("No such file")
            || error_msg.contains("Failed to load"),
        "error message should describe the problem: {}",
        error_msg
    );
}

// ---------------------------------------------------------------------------
// Field Presence Tests
// ---------------------------------------------------------------------------

#[test]
fn json_output_always_has_required_fields() {
    let output = oxigaf_cmd()
        .args(["--json", "doctor"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = parse_json_output(&output);

    // These fields should always be present
    assert!(
        json.get("version").is_some(),
        "version field should be present"
    );
    assert!(
        json.get("command").is_some(),
        "command field should be present"
    );
    assert!(
        json.get("status").is_some(),
        "status field should be present"
    );
}

#[test]
fn json_output_optional_fields_can_be_absent() {
    let output = oxigaf_cmd()
        .args(["--json", "doctor"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let json = parse_json_output(&output);

    // These fields are optional and may not be present
    // Just verify we can check for them without panicking
    let _artifacts = json.get("artifacts");
    let _warnings = json.get("warnings");
    let _metadata = json.get("metadata");

    // If they are present, they should be the right type
    if let Some(artifacts) = json.get("artifacts") {
        assert!(
            artifacts.is_array(),
            "artifacts should be an array if present"
        );
    }

    if let Some(warnings) = json.get("warnings") {
        assert!(
            warnings.is_array(),
            "warnings should be an array if present"
        );
    }
}
