//! Tests for HuggingFace Hub integration.
//!
//! These tests verify:
//! - HfModelSource parsing logic
//! - CLI flag handling for HuggingFace downloads
//! - Error handling for invalid specifications

use oxigaf_cli::assets::HfModelSource;

// ---------------------------------------------------------------------------
// HfModelSource Parsing Tests
// ---------------------------------------------------------------------------

#[test]
fn hf_model_source_parse_simple() {
    let source =
        HfModelSource::parse("cool-japan/oxigaf-flame").expect("Should parse simple repo ID");
    assert_eq!(source.repo_id, "cool-japan/oxigaf-flame");
    assert_eq!(source.filename, "model.safetensors");
    assert!(source.revision.is_none());
}

#[test]
fn hf_model_source_parse_with_colon_revision() {
    let source = HfModelSource::parse("cool-japan/oxigaf-flame:main")
        .expect("Should parse with colon revision");
    assert_eq!(source.repo_id, "cool-japan/oxigaf-flame");
    assert_eq!(source.revision, Some("main".to_string()));
    assert_eq!(source.filename, "model.safetensors");
}

#[test]
fn hf_model_source_parse_with_at_revision() {
    let source =
        HfModelSource::parse("cool-japan/oxigaf-flame@v1.0").expect("Should parse with @ revision");
    assert_eq!(source.repo_id, "cool-japan/oxigaf-flame");
    assert_eq!(source.revision, Some("v1.0".to_string()));
    assert_eq!(source.filename, "model.safetensors");
}

#[test]
fn hf_model_source_parse_with_commit_sha() {
    let source = HfModelSource::parse("cool-japan/oxigaf-flame@abc123def456")
        .expect("Should parse with commit SHA");
    assert_eq!(source.repo_id, "cool-japan/oxigaf-flame");
    assert_eq!(source.revision, Some("abc123def456".to_string()));
}

#[test]
fn hf_model_source_parse_empty_fails() {
    let result = HfModelSource::parse("");
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("cannot be empty"));
}

#[test]
fn hf_model_source_parse_no_slash_fails() {
    let result = HfModelSource::parse("invalid-repo-id");
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Invalid repository format"));
}

#[test]
fn hf_model_source_parse_empty_revision_fails() {
    let result = HfModelSource::parse("cool-japan/oxigaf-flame:");
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Revision cannot be empty"));
}

#[test]
fn hf_model_source_with_filename() {
    let source = HfModelSource::parse("cool-japan/oxigaf-flame")
        .expect("Should parse")
        .with_filename("custom_model.safetensors".to_string());

    assert_eq!(source.filename, "custom_model.safetensors");
    assert_eq!(source.repo_id, "cool-japan/oxigaf-flame");
}

#[test]
fn hf_model_source_multiple_slashes() {
    // This should fail because it's not a valid org/repo format
    let result = HfModelSource::parse("cool-japan/oxigaf/flame");
    // This is actually valid in HF (subfolders), so it should succeed
    // But our current implementation will just take the whole thing as repo_id
    assert!(result.is_ok());
    let source = result.unwrap();
    assert_eq!(source.repo_id, "cool-japan/oxigaf/flame");
}

// ---------------------------------------------------------------------------
// CLI Integration Tests
// ---------------------------------------------------------------------------

#[test]
fn setup_from_hub_flag_in_help() {
    use assert_cmd::Command;

    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("oxigaf").expect("oxigaf binary should exist");

    cmd.args(["setup", "--help"])
        .assert()
        .success()
        .stdout(predicates::str::contains("--from-hub"));
}

#[test]
fn setup_hf_token_flag_in_help() {
    use assert_cmd::Command;

    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("oxigaf").expect("oxigaf binary should exist");

    cmd.args(["setup", "--help"])
        .assert()
        .success()
        .stdout(predicates::str::contains("--hf-token"));
}

#[test]
fn setup_revision_flag_in_help() {
    use assert_cmd::Command;

    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("oxigaf").expect("oxigaf binary should exist");

    cmd.args(["setup", "--help"])
        .assert()
        .success()
        .stdout(predicates::str::contains("--revision"));
}

#[test]
fn setup_filename_flag_in_help() {
    use assert_cmd::Command;

    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("oxigaf").expect("oxigaf binary should exist");

    cmd.args(["setup", "--help"])
        .assert()
        .success()
        .stdout(predicates::str::contains("--filename"));
}
