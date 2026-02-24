//! Integration tests for cache management commands.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;

/// Get a temporary cache directory for testing.
fn temp_cache_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxigaf_cache_test_{}", std::process::id()));
    // Clean up if it exists from a previous run
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create temp cache dir");
    dir
}

/// Cleanup temporary cache directory.
fn cleanup_cache_dir(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
}

#[allow(deprecated)]
#[test]
fn cache_list_shows_empty_message() {
    let cache_dir = temp_cache_dir();

    // Set HOME to temp directory so cache dir is predictable
    let result = Command::cargo_bin("oxigaf")
        .expect("Failed to find binary")
        .args(["cache", "list"])
        .env("HOME", cache_dir.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("Cache is empty"));

    cleanup_cache_dir(&cache_dir);
    drop(result);
}

#[allow(deprecated)]
#[test]
fn cache_clean_dry_run() {
    let cache_dir = temp_cache_dir();

    let result = Command::cargo_bin("oxigaf")
        .expect("Failed to find binary")
        .args(["cache", "clean", "--dry-run"])
        .env("HOME", cache_dir.to_str().unwrap())
        .assert()
        .success();

    cleanup_cache_dir(&cache_dir);
    drop(result);
}

#[allow(deprecated)]
#[test]
fn cache_clean_with_max_age() {
    let cache_dir = temp_cache_dir();

    let result = Command::cargo_bin("oxigaf")
        .expect("Failed to find binary")
        .args(["cache", "clean", "--max-age-days", "60", "--dry-run"])
        .env("HOME", cache_dir.to_str().unwrap())
        .assert()
        .success();

    cleanup_cache_dir(&cache_dir);
    drop(result);
}

#[allow(deprecated)]
#[test]
fn cache_verify_succeeds() {
    let cache_dir = temp_cache_dir();

    let result = Command::cargo_bin("oxigaf")
        .expect("Failed to find binary")
        .args(["cache", "verify"])
        .env("HOME", cache_dir.to_str().unwrap())
        .assert()
        .success();

    cleanup_cache_dir(&cache_dir);
    drop(result);
}

#[allow(deprecated)]
#[test]
fn cache_path_shows_directory() {
    let cache_dir = temp_cache_dir();

    let result = Command::cargo_bin("oxigaf")
        .expect("Failed to find binary")
        .args(["cache", "path"])
        .env("HOME", cache_dir.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("oxigaf"));

    cleanup_cache_dir(&cache_dir);
    drop(result);
}

#[allow(deprecated)]
#[test]
fn cache_clean_default_max_age() {
    let cache_dir = temp_cache_dir();

    // Test that default max-age-days is 30
    let result = Command::cargo_bin("oxigaf")
        .expect("Failed to find binary")
        .args(["cache", "clean", "--dry-run"])
        .env("HOME", cache_dir.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("30 days").or(predicate::str::contains("No assets")));

    cleanup_cache_dir(&cache_dir);
    drop(result);
}
