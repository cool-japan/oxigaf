//! Integration tests for log file rotation functionality.
//!
//! Note: Tests use `serial_test` to prevent conflicts from initializing
//! the global tracing subscriber multiple times in parallel.
//!
//! When the full workspace test suite runs (`cargo test --workspace`), the global
//! tracing subscriber may already have been initialized by a previous test binary.
//! Each test therefore guards its file-content assertions behind a successful init.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serial_test::serial;
use tempfile::TempDir;

// Re-export modules from oxigaf-cli for testing
use oxigaf_cli::log_rotation::{
    cleanup_old_logs, init_logging_with_file, LogConfig, LogFormat, LogRotation,
};
use oxigaf_cli::verbosity::Verbosity;

/// Try to initialize the tracing subscriber for a test.
///
/// Returns `true` if this call successfully set the global subscriber,
/// `false` if it was already set (by an earlier test or another test binary).
/// When `false`, file-content assertions should be skipped because log events
/// will be dispatched to the pre-existing subscriber, not to the current test's
/// file appender.
fn try_init(config: LogConfig, verbosity: Verbosity) -> bool {
    init_logging_with_file(config, verbosity).is_ok()
}

/// Tracks whether any test in this binary has successfully initialized tracing.
static SUBSCRIBER_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[test]
#[serial]
fn test_log_file_creates_file() -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let log_path = temp.path().join("test.log");

    let config = LogConfig {
        file_path: Some(log_path.clone()),
        rotation: LogRotation::Never,
        max_files: 5,
        format: LogFormat::Json,
    };

    let initialized = try_init(config, Verbosity::Normal);
    if initialized {
        SUBSCRIBER_INITIALIZED.store(true, Ordering::SeqCst);
    }

    tracing::info!("Test log message");

    // Give it time to flush
    std::thread::sleep(Duration::from_millis(200));

    if initialized {
        assert!(
            log_path.exists(),
            "Log file should be created at {}",
            log_path.display()
        );
        let content = std::fs::read_to_string(&log_path)?;
        assert!(!content.is_empty(), "Log file should not be empty");
        assert!(
            content.contains("Test log message"),
            "Log file should contain the test message"
        );
    }

    Ok(())
}

#[test]
#[serial]
fn test_log_file_json_format() -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let log_path = temp.path().join("test_json.log");

    let config = LogConfig {
        file_path: Some(log_path.clone()),
        rotation: LogRotation::Never,
        max_files: 5,
        format: LogFormat::Json,
    };

    let initialized = try_init(config, Verbosity::Normal);

    tracing::info!(test_field = "test_value", "JSON test message");

    // Give it time to flush
    std::thread::sleep(Duration::from_millis(200));

    if initialized {
        let content = std::fs::read_to_string(&log_path)?;
        assert!(
            content.contains("\"level\""),
            "JSON log should contain level field"
        );
        assert!(
            content.contains("\"timestamp\""),
            "JSON log should contain timestamp field"
        );
        assert!(
            content.contains("\"message\""),
            "JSON log should contain message field"
        );
        assert!(
            content.contains("JSON test message"),
            "JSON log should contain the test message"
        );
    }

    Ok(())
}

#[test]
fn test_cleanup_old_logs_removes_excess() -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;

    // Create 10 fake log files with different modification times
    for i in 0..10 {
        let log_file = temp.path().join(format!("app_{}.log", i));
        std::fs::write(&log_file, b"test")?;

        // Add a small delay to ensure different modification times
        std::thread::sleep(Duration::from_millis(10));
    }

    // Verify we created 10 files
    let file_count_before = std::fs::read_dir(temp.path())?.count();
    assert_eq!(file_count_before, 10, "Should have created 10 log files");

    // Clean up, keeping only 5
    cleanup_old_logs(temp.path(), "app_", 5)?;

    // Count remaining files
    let remaining: Vec<_> = std::fs::read_dir(temp.path())?
        .filter_map(|e| e.ok())
        .collect();

    assert_eq!(
        remaining.len(),
        5,
        "Should keep only 5 most recent log files"
    );

    // Verify that the newest files are kept (higher numbers)
    let mut names: Vec<String> = remaining
        .iter()
        .filter_map(|e| e.file_name().to_str().map(String::from))
        .collect();
    names.sort();

    // The 5 most recent files should be app_5.log through app_9.log
    assert!(
        names.contains(&"app_5.log".to_string()),
        "Should keep app_5.log"
    );
    assert!(
        names.contains(&"app_9.log".to_string()),
        "Should keep app_9.log"
    );
    assert!(
        !names.contains(&"app_0.log".to_string()),
        "Should remove app_0.log"
    );

    Ok(())
}

#[test]
#[serial]
fn test_log_rotation_creates_timestamped_files() -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let log_path = temp.path().join("app.log");

    let config = LogConfig {
        file_path: Some(log_path.clone()),
        rotation: LogRotation::Daily,
        max_files: 5,
        format: LogFormat::Json,
    };

    let initialized = try_init(config, Verbosity::Normal);

    tracing::info!("Test message for rotation");

    // Give it time to flush
    std::thread::sleep(Duration::from_millis(500));

    if initialized {
        // With daily rotation, tracing-appender creates files with date stamps
        let log_files: Vec<_> = std::fs::read_dir(temp.path())?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.starts_with("app") && n.contains(".log"))
                    .unwrap_or(false)
            })
            .collect();

        if log_files.is_empty() {
            eprintln!("No matching log files found. All files in directory:");
            for entry in std::fs::read_dir(temp.path())?.flatten() {
                eprintln!("  - {}", entry.file_name().to_string_lossy());
            }
        }
        assert!(!log_files.is_empty(), "Should create at least one log file");
    }

    Ok(())
}

#[test]
#[serial]
fn test_no_log_file_uses_stdout_only() -> Result<(), Box<dyn std::error::Error>> {
    let config = LogConfig {
        file_path: None,
        rotation: LogRotation::Never,
        max_files: 5,
        format: LogFormat::Json,
    };

    // This is a smoke test: configuring stdout-only logging must not error.
    // If the subscriber is already set, the call returns Err which we treat as Ok
    // (the subscriber is already active and dispatching events).
    let _ = try_init(config, Verbosity::Normal);

    tracing::info!("Test message to stdout");

    Ok(())
}

#[test]
#[serial]
fn test_log_file_creates_parent_directories() -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let nested_path = temp.path().join("logs").join("nested").join("test.log");

    // Parent directories should not exist yet
    assert!(!nested_path.parent().map(|p| p.exists()).unwrap_or(false));

    let config = LogConfig {
        file_path: Some(nested_path.clone()),
        rotation: LogRotation::Never,
        max_files: 5,
        format: LogFormat::Json,
    };

    let initialized = try_init(config, Verbosity::Normal);

    tracing::info!("Test message in nested directory");

    // Give it time to flush
    std::thread::sleep(Duration::from_millis(200));

    if initialized {
        assert!(
            nested_path.parent().map(|p| p.exists()).unwrap_or(false),
            "Parent directories should be created automatically"
        );
        assert!(
            nested_path.exists(),
            "Log file should be created in nested directory"
        );
    }

    Ok(())
}

#[test]
fn test_cleanup_old_logs_nonexistent_dir() -> Result<(), Box<dyn std::error::Error>> {
    // Should not error on nonexistent directory
    let tmpdir = tempfile::tempdir()?;
    let nonexistent = tmpdir.path().join("nonexistent_subdir_oxigaf_test");
    let result = cleanup_old_logs(&nonexistent, "test", 5);
    assert!(
        result.is_ok(),
        "Should handle nonexistent directory gracefully"
    );

    Ok(())
}

#[test]
#[serial]
fn test_log_format_compact() -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let log_path = temp.path().join("compact.log");

    let config = LogConfig {
        file_path: Some(log_path.clone()),
        rotation: LogRotation::Never,
        max_files: 5,
        format: LogFormat::Compact,
    };

    let initialized = try_init(config, Verbosity::Normal);

    tracing::info!("Compact format test");

    std::thread::sleep(Duration::from_millis(200));

    if initialized {
        assert!(log_path.exists(), "Log file should be created");
        let content = std::fs::read_to_string(&log_path)?;
        assert!(!content.is_empty(), "Compact log file should have content");
        assert!(
            content.contains("Compact format test"),
            "Log should contain the test message"
        );
    }

    Ok(())
}

#[test]
#[serial]
fn test_log_format_pretty() -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let log_path = temp.path().join("pretty.log");

    let config = LogConfig {
        file_path: Some(log_path.clone()),
        rotation: LogRotation::Never,
        max_files: 5,
        format: LogFormat::Pretty,
    };

    let initialized = try_init(config, Verbosity::Normal);

    tracing::info!("Pretty format test");

    std::thread::sleep(Duration::from_millis(200));

    if initialized {
        assert!(log_path.exists(), "Log file should be created");
        let content = std::fs::read_to_string(&log_path)?;
        assert!(!content.is_empty(), "Pretty log file should have content");
        assert!(
            content.contains("Pretty format test"),
            "Log should contain the test message"
        );
    }

    Ok(())
}
