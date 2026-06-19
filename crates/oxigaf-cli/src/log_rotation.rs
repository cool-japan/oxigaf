//! Log file rotation and structured logging.
//!
//! Provides structured logging to files with rotation support.
//! Supports JSON Lines format, timestamps, log levels, and automatic
//! cleanup of old log files.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

use crate::verbosity::Verbosity;

/// Configuration for log file output and rotation.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Optional path to the log file.
    pub file_path: Option<PathBuf>,
    /// Log rotation strategy.
    pub rotation: LogRotation,
    /// Maximum number of log files to keep.
    pub max_files: usize,
    /// Log format for file output.
    pub format: LogFormat,
}

/// Log rotation strategies.
#[derive(Debug, Clone, Copy)]
pub enum LogRotation {
    /// Never rotate (single file).
    Never,
    /// Rotate hourly.
    Hourly,
    /// Rotate daily.
    Daily,
    /// Rotate by size (bytes).
    ///
    /// Note: tracing-appender doesn't support size-based rotation directly,
    /// so this is approximated with daily rotation.
    #[allow(dead_code)]
    Size(u64),
}

/// Log output format.
#[derive(Debug, Clone, Copy)]
pub enum LogFormat {
    /// JSON Lines format (recommended for parsing).
    Json,
    /// Pretty-printed format (human-readable).
    Pretty,
    /// Compact format (minimal whitespace).
    Compact,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            file_path: None,
            rotation: LogRotation::Size(10 * 1024 * 1024), // 10MB
            max_files: 5,
            format: LogFormat::Json,
        }
    }
}

/// Initialize logging with optional file output and rotation.
///
/// Sets up dual logging: file (if specified) and stdout.
/// File logs use the specified format without ANSI colors.
/// Console logs use standard format with colors.
///
/// # Arguments
///
/// * `log_config` - Configuration for log file and rotation
/// * `verbosity` - Verbosity level from CLI flags
///
/// # Errors
///
/// Returns error if:
/// - Log directory cannot be created
/// - File appender cannot be initialized
/// - Subscriber cannot be set as global default
pub fn init_logging_with_file(log_config: LogConfig, verbosity: Verbosity) -> Result<()> {
    use tracing_subscriber::filter::LevelFilter;

    let filter = LevelFilter::from_level(verbosity.tracing_level());
    let env_filter = EnvFilter::from_default_env().add_directive(filter.into());

    if let Some(ref log_path) = log_config.file_path {
        // Ensure parent directory exists
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create log directory: {}", parent.display()))?;
        }

        // Create file appender with rotation
        let file_appender = create_appender(log_path, log_config.rotation)?;

        // Create file layer based on format
        let file_layer = match log_config.format {
            LogFormat::Json => fmt::layer()
                .json()
                .with_writer(file_appender)
                .with_ansi(false)
                .with_filter(env_filter.clone())
                .boxed(),
            LogFormat::Pretty => fmt::layer()
                .pretty()
                .with_writer(file_appender)
                .with_ansi(false)
                .with_filter(env_filter.clone())
                .boxed(),
            LogFormat::Compact => fmt::layer()
                .compact()
                .with_writer(file_appender)
                .with_ansi(false)
                .with_filter(env_filter.clone())
                .boxed(),
        };

        // Create stdout layer
        let stdout_layer = fmt::layer()
            .with_target(verbosity >= Verbosity::Debug)
            .with_file(verbosity >= Verbosity::Debug)
            .with_line_number(verbosity >= Verbosity::Debug)
            .with_filter(env_filter);

        // Combine layers
        tracing_subscriber::registry()
            .with(file_layer)
            .with(stdout_layer)
            .try_init()
            .map_err(|e| anyhow::anyhow!("Failed to initialize tracing subscriber: {}", e))?;

        // Log rotation metadata
        tracing::info!(
            log_file = %log_path.display(),
            rotation = ?log_config.rotation,
            max_files = log_config.max_files,
            "Logging to file with rotation"
        );
    } else {
        // Console-only logging
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_target(verbosity >= Verbosity::Debug)
            .with_file(verbosity >= Verbosity::Debug)
            .with_line_number(verbosity >= Verbosity::Debug)
            .try_init()
            .map_err(|e| anyhow::anyhow!("Failed to initialize tracing subscriber: {}", e))?;
    }

    Ok(())
}

/// Create a rolling file appender based on rotation strategy.
///
/// # Arguments
///
/// * `log_path` - Full path to the log file
/// * `rotation` - Rotation strategy to use
///
/// # Errors
///
/// Returns error if the log path is invalid or the appender cannot be created.
fn create_appender(log_path: &Path, rotation: LogRotation) -> Result<RollingFileAppender> {
    let dir = log_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid log path: no parent directory"))?;

    let filename = log_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid log filename"))?;

    let appender = match rotation {
        LogRotation::Never => RollingFileAppender::new(Rotation::NEVER, dir, filename),
        LogRotation::Hourly => RollingFileAppender::new(Rotation::HOURLY, dir, filename),
        LogRotation::Daily | LogRotation::Size(_) => {
            // tracing-appender doesn't support size-based rotation directly
            // Use daily rotation as approximation
            RollingFileAppender::new(Rotation::DAILY, dir, filename)
        }
    };

    Ok(appender)
}

/// Clean up old log files, keeping only the most recent max_files.
///
/// Sorts log files by modification time and removes the oldest ones
/// if the total count exceeds max_files.
///
/// # Arguments
///
/// * `log_dir` - Directory containing log files
/// * `prefix` - Filename prefix to match (e.g., "oxigaf" matches "oxigaf.log", "oxigaf.2026-01-01.log", etc.)
/// * `max_files` - Maximum number of log files to keep
///
/// # Errors
///
/// Returns error if:
/// - Directory cannot be read
/// - File metadata cannot be accessed
/// - Files cannot be removed
pub fn cleanup_old_logs(log_dir: &Path, prefix: &str, max_files: usize) -> Result<()> {
    if !log_dir.exists() {
        return Ok(());
    }

    let mut log_files: Vec<_> = std::fs::read_dir(log_dir)
        .with_context(|| format!("Failed to read log directory: {}", log_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with(prefix) && (n.ends_with(".log") || n.contains(".log.")))
                .unwrap_or(false)
        })
        .collect();

    // Sort by modification time (oldest first)
    log_files.sort_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()));

    // Remove oldest files if we have more than max_files
    if log_files.len() > max_files {
        let to_remove = log_files.len() - max_files;
        for entry in log_files.iter().take(to_remove) {
            let path = entry.path();
            std::fs::remove_file(&path)
                .with_context(|| format!("Failed to remove old log file: {}", path.display()))?;
            tracing::debug!(removed_log = %path.display(), "Removed old log file");
        }
        tracing::info!(
            removed_count = to_remove,
            max_files = max_files,
            "Cleaned up old log files"
        );
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert!(config.file_path.is_none());
        assert_eq!(config.max_files, 5);
        assert!(matches!(config.format, LogFormat::Json));
        assert!(matches!(config.rotation, LogRotation::Size(10485760)));
    }

    #[test]
    fn test_cleanup_old_logs_nonexistent_dir() {
        // Should not error on nonexistent directory
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        let nonexistent = tmpdir.path().join("nonexistent_subdir_12345");
        let result = cleanup_old_logs(&nonexistent, "test", 5);
        assert!(result.is_ok());
    }
}
