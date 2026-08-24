//! Log file rotation and structured logging.
//!
//! Provides structured logging to files with rotation support.
//! Supports JSON Lines format, timestamps, log levels, and automatic
//! cleanup of old log files.

use std::fs::OpenOptions;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{
    fmt, fmt::writer::BoxMakeWriter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
    Layer,
};

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
    /// Note: `tracing-appender` has no native size-based strategy, so this
    /// is implemented locally by [`SizeRotatingWriter`]: once the active
    /// file reaches the given byte count, it is renamed to `<file>.1`
    /// (shifting any existing `.1..N-1` backups up by one slot and
    /// dropping the oldest) and a fresh file is opened in its place.
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
        let file_appender = create_appender(log_path, log_config.rotation, log_config.max_files)?;

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
/// Time-based strategies (`Never`/`Hourly`/`Daily`) are handled by
/// `tracing-appender`'s [`RollingFileAppender`]; [`LogRotation::Size`] is
/// handled by the local [`SizeRotatingWriter`] since `tracing-appender` has
/// no size-based strategy of its own. Both are erased behind a single
/// [`BoxMakeWriter`] so callers don't need to care which one is active.
///
/// # Arguments
///
/// * `log_path` - Full path to the log file
/// * `rotation` - Rotation strategy to use
/// * `max_files` - For [`LogRotation::Size`], the maximum number of files
///   (active file plus numbered backups) to keep on disk.
///
/// # Errors
///
/// Returns error if the log path is invalid or the appender cannot be created
/// (e.g. the log directory is not writable).
fn create_appender(
    log_path: &Path,
    rotation: LogRotation,
    max_files: usize,
) -> Result<BoxMakeWriter> {
    let dir = log_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid log path: no parent directory"))?;

    let filename = log_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid log filename"))?;

    let build_rolling = |tracing_rotation: Rotation| -> Result<BoxMakeWriter> {
        let appender = RollingFileAppender::builder()
            .rotation(tracing_rotation)
            .filename_prefix(filename)
            .build(dir)
            .with_context(|| format!("Failed to create log appender in {}", dir.display()))?;
        Ok(BoxMakeWriter::new(appender))
    };

    match rotation {
        LogRotation::Never => build_rolling(Rotation::NEVER),
        LogRotation::Hourly => build_rolling(Rotation::HOURLY),
        LogRotation::Daily => build_rolling(Rotation::DAILY),
        LogRotation::Size(max_bytes) => {
            let writer = SizeRotatingWriter::new(dir, filename, max_bytes, max_files)
                .with_context(|| {
                    format!(
                        "Failed to create size-rotating log appender in {}",
                        dir.display()
                    )
                })?;
            Ok(BoxMakeWriter::new(Mutex::new(writer)))
        }
    }
}

// ---------------------------------------------------------------------------
// SizeRotatingWriter
// ---------------------------------------------------------------------------

/// A [`Write`](std::io::Write) implementation that rotates the target file
/// once its size reaches `max_bytes`.
///
/// `tracing-appender`'s [`RollingFileAppender`] only supports time-based
/// rotation (never/hourly/daily); this fills the size-based gap that
/// [`LogRotation::Size`] promises. On rotation the active file is renamed to
/// `<filename>.1`, any existing `.1..max_files-1` backups are shifted up by
/// one slot, and anything that falls off the end beyond `max_files` is
/// deleted. Wrapped in a [`Mutex`] (which implements `MakeWriter` for any
/// `W: Write`) so it can be shared across the tracing writer callbacks that
/// fire on every log event.
struct SizeRotatingWriter {
    dir: PathBuf,
    filename: String,
    max_bytes: u64,
    max_files: usize,
    file: std::fs::File,
    written: u64,
}

impl SizeRotatingWriter {
    /// Open (or create) `dir/filename` for size-rotating appends.
    ///
    /// `max_bytes` and `max_files` are clamped to a minimum of 1 so a
    /// misconfigured `0` cannot produce an unbounded rotate-on-every-write
    /// loop or delete the active file on every write.
    fn new(dir: &Path, filename: &str, max_bytes: u64, max_files: usize) -> io::Result<Self> {
        let dir = dir.to_path_buf();
        let filename = filename.to_string();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(&filename))?;
        let written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            dir,
            filename,
            max_bytes: max_bytes.max(1),
            max_files: max_files.max(1),
            file,
            written,
        })
    }

    fn active_path(&self) -> PathBuf {
        self.dir.join(&self.filename)
    }

    fn backup_path(&self, index: usize) -> PathBuf {
        self.dir.join(format!("{}.{index}", self.filename))
    }

    /// Roll the active file into `.1` (shifting older numbered backups up
    /// by one and dropping anything beyond `max_files`), then open a fresh
    /// active file.
    fn rotate(&mut self) -> io::Result<()> {
        self.file.flush()?;

        if self.max_files <= 1 {
            // No backups requested: just start the active file over.
            self.file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(self.active_path())?;
            self.written = 0;
            return Ok(());
        }

        // Backup slots are `.1..=(max_files - 1)`; together with the active
        // file that is `max_files` files on disk.
        let last_backup = self.max_files - 1;

        let oldest = self.backup_path(last_backup);
        if oldest.exists() {
            std::fs::remove_file(&oldest)?;
        }
        for idx in (1..last_backup).rev() {
            let from = self.backup_path(idx);
            if from.exists() {
                std::fs::rename(&from, self.backup_path(idx + 1))?;
            }
        }
        let active = self.active_path();
        if active.exists() {
            std::fs::rename(&active, self.backup_path(1))?;
        }

        self.file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&active)?;
        self.written = 0;
        Ok(())
    }
}

impl io::Write for SizeRotatingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Only rotate a non-empty file: a single write already larger than
        // `max_bytes` must not spin rotating an empty file on every call.
        if self.written > 0 && self.written + buf.len() as u64 > self.max_bytes {
            self.rotate()?;
        }
        let n = self.file.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
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
    use std::io::Write;

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

    // ------------------------------------------------------------------
    // create_appender tests
    // ------------------------------------------------------------------

    #[test]
    fn test_create_appender_errors_on_missing_dir_instead_of_panicking() {
        // Regression: the old implementation called `RollingFileAppender::new`,
        // which panics internally on failure, so a bad log directory would
        // crash the whole process instead of surfacing the documented `Err`.
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        let missing_dir = tmpdir.path().join("does_not_exist_subdir");
        let log_path = missing_dir.join("app.log");
        let result = create_appender(&log_path, LogRotation::Daily, 5);
        assert!(
            result.is_err(),
            "expected an Err when the log directory does not exist"
        );
    }

    #[test]
    fn test_create_appender_size_variant_errors_on_missing_dir() {
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        let missing_dir = tmpdir.path().join("does_not_exist_subdir");
        let log_path = missing_dir.join("app.log");
        let result = create_appender(&log_path, LogRotation::Size(1024), 5);
        assert!(
            result.is_err(),
            "expected an Err when the log directory does not exist"
        );
    }

    #[test]
    fn test_create_appender_never_variant_creates_file() {
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        let log_path = tmpdir.path().join("app.log");
        let _writer = create_appender(&log_path, LogRotation::Never, 5).expect("create appender");
        assert!(log_path.exists(), "log file should be created immediately");
    }

    #[test]
    fn test_create_appender_size_variant_creates_file() {
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        let log_path = tmpdir.path().join("app.log");
        let _writer =
            create_appender(&log_path, LogRotation::Size(4096), 5).expect("create appender");
        assert!(log_path.exists(), "log file should be created immediately");
    }

    // ------------------------------------------------------------------
    // SizeRotatingWriter tests
    // ------------------------------------------------------------------

    #[test]
    fn test_size_rotating_writer_rotates_past_max_bytes() {
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        let mut writer =
            SizeRotatingWriter::new(tmpdir.path(), "app.log", 10, 3).expect("create writer");

        // First write fills the file to exactly max_bytes; must not rotate
        // an empty file just because a single write reaches the threshold.
        writer.write_all(b"0123456789").expect("first write");
        writer.flush().expect("flush");
        assert!(tmpdir.path().join("app.log").exists());
        assert!(!tmpdir.path().join("app.log.1").exists());

        // This write pushes the active file over max_bytes, so the existing
        // 10 bytes must be rotated into `app.log.1` first.
        writer.write_all(b"more").expect("second write");
        writer.flush().expect("flush");

        let backup = tmpdir.path().join("app.log.1");
        assert!(
            backup.exists(),
            "expected a .1 backup after crossing max_bytes"
        );
        let backup_content = std::fs::read_to_string(&backup).expect("read backup");
        assert_eq!(backup_content, "0123456789");

        let active_content =
            std::fs::read_to_string(tmpdir.path().join("app.log")).expect("read active");
        assert_eq!(active_content, "more");
    }

    #[test]
    fn test_size_rotating_writer_prunes_beyond_max_files() {
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        // max_bytes=1 forces a rotation on every write after the first;
        // max_files=3 means at most 3 files (active + 2 backups) survive.
        let mut writer =
            SizeRotatingWriter::new(tmpdir.path(), "app.log", 1, 3).expect("create writer");

        for i in 0..10u8 {
            writer.write_all(&[b'a' + i]).expect("write");
        }
        writer.flush().expect("flush");

        let total_files = std::fs::read_dir(tmpdir.path()).expect("read dir").count();
        assert!(
            total_files <= 3,
            "expected at most 3 files on disk, found {total_files}"
        );
        assert!(tmpdir.path().join("app.log").exists());
    }

    #[test]
    fn test_size_rotating_writer_normalizes_zero_bounds() {
        // max_bytes=0 / max_files=0 must not panic; both are clamped to a
        // minimum of 1 internally.
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        let mut writer =
            SizeRotatingWriter::new(tmpdir.path(), "app.log", 0, 0).expect("create writer");
        writer.write_all(b"hello").expect("write");
        writer.write_all(b"world").expect("second write");
        writer.flush().expect("flush");
        assert!(tmpdir.path().join("app.log").exists());
    }

    #[test]
    fn test_size_rotating_writer_resumes_existing_file_size() {
        // Reopening a writer over a file that already has content must count
        // the existing bytes, not silently reset the size counter to zero.
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        std::fs::write(tmpdir.path().join("app.log"), b"0123456789").expect("seed file");

        let mut writer =
            SizeRotatingWriter::new(tmpdir.path(), "app.log", 10, 3).expect("create writer");
        // Any further write must immediately exceed max_bytes=10 and rotate.
        writer.write_all(b"x").expect("write");
        writer.flush().expect("flush");

        assert!(
            tmpdir.path().join("app.log.1").exists(),
            "pre-existing file content should count toward max_bytes"
        );
    }
}
