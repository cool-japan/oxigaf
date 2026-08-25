//! CLI error handling with user-friendly messages and exit codes.
//!
//! This module provides:
//! - [`CliError`] enum with descriptive variants for all CLI failure modes
//! - Standardized exit codes for scripting and automation
//! - User-friendly error formatting with actionable suggestions

use std::path::PathBuf;

use thiserror::Error;

// ---------------------------------------------------------------------------
// Exit Codes
// ---------------------------------------------------------------------------

/// Exit code for successful execution.
pub const EXIT_SUCCESS: i32 = 0;

/// Exit code for general errors (catch-all).
pub const EXIT_GENERAL_ERROR: i32 = 1;

/// Exit code for configuration errors (invalid TOML, validation failure).
pub const EXIT_CONFIG_ERROR: i32 = 2;

/// Exit code for I/O errors (file not found, permission denied).
pub const EXIT_IO_ERROR: i32 = 3;

/// Exit code for GPU-related errors (no adapter, device creation failed).
pub const EXIT_GPU_ERROR: i32 = 4;

/// Exit code for asset/model download failures.
pub const EXIT_ASSET_ERROR: i32 = 5;

/// Exit code for training errors (NaN loss, memory exhaustion).
pub const EXIT_TRAINING_ERROR: i32 = 6;

/// Exit code for export errors (format not supported, write failed).
pub const EXIT_EXPORT_ERROR: i32 = 7;

/// Exit code when process is interrupted (SIGINT / Ctrl+C).
///
/// Matches the shell convention of `128 + SIGINT`; `main.rs`'s signal
/// handler exits with it after restoring the terminal.
pub const EXIT_INTERRUPTED: i32 = 130;

// ---------------------------------------------------------------------------
// CliError Enum
// ---------------------------------------------------------------------------

/// Comprehensive CLI error type with user-friendly messages.
#[derive(Debug, Error)]
pub enum CliError {
    /// Configuration file not found.
    #[error("Configuration file not found: {path}")]
    ConfigNotFound {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Configuration file parsing failed.
    #[error("Failed to parse configuration file: {path}")]
    ConfigParseError {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    /// Configuration validation failed.
    #[error("Configuration validation failed: {reason}")]
    ConfigValidationError { reason: String },

    /// FLAME model directory not found or invalid.
    #[error("FLAME model not found or invalid: {path}")]
    FlameModelInvalid { path: PathBuf, reason: String },

    /// GPU adapter not available.
    #[error("GPU not available: {backend}")]
    GpuNotAvailable {
        backend: String,
        fallback: Option<String>,
    },

    /// GPU device creation failed.
    #[error("GPU device creation failed")]
    GpuDeviceError {
        #[source]
        source: anyhow::Error,
    },

    /// Asset download failed.
    #[error("Failed to download asset: {name}")]
    AssetDownloadFailed {
        name: String,
        url: String,
        cause: String,
    },

    /// Export format not supported.
    #[error("Unsupported export format: {format}")]
    ExportFormatUnsupported {
        format: String,
        supported: Vec<String>,
    },

    /// Checkpoint file corrupted or incompatible.
    #[error("Checkpoint corrupted or incompatible: {path}")]
    CheckpointCorrupted {
        path: PathBuf,
        expected_version: String,
    },

    /// Insufficient GPU memory.
    #[error("Insufficient GPU memory: requires {required_mb}MB, available {available_mb}MB")]
    InsufficientVram { required_mb: u64, available_mb: u64 },

    /// Input path (video/image file, or a frame directory) is invalid.
    // "input" rather than "input file": the same variant reports directory
    // problems (e.g. image_io's "not an existing directory"), so the noun
    // must not contradict the reason.
    #[error("Invalid input: {path} — {reason}")]
    InputInvalid { path: PathBuf, reason: String },

    /// Model file not found or invalid format.
    #[error("Model file not found or invalid: {path}")]
    ModelLoadError {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },

    /// Training failed.
    #[error("Training failed at iteration {iteration}")]
    TrainingFailed {
        iteration: u32,
        #[source]
        source: anyhow::Error,
    },

    /// Generic I/O error.
    #[error("I/O error: {context}")]
    IoError {
        context: String,
        #[source]
        source: std::io::Error,
    },

    /// glTF export failed.
    #[error("glTF export failed: {0}")]
    GltfExport(String),

    /// Point cloud export failed.
    #[error("Point cloud export failed: {0}")]
    PointCloudExport(String),

    /// Video export failed.
    #[error("Video export failed: {0}")]
    VideoExport(String),

    /// Mesh export failed.
    #[error("Mesh export failed: {0}")]
    MeshExport(String),

    /// Any other error wrapped via anyhow.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl CliError {
    /// Get the appropriate exit code for this error.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::ConfigNotFound { .. }
            | Self::ConfigParseError { .. }
            | Self::ConfigValidationError { .. } => EXIT_CONFIG_ERROR,

            Self::IoError { .. } | Self::InputInvalid { .. } => EXIT_IO_ERROR,

            Self::GpuNotAvailable { .. }
            | Self::GpuDeviceError { .. }
            | Self::InsufficientVram { .. } => EXIT_GPU_ERROR,

            Self::AssetDownloadFailed { .. } => EXIT_ASSET_ERROR,

            Self::TrainingFailed { .. } | Self::CheckpointCorrupted { .. } => EXIT_TRAINING_ERROR,

            Self::ExportFormatUnsupported { .. }
            | Self::FlameModelInvalid { .. }
            | Self::ModelLoadError { .. }
            | Self::GltfExport(_)
            | Self::PointCloudExport(_)
            | Self::VideoExport(_)
            | Self::MeshExport(_) => EXIT_EXPORT_ERROR,

            Self::Other(_) => EXIT_GENERAL_ERROR,
        }
    }

    /// Get a user-friendly suggestion for how to resolve this error.
    #[must_use]
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            Self::ConfigNotFound { .. } => Some(
                "Create a configuration file with `oxigaf config init`, \
                 or specify a path with `--config <path>`",
            ),
            Self::ConfigParseError { .. } => Some(
                "Check the TOML syntax. Common issues: missing quotes around strings, \
                 unclosed brackets, or invalid escape sequences.",
            ),
            Self::ConfigValidationError { .. } => Some(
                "Review the configuration values. Run `oxigaf config validate <path>` \
                 for detailed field-by-field validation.",
            ),
            Self::FlameModelInvalid { .. } => Some(
                "Run `oxigaf setup` to download the required FLAME model files, \
                 or verify the path points to a directory with .npy files.",
            ),
            Self::GpuNotAvailable { .. } => Some(
                "Ensure GPU drivers are installed. Run `oxigaf doctor` \
                 to check wgpu backend availability.",
            ),
            Self::GpuDeviceError { .. } => Some(
                "Try a different GPU index with `--device <index>`, \
                 or check if another process is using the GPU.",
            ),
            Self::AssetDownloadFailed { .. } => Some(
                "Check your internet connection. You can also download \
                 the asset manually and place it in ~/.cache/oxigaf/",
            ),
            Self::ExportFormatUnsupported { .. } => {
                Some("Use one of the supported formats: ply, safetensors, gltf, json, pointcloud.")
            }
            Self::CheckpointCorrupted { .. } => Some(
                "The checkpoint may be from an incompatible version. \
                 Try starting fresh or use an older checkpoint.",
            ),
            Self::InsufficientVram { .. } => Some(
                "Reduce `views_per_step` or `image_size` in the config, \
                 or use a GPU with more memory.",
            ),
            Self::InputInvalid { .. } => Some(
                "Ensure the input is a valid video file or directory of frames. \
                 Supported formats: mp4, avi, mov, jpg, png.",
            ),
            Self::ModelLoadError { .. } => Some(
                "Ensure the file is a valid .ply or .json checkpoint. \
                 Check file permissions and integrity.",
            ),
            Self::GltfExport(_)
            | Self::PointCloudExport(_)
            | Self::VideoExport(_)
            | Self::MeshExport(_) => Some(
                "Check output path permissions and available disk space. \
                 Ensure the output directory exists.",
            ),
            Self::TrainingFailed { .. } => Some(
                "Check for NaN losses or memory issues. Try reducing \
                 learning rates or batch size.",
            ),
            Self::IoError { .. } => Some("Check file permissions and available disk space."),
            Self::Other(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type alias
// ---------------------------------------------------------------------------

/// Convenience type alias for CLI operations.
pub type CliResult<T> = Result<T, CliError>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every suggestion that names a subcommand must name one the parser
    /// accepts. `cli::tests` asserts the same property from the parser's
    /// side; this one guards the strings themselves against re-acquiring the
    /// `config-cmd` spelling that never matched what users could type.
    #[test]
    fn suggestions_never_name_the_retired_config_cmd_spelling() {
        let errors = [
            CliError::ConfigNotFound {
                path: PathBuf::from("oxigaf.toml"),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
            },
            CliError::ConfigValidationError {
                reason: "total_iterations must be > 0".to_string(),
            },
        ];
        for error in errors {
            let text = error.suggestion().unwrap_or_default();
            assert!(
                text.contains("oxigaf config "),
                "suggestion no longer points at `oxigaf config`: {text}"
            );
            assert!(
                !text.contains("config-cmd"),
                "suggestion still names the retired `config-cmd`: {text}"
            );
        }
    }

    /// The taxonomy exists so a scripted caller can branch on *why* a run
    /// failed; collapsing distinct classes onto the catch-all would silently
    /// undo that.
    #[test]
    fn exit_codes_separate_the_failure_classes() {
        assert_eq!(
            CliError::ConfigValidationError {
                reason: String::new()
            }
            .exit_code(),
            EXIT_CONFIG_ERROR
        );
        assert_eq!(
            CliError::GpuNotAvailable {
                backend: "any".to_string(),
                fallback: None,
            }
            .exit_code(),
            EXIT_GPU_ERROR
        );
        assert_eq!(
            CliError::MeshExport("no isosurface".to_string()).exit_code(),
            EXIT_EXPORT_ERROR
        );
        assert_eq!(
            CliError::Other(anyhow::anyhow!("boom")).exit_code(),
            EXIT_GENERAL_ERROR
        );
    }

    /// Regression: `InputInvalid`'s `#[error(...)]` format string used to
    /// name only `{path}`, so every caller's `reason` — the one field that
    /// actually explains what is wrong with the input — was built and
    /// carried around but never reached the user. `commands::quality` and
    /// `commands::video` used to work around this by attaching the same
    /// text again as `anyhow` context; now that `Display` renders `reason`
    /// itself, the message must contain it exactly once.
    #[test]
    fn input_invalid_display_renders_the_reason_exactly_once() {
        let reason = "not an existing file";
        let err = CliError::InputInvalid {
            path: PathBuf::from("frames/000.png"),
            reason: reason.to_string(),
        };
        let rendered = err.to_string();
        assert!(
            rendered.contains("frames/000.png"),
            "message dropped the path: {rendered}"
        );
        assert!(
            rendered.contains(reason),
            "message dropped the reason: {rendered}"
        );
        assert_eq!(
            rendered.matches(reason).count(),
            1,
            "reason must appear exactly once, not duplicated by a leftover \
             context layer: {rendered}"
        );
    }
}
