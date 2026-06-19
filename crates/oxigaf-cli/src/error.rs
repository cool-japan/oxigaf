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
#[allow(dead_code)]
pub const EXIT_INTERRUPTED: i32 = 130;

// ---------------------------------------------------------------------------
// CliError Enum
// ---------------------------------------------------------------------------

/// Comprehensive CLI error type with user-friendly messages.
#[derive(Debug, Error)]
#[allow(dead_code)]
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

    /// Input file (video/image) is invalid.
    #[error("Invalid input file: {path}")]
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
                "Review the configuration values. Run `oxigaf config validate` \
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
#[allow(dead_code)]
pub type CliResult<T> = Result<T, CliError>;
