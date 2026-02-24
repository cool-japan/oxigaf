//! Error types for the FLAME module.

/// Errors that can occur when working with the FLAME model.
#[derive(Debug, thiserror::Error)]
pub enum FlameError {
    /// I/O error reading model files.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error loading `.npy` array data.
    #[error("Failed to load array '{name}': {source}")]
    NpyLoad {
        name: String,
        source: ndarray_npy::ReadNpyError,
    },

    /// Error loading `.npz` archive data.
    #[cfg(feature = "npz")]
    #[error("Failed to load NPZ array '{name}': {source}")]
    NpzLoad {
        name: String,
        source: ndarray_npy::ReadNpzError,
    },

    /// Array shape does not match expected dimensions.
    #[error("Shape mismatch for '{name}': expected {expected}, got {got}")]
    ShapeMismatch {
        name: String,
        expected: String,
        got: String,
    },

    /// Model directory does not exist or is missing required files.
    #[error("Model directory error: {0}")]
    ModelDir(String),

    /// Invalid parameter dimensions.
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    /// Export failed (OBJ/PLY/glTF).
    #[error("Export failed: {format} - {reason}")]
    Export {
        /// Export format (e.g., "OBJ", "PLY", "glTF").
        format: String,
        /// Detailed reason for failure.
        reason: String,
    },

    /// Animation sequence error.
    #[error("Animation error at frame {frame}: {reason}")]
    Animation {
        /// Frame index where error occurred.
        frame: usize,
        /// Detailed reason for failure.
        reason: String,
    },

    /// Landmark detection or projection error.
    #[error("Landmark error for index {index}: {reason}")]
    Landmark {
        /// Landmark index that caused the error.
        index: usize,
        /// Detailed reason for failure.
        reason: String,
    },

    /// Expression transfer failure.
    #[error("Expression transfer failed: {reason}")]
    Transfer {
        /// Detailed reason for failure.
        reason: String,
    },

    /// Numerical computation error (e.g., matrix inversion failure).
    #[error("Numerical error: {reason}")]
    Numerical {
        /// Detailed reason for failure.
        reason: String,
    },

    /// Index out of bounds error.
    #[error("Index out of bounds: {context} - index {index} >= {len}")]
    IndexOutOfBounds {
        /// Context where the error occurred.
        context: String,
        /// The invalid index.
        index: usize,
        /// The maximum valid length.
        len: usize,
    },

    /// Error loading safetensors file.
    #[error("Failed to load safetensors from '{path}': {message}")]
    SafeTensorsLoad {
        /// Path to the safetensors file.
        path: std::path::PathBuf,
        /// Error details.
        message: String,
    },

    /// Error saving safetensors file.
    #[error("Failed to save safetensors to '{path}': {message}")]
    SafeTensorsSave {
        /// Path to the safetensors file.
        path: std::path::PathBuf,
        /// Error details.
        message: String,
    },

    /// Missing tensor in safetensors file.
    #[error("Missing tensor '{name}' in safetensors file: {message}")]
    SafeTensorsMissing {
        /// Name of the missing tensor.
        name: String,
        /// Error details.
        message: String,
    },

    /// Invalid dtype in safetensors tensor.
    #[error("Invalid dtype for tensor '{name}': expected {expected}, got {got}")]
    SafeTensorsInvalidDtype {
        /// Name of the tensor.
        name: String,
        /// Expected dtype.
        expected: String,
        /// Actual dtype.
        got: String,
    },

    /// I/O error with path context.
    #[error("I/O error for '{path}': {source}")]
    IoError {
        /// The I/O error source.
        source: std::io::Error,
        /// Path where error occurred.
        path: std::path::PathBuf,
    },
}

impl FlameError {
    /// Create an export error with format and reason.
    #[must_use]
    pub fn export(format: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Export {
            format: format.into(),
            reason: reason.into(),
        }
    }

    /// Create an animation error with frame and reason.
    #[must_use]
    pub fn animation(frame: usize, reason: impl Into<String>) -> Self {
        Self::Animation {
            frame,
            reason: reason.into(),
        }
    }

    /// Create a landmark error with index and reason.
    #[must_use]
    pub fn landmark(index: usize, reason: impl Into<String>) -> Self {
        Self::Landmark {
            index,
            reason: reason.into(),
        }
    }

    /// Create a transfer error with reason.
    #[must_use]
    pub fn transfer(reason: impl Into<String>) -> Self {
        Self::Transfer {
            reason: reason.into(),
        }
    }

    /// Create a numerical error with reason.
    #[must_use]
    pub fn numerical(reason: impl Into<String>) -> Self {
        Self::Numerical {
            reason: reason.into(),
        }
    }

    /// Create an index out of bounds error.
    #[must_use]
    pub fn index_out_of_bounds(context: impl Into<String>, index: usize, len: usize) -> Self {
        Self::IndexOutOfBounds {
            context: context.into(),
            index,
            len,
        }
    }
}
