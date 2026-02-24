//! Error types for oxigaf-bridge
//!
//! This module defines comprehensive error types for weight conversion operations.

use thiserror::Error;

/// Main error type for oxigaf-bridge operations
#[derive(Debug, Error)]
pub enum BridgeError {
    /// Error during weight conversion
    #[error("Weight conversion error: {0}")]
    WeightConversion(String),

    /// Error during layer name mapping
    #[error("Layer mapping error: {0}")]
    LayerMapping(String),

    /// Error during precision conversion
    #[error("Precision conversion error: {0}")]
    PrecisionConversion(String),

    /// Error from safetensors operations
    #[error("Safetensors error: {0}")]
    SafeTensors(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serde JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Safetensors library error
    #[error("Safetensors library error: {0}")]
    SafeTensorsLib(#[from] safetensors::SafeTensorError),

    /// Invalid tensor shape
    #[error("Invalid tensor shape: expected {expected:?}, got {actual:?}")]
    InvalidShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Unsupported dtype
    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    /// Missing tensor in file
    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// Conversion error
    #[error("Conversion error: {0}")]
    Conversion(String),
}

/// Specialized result type for bridge operations
pub type Result<T> = std::result::Result<T, BridgeError>;
