//! Error type for HDR tone mapping operations.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;

/// Errors produced by HDR tone mapping operations.
#[derive(Debug, Error)]
pub enum ToneMappingError {
    /// Image slice has incorrect length.
    #[error("Invalid image: {0}")]
    InvalidImage(String),
    /// A configuration parameter is out of valid range.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    /// The image slice is empty (no pixels to process).
    #[error("Empty image")]
    EmptyImage,
    /// Buffer size does not match width×height×channels.
    #[error("Image size mismatch: buffer {got} != {width}x{height}x{channels}")]
    SizeMismatch {
        /// Actual buffer length.
        got: usize,
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
        /// Number of channels.
        channels: usize,
    },
    /// A named parameter is invalid.
    #[error("Invalid parameter {name}: {reason}")]
    InvalidParam {
        /// Parameter name.
        name: String,
        /// Reason the value is invalid.
        reason: String,
    },
}
