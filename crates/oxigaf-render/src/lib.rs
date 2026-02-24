//! # oxigaf-render
//!
//! Differentiable 3D Gaussian Splatting rasterizer using wgpu compute shaders.
//!
//! The rasterizer implements the full 3DGS pipeline:
//! - **Forward**: project → sort → rasterize (per-tile alpha-blending)
//! - **Backward**: reverse-order gradient computation through the rasterizer
//!
//! FLAME mesh binding allows Gaussians to be anchored to a parametric head model.
//!
//! ## Cargo Features
//!
//! This crate supports the following feature flags:
//!
//! - **`default`** = `[]`:
//!   Minimal configuration with no extra features
//!
//! - **`gpu_debug`**:
//!   Enables GPU debug mode with validation layers:
//!   - Vulkan validation layers on Linux/Windows
//!   - Metal API validation on macOS
//!   - DirectX debug layer on Windows
//!   - Enhanced error messages and warnings
//!   - **Warning**: Adds significant runtime overhead (10-100× slower)
//!
//! The `gpu_debug` feature is useful for:
//! - Debugging shader errors
//! - Validating buffer usage
//! - Catching GPU API misuse
//! - Performance profiling with detailed traces
//!
//! Example usage:
//! ```toml
//! # In Cargo.toml
//! # For production use (fast, minimal validation)
//! oxigaf-render = { version = "0.1" }
//!
//! # For development/debugging (slow, extensive validation)
//! oxigaf-render = { version = "0.1", features = ["gpu_debug"] }
//! ```

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

pub mod binding;
pub mod buffers;
pub mod config;
pub mod cpu_reference;
pub mod gaussian;
pub mod pipeline;
pub mod pool;
pub mod rasterizer;
pub mod sort;

pub use config::RasterConfig;
pub use cpu_reference::{CpuCamera, CpuRasterizer, CpuRenderOutput};
pub use pool::{BufferPool, PoolStats, PooledBuffer};
pub use rasterizer::{GaussianGradients, Rasterizer, RenderCamera, RenderOutput};

use thiserror::Error;

/// Reason for GPU device being lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceLostReason {
    /// Device was explicitly destroyed.
    Destroyed,
    /// Unknown reason.
    Unknown,
    /// Device was disconnected (e.g., GPU unplugged).
    DeviceDisconnected,
    /// Driver was updated while rendering.
    DriverUpdate,
    /// Out of GPU memory.
    OutOfMemory,
}

impl std::fmt::Display for DeviceLostReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Destroyed => write!(f, "device destroyed"),
            Self::Unknown => write!(f, "unknown reason"),
            Self::DeviceDisconnected => write!(f, "device disconnected"),
            Self::DriverUpdate => write!(f, "driver update"),
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

/// Errors that can occur during rendering operations.
#[derive(Debug, Error)]
pub enum RenderError {
    // --- GPU initialization ---
    /// General GPU initialization error.
    #[error("GPU initialization error: {0}")]
    GpuInit(String),

    /// No suitable GPU adapter found.
    #[error("No suitable GPU adapter found")]
    AdapterNotFound,

    /// Failed to create GPU device.
    #[error("Failed to create GPU device: {0}")]
    DeviceCreationFailed(String),

    /// GPU device was lost during operation.
    #[error("GPU device lost ({reason}): {message}")]
    DeviceLost {
        /// Reason for device loss.
        reason: DeviceLostReason,
        /// Additional message.
        message: String,
    },

    // --- Shader compilation ---
    /// Shader compilation failed.
    #[error("Shader compilation failed for '{shader_name}': {error}")]
    ShaderCompilation {
        /// Name of the shader that failed to compile.
        shader_name: String,
        /// Compilation error message.
        error: String,
    },

    /// Shader validation failed.
    #[error("Shader validation failed for '{shader_name}': {error}")]
    ShaderValidation {
        /// Name of the shader that failed validation.
        shader_name: String,
        /// Validation error message.
        error: String,
    },

    // --- Buffer operations ---
    /// Buffer allocation failed.
    #[error("Buffer allocation failed for '{buffer_name}' (requested {requested_size} bytes)")]
    BufferAllocation {
        /// Name of the buffer.
        buffer_name: String,
        /// Requested size in bytes.
        requested_size: u64,
    },

    /// Buffer overflow.
    #[error("Buffer overflow for '{buffer_name}' (max: {max_size}, requested: {requested})")]
    BufferOverflow {
        /// Name of the buffer.
        buffer_name: String,
        /// Maximum size in bytes.
        max_size: u64,
        /// Requested size in bytes.
        requested: u64,
    },

    /// Buffer mapping failed.
    #[error("Buffer map failed for '{buffer_name}': {error}")]
    BufferMapFailed {
        /// Name of the buffer.
        buffer_name: String,
        /// Error message.
        error: String,
    },

    // --- Rasterization ---
    /// General rasterization error.
    #[error("Rasterization error: {0}")]
    Rasterize(String),

    /// Compute dispatch limit exceeded.
    #[error("Dispatch limit exceeded for {dimension}: requested {requested}, max {max}")]
    DispatchLimitExceeded {
        /// Which dimension (x, y, or z).
        dimension: String,
        /// Requested workgroup count.
        requested: u32,
        /// Maximum allowed.
        max: u32,
    },

    /// Too many Gaussians for allocated buffers.
    #[error("Too many Gaussians: {count} exceeds maximum {max}")]
    TooManyGaussians {
        /// Actual count.
        count: u32,
        /// Maximum allowed.
        max: u32,
    },

    /// Too many tile-Gaussian pairs.
    #[error("Too many tile pairs: {count} exceeds allocated {allocated}")]
    TooManyTilePairs {
        /// Actual count.
        count: u32,
        /// Allocated capacity.
        allocated: u32,
    },

    // --- Validation ---
    /// Invalid Gaussian data.
    #[error("Invalid Gaussian at index {index}: {reason}")]
    InvalidGaussian {
        /// Index of the invalid Gaussian.
        index: usize,
        /// Reason for invalidity.
        reason: String,
    },

    /// Invalid quaternion (not normalized or zero).
    #[error("Invalid quaternion at index {index}: norm = {norm}")]
    InvalidQuaternion {
        /// Index of the Gaussian with invalid quaternion.
        index: usize,
        /// Norm of the quaternion.
        norm: f32,
    },

    /// Invalid scale values.
    #[error("Invalid scale at index {index}: values = {values:?}")]
    InvalidScale {
        /// Index of the Gaussian with invalid scale.
        index: usize,
        /// Scale values.
        values: [f32; 3],
    },

    /// Mismatched buffer sizes.
    #[error("Mismatched buffer sizes: expected {expected}, got {actual}")]
    MismatchedBufferSizes {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    // --- I/O ---
    /// Image save failed.
    #[error("Image save failed: {0}")]
    ImageSaveFailed(String),

    /// Gradient readback failed.
    #[error("Gradient readback failed: {0}")]
    GradientReadbackFailed(String),

    /// Channel receive error.
    #[error("Channel receive error: {0}")]
    ChannelRecvError(String),
}

impl From<wgpu::RequestDeviceError> for RenderError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        RenderError::DeviceCreationFailed(err.to_string())
    }
}

impl From<std::sync::mpsc::RecvError> for RenderError {
    fn from(err: std::sync::mpsc::RecvError) -> Self {
        RenderError::ChannelRecvError(err.to_string())
    }
}

impl From<image::ImageError> for RenderError {
    fn from(err: image::ImageError) -> Self {
        RenderError::ImageSaveFailed(err.to_string())
    }
}
