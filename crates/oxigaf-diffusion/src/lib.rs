//! # oxigaf-diffusion
//!
//! Multi-view diffusion model inference for GAF.
//!
//! Implements the full pipeline: CLIP image encoding → multi-view U-Net
//! denoising with camera-conditioned cross-view attention → VAE decoding.
//!
//! ## Cargo Features
//!
//! This crate supports the following feature flags:
//!
//! - **`default`** = `["accelerate", "flash_attention"]`:
//!   Default features for CPU-only inference with optimizations
//!
//! - **`accelerate`**:
//!   Uses platform-native BLAS/LAPACK for CPU tensor operations
//!   - macOS: Accelerate framework
//!   - Linux: OpenBLAS or Intel MKL
//!
//! - **`cuda`** (platform-specific):
//!   Enables NVIDIA GPU acceleration via CUDA
//!   - Requires CUDA toolkit installed
//!   - Not available on macOS
//!
//! - **`metal`** (platform-specific):
//!   Enables Apple Silicon GPU acceleration via Metal
//!   - macOS only
//!   - Optimized for M1/M2/M3 chips
//!
//! - **`flash_attention`** (enabled by default):
//!   Memory-efficient attention with O(N) complexity instead of O(N²)
//!   - Reduces memory usage by 2-4× for large images
//!   - Tiled computation for better cache locality
//!
//! - **`mixed_precision`** (planned, not yet implemented):
//!   FP16/BF16 inference for reduced memory usage
//!   - Faster on GPUs with Tensor Cores
//!   - Lower memory footprint
//!
//! Example usage:
//! ```toml
//! # In Cargo.toml
//! # For CPU-only with flash attention
//! oxigaf-diffusion = { version = "0.1", default-features = true }
//!
//! # For Apple Silicon with Metal acceleration
//! oxigaf-diffusion = { version = "0.1", features = ["metal", "flash_attention"] }
//!
//! # For NVIDIA GPU with CUDA
//! oxigaf-diffusion = { version = "0.1", features = ["cuda", "flash_attention"] }
//! ```

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

pub mod attention;
pub mod camera;
pub mod clip;
pub mod config;
#[cfg(feature = "flash_attention")]
pub mod flash_attention;
pub mod pipeline;
pub mod scheduler;
pub mod unet;
pub mod upsampler;
pub mod vae;

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during diffusion model operations.
#[derive(Debug, Error)]
pub enum DiffusionError {
    // -------------------------------------------------------------------------
    // Model Loading Errors
    // -------------------------------------------------------------------------
    /// Generic model loading error with context message.
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    /// Weight file not found at expected path.
    #[error("Weight not found: layer '{layer}', expected shape {expected_shape:?}")]
    WeightNotFound {
        layer: String,
        expected_shape: Vec<usize>,
    },

    /// Weight shape does not match expected dimensions.
    #[error("Weight shape mismatch: layer '{layer}', expected {expected:?}, got {got:?}")]
    WeightShapeMismatch {
        layer: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Safetensors file is corrupted or invalid.
    #[error("Safetensors corrupt: {path:?}, reason: {reason}")]
    SafetensorsCorrupt { path: PathBuf, reason: String },

    // -------------------------------------------------------------------------
    // Tensor Operation Errors
    // -------------------------------------------------------------------------
    /// Tensor shape mismatch during operation.
    #[error("Shape mismatch in '{op}': expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        op: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Data type mismatch between tensors.
    #[error("Dtype mismatch: expected {expected}, got {got}")]
    DtypeMismatch { expected: String, got: String },

    /// Device mismatch between tensors.
    #[error("Device mismatch: expected {expected}, got {got}")]
    DeviceMismatch { expected: String, got: String },

    // -------------------------------------------------------------------------
    // Numerical Errors
    // -------------------------------------------------------------------------
    /// NaN detected in tensor during computation.
    #[error("NaN detected in layer '{layer}' at timestep {timestep:?}")]
    NanDetected {
        layer: String,
        timestep: Option<usize>,
    },

    /// Infinity detected in tensor during computation.
    #[error("Inf detected in layer '{layer}' at timestep {timestep:?}")]
    InfDetected {
        layer: String,
        timestep: Option<usize>,
    },

    /// General numerical instability.
    #[error("Numerical instability: {context}")]
    NumericalInstability { context: String },

    // -------------------------------------------------------------------------
    // Inference Errors
    // -------------------------------------------------------------------------
    /// Generic inference error with context.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Invalid timestep value.
    #[error("Invalid timestep: {value}, max allowed: {max}")]
    InvalidTimestep { value: usize, max: usize },

    /// Invalid number of views provided.
    #[error("Invalid view count: expected {expected}, got {got}")]
    InvalidViewCount { expected: usize, got: usize },

    /// Invalid latent tensor shape.
    #[error("Invalid latent shape: expected {expected:?}, got {got:?}")]
    InvalidLatentShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Skip connection underflow during U-Net forward pass.
    #[error(
        "Skip connection underflow: expected {expected} connections, only {available} available"
    )]
    SkipConnectionUnderflow { expected: usize, available: usize },

    // -------------------------------------------------------------------------
    // Pipeline Errors
    // -------------------------------------------------------------------------
    /// Scheduler not initialized before use.
    #[error("Scheduler not initialized: call set_timesteps() first")]
    SchedulerNotInitialized,

    /// CLIP encoding failed.
    #[error("CLIP encoding failed: {0}")]
    ClipEncodingFailed(String),

    /// VAE encoding failed.
    #[error("VAE encoding failed: {0}")]
    VaeEncodeFailed(String),

    /// VAE decoding failed.
    #[error("VAE decoding failed: {0}")]
    VaeDecodeFailed(String),

    /// U-Net forward pass failed.
    #[error("U-Net forward failed at timestep {timestep}: {reason}")]
    UnetForwardFailed { timestep: usize, reason: String },

    // -------------------------------------------------------------------------
    // I/O Errors
    // -------------------------------------------------------------------------
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Image processing error.
    #[error("Image processing error: {0}")]
    ImageProcessingError(String),

    // -------------------------------------------------------------------------
    // Candle Backend Errors
    // -------------------------------------------------------------------------
    /// Error from candle tensor operations.
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Result type for diffusion operations.
pub type DiffusionResult<T> = std::result::Result<T, DiffusionError>;

// Re-exports
pub use clip::ClipImageEncoder;
pub use config::DiffusionConfig;
pub use pipeline::{MultiViewDiffusionPipeline, MultiViewOutput};
pub use scheduler::{DdimScheduler, PredictionType};
pub use unet::MultiViewUNet;
pub use upsampler::{LatentUpsampler, UpsamplerMode};
pub use vae::Vae;

// Flash attention exports (only when feature is enabled)
#[cfg(feature = "flash_attention")]
pub use flash_attention::{
    flash_attention, flash_attention_with_config, FlashAttention, FlashAttentionConfig,
};
