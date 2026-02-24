//! # oxigaf-trainer
//!
//! GAF optimization pipeline — iterative denoising distillation.
//!
//! Provides:
//! - Gaussian initialization on FLAME mesh surfaces
//! - Per-parameter Adam optimizer with group-wise learning rates
//! - Photometric + structural loss computation (L1, SSIM)
//! - Adaptive density control (split / clone / prune)
//! - Checkpoint save / load (JSON + flat f32 arrays)
//! - Metric tracking (PSNR, SSIM history)
//! - Main training loop orchestrating render ↔ diffusion distillation

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

pub mod checkpoint;
pub mod config;
pub mod density;
pub mod diffusion_target;
pub mod init;
pub mod loss;
pub mod lpips;
pub mod metrics;
pub mod optimizer;
pub mod tensorboard;
pub mod trainer;

use thiserror::Error;

/// Current checkpoint format version.
pub const CHECKPOINT_VERSION: u32 = 1;

/// Errors produced by the trainer subsystem.
#[derive(Debug, Error)]
pub enum TrainerError {
    // ---- Initialization ----
    #[error("Initialization error: {0}")]
    Init(String),

    #[error("Empty model: no Gaussians to optimize")]
    EmptyModel,

    #[error("Mesh has no faces for Gaussian initialization")]
    EmptyMesh,

    // ---- Training ----
    #[error("Training error: {0}")]
    Training(String),

    // ---- Numerical Issues ----
    #[error("NaN detected in {parameter}: index {index}")]
    NanDetected { parameter: String, index: usize },

    #[error("Infinity detected in {parameter}: index {index}")]
    InfDetected { parameter: String, index: usize },

    #[error("Gradient explosion: norm {norm:.2e} exceeds threshold {threshold:.2e}")]
    GradientExplosion { norm: f32, threshold: f32 },

    // ---- Configuration ----
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Parameter out of range: {param} = {value}, expected {expected}")]
    ParameterOutOfRange {
        param: String,
        value: String,
        expected: String,
    },

    // ---- Checkpoint ----
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    #[error("Checkpoint corrupted: {0}")]
    CheckpointCorrupted(String),

    #[error("Checkpoint version mismatch: found {found}, expected {expected}")]
    CheckpointVersionMismatch { found: u32, expected: u32 },

    #[error("Checkpoint data mismatch: {field} has length {actual}, expected {expected}")]
    CheckpointDataMismatch {
        field: String,
        actual: usize,
        expected: usize,
    },

    // ---- Optimizer ----
    #[error("Optimizer error: {0}")]
    Optimizer(String),

    #[error("Gradient buffer size mismatch: expected {expected}, got {actual}")]
    GradientSizeMismatch { expected: usize, actual: usize },

    // ---- Loss ----
    #[error("Loss computation error: {0}")]
    Loss(String),

    #[error("Image dimension mismatch: expected {expected}, got {actual}")]
    ImageDimensionMismatch { expected: usize, actual: usize },

    // ---- Density Control ----
    #[error("Density control error: {0}")]
    DensityControl(String),

    #[error("Model size mismatch: expected {expected}, got {actual}")]
    ModelSizeMismatch { expected: usize, actual: usize },

    // ---- GPU/Memory ----
    #[error("GPU out of memory: requested {requested} bytes, available {available}")]
    GpuOom { requested: usize, available: usize },

    #[error("GPU buffer overflow: {0}")]
    GpuBufferOverflow(String),

    // ---- Diffusion ----
    #[error("Diffusion pipeline not loaded")]
    DiffusionNotLoaded,

    #[error("View generation failed for camera {camera_idx}: {reason}")]
    ViewGenerationFailed { camera_idx: usize, reason: String },

    // ---- External Errors ----
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Render error: {0}")]
    Render(#[from] oxigaf_render::RenderError),

    #[error("Diffusion error: {0}")]
    Diffusion(#[from] oxigaf_diffusion::DiffusionError),

    #[error("FLAME error: {0}")]
    Flame(#[from] oxigaf_flame::FlameError),
}

// ---- Re-exports ----

pub use config::{DensityConfig, InitConfig, LossConfig, OptimizerConfig, TrainingConfig};
pub use diffusion_target::{
    DiffusionTargetConfig, DiffusionTargetGenerator, SdsLoss, SdsWeighting, TemporalConsistency,
    ViewConsistencyLoss,
};
pub use loss::LpipsLossComputer;
pub use lpips::{lpips_loss, LpipsDistance, LpipsWeights, VggFeatureExtractor};
pub use tensorboard::{LearningRates, TensorBoardConfig, TensorBoardWriter, TrainingMetricsLogger};
pub use trainer::{StepOutput, Trainer};
