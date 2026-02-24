//! # oxigaf
//!
//! **Pure Rust Gaussian Avatar Framework** — unified API for the OxiGAF ecosystem.
//!
//! OxiGAF implements [GAF (Gaussian Avatars Reconstructed from Multi-view Images using Feed-forward)](https://www.microsoft.com/en-us/research/project/gaf/)
//! in pure Rust with no Python or C/C++ dependencies in the default build.
//!
//! ## Key Features
//!
//! - **FLAME parametric head model** — 5023 vertices, Linear Blend Skinning (LBS)
//! - **Multi-view diffusion** — novel view synthesis via CLIP + U-Net + VAE
//! - **Differentiable 3DGS rasterizer** — wgpu compute shaders with FLAME binding
//! - **Full training pipeline** — Adam optimizer, density control, checkpointing
//!
//! ## Project Structure
//!
//! ```text
//! oxigaf (meta-crate)
//! ├── oxigaf-flame     — FLAME model + mesh utilities
//! ├── oxigaf-diffusion — Multi-view diffusion inference
//! ├── oxigaf-render    — wgpu 3DGS rasterizer
//! ├── oxigaf-trainer   — Training orchestration
//! └── oxigaf-cli       — Command-line interface
//! ```
//!
//! ## Quick Start
//!
//! Add the dependency to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! oxigaf = "0.1"
//!
//! # For GPU acceleration:
//! # oxigaf = { version = "0.1", features = ["cuda"] }  # NVIDIA
//! # oxigaf = { version = "0.1", features = ["metal"] } # Apple Silicon
//! ```
//!
//! ### Minimal Example: Load FLAME and Generate a Mesh
//!
//! ```rust,ignore
//! use oxigaf::prelude::*;
//!
//! fn main() -> oxigaf::Result<()> {
//!     // Load FLAME model from directory containing .npy files
//!     let model = FlameModel::load("path/to/flame/model")?;
//!
//!     // Create neutral parameters (zero shape, expression, pose)
//!     let params = FlameParams::neutral();
//!
//!     // Run forward pass to get posed mesh
//!     let mesh = model.forward(&params);
//!
//!     println!("Generated mesh with {} vertices", mesh.vertices.len());
//!     Ok(())
//! }
//! ```
//!
//! ## Data Flow
//!
//! ```text
//! Input Image
//!      │
//!      ▼
//! ┌─────────────┐     ┌───────────────────┐
//! │ CLIP Encoder│ ──▶ │ Multi-View U-Net  │
//! └─────────────┘     └───────────────────┘
//!                             │
//!                             ▼
//!                     ┌───────────────┐
//!                     │  VAE Decoder  │
//!                     └───────────────┘
//!                             │
//!      ┌──────────────────────┼──────────────────────┐
//!      │                      │                      │
//!      ▼                      ▼                      ▼
//! View 0 (RGB)          View 1 (RGB)          View N (RGB)
//!      │                      │                      │
//!      └──────────────────────┼──────────────────────┘
//!                             │
//!                             ▼
//!                     ┌───────────────┐
//!                     │ FLAME Model   │ ◀── Pose Params
//!                     └───────────────┘
//!                             │
//!                             ▼
//!                     ┌───────────────┐
//!                     │ Gaussian Init │ (sample mesh surface)
//!                     └───────────────┘
//!                             │
//!                             ▼
//!                     ┌───────────────┐
//!                     │3DGS Rasterizer│ ──▶ Rendered Views
//!                     └───────────────┘
//!                             │
//!                             ▼
//!                     ┌───────────────┐
//!                     │ Loss (L1+SSIM)│
//!                     └───────────────┘
//!                             │
//!                             ▼
//!                     ┌───────────────┐
//!                     │Adam Optimizer │ ──▶ Updated Gaussians
//!                     └───────────────┘
//! ```
//!
//! ## Feature Flags
//!
//! ### GPU Backend Features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `default` | Pure CPU inference (no CUDA/Metal dependencies) |
//! | `cuda` | NVIDIA GPU acceleration via candle CUDA backend (requires CUDA toolkit) |
//! | `metal` | Apple Silicon acceleration via Metal (automatic on macOS with M1/M2/M3) |
//!
//! ### Performance Optimization Features
//!
//! | Feature | Description | Speedup |
//! |---------|-------------|---------|
//! | `simd` | SIMD-accelerated FLAME operations (requires nightly Rust) | 3-4× faster |
//! | `parallel` | Parallel batch processing with rayon | Near-linear with cores |
//! | `flash_attention` | Memory-efficient O(N) attention (enabled by default) | 2-4× less memory |
//! | `mixed_precision` | FP16/BF16 inference (planned, not yet implemented) | 2× faster on GPUs |
//!
//! ### Debug Features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `gpu_debug` | Enable GPU validation layers (adds 10-100× overhead) |
//!
//! ### Convenience Feature Bundles
//!
//! | Feature | Enables |
//! |---------|---------|
//! | `full_performance` | `simd`, `parallel`, `flash_attention` |
//! | `all_features` | All available features (except GPU backends) |
//!
//! ### Examples
//!
//! ```toml
//! # CPU-only with all optimizations (requires nightly for SIMD)
//! oxigaf = { version = "0.1", features = ["full_performance"] }
//!
//! # Apple Silicon with Metal acceleration
//! oxigaf = { version = "0.1", features = ["metal", "parallel", "flash_attention"] }
//!
//! # Development with GPU debugging enabled
//! oxigaf = { version = "0.1", features = ["gpu_debug"] }
//!
//! # NVIDIA GPU with all optimizations
//! oxigaf = { version = "0.1", features = ["cuda", "full_performance"] }
//! ```
//!
//! The wgpu rasterizer automatically selects the best available backend:
//! - **Vulkan** — Linux/Windows default
//! - **Metal** — macOS default
//! - **DX12** — Windows alternative
//!
//! ## Version Compatibility
//!
//! | Component | Minimum Version | Tested Version |
//! |-----------|-----------------|----------------|
//! | Rust      | 1.75+           | 1.85.0         |
//! | wgpu      | 27.x            | 27.x           |
//! | candle    | 0.9.x           | 0.9.x          |
//! | nalgebra  | 0.34.x          | 0.34.x         |
//! | glam      | 0.31.x          | 0.31.x         |
//!
//! ### GPU Requirements
//!
//! - **wgpu**: Any Vulkan 1.1+ / Metal 2.0+ / DX12 GPU
//! - **candle CUDA**: NVIDIA compute capability 7.0+ (Volta and newer)
//! - **candle Metal**: Apple M1+ chips
//!
//! ## Module Responsibilities
//!
//! - [`flame`] — FLAME model I/O, LBS forward pass, normal map rendering
//! - [`diffusion`] — CLIP + U-Net + VAE, DDIM scheduling
//! - [`render`] — GPU projection, sorting, alpha-blending
//! - [`trainer`] — Loss computation, optimizer, density control, checkpoints
//!
//! ## Migration from Python GAF
//!
//! | Python (GAF/PyTorch) | Rust (oxigaf) |
//! |----------------------|---------------|
//! | `torch.Tensor` | `candle_core::Tensor` |
//! | `FLAME(...)` | `FlameModel::load(...)` |
//! | `model(params)` | `model.forward(&params)` |
//! | `trimesh.Trimesh` | `Mesh` |
//! | `PIL.Image` | `image::DynamicImage` |
//!
//! Key API differences:
//! - No automatic differentiation by default (explicit gradients)
//! - Immutable-by-default parameters
//! - Error handling via `Result<T, E>` instead of exceptions
//! - Builder patterns for complex configurations (e.g., `FlameParamsBuilder`)

#![deny(clippy::unwrap_used)]

use thiserror::Error;

/// FLAME parametric head model.
///
/// Provides FLAME model loading, forward pass (LBS), mesh utilities,
/// normal map rendering, and surface point sampling.
pub use oxigaf_flame as flame;

/// Multi-view diffusion model inference.
///
/// Implements the full diffusion pipeline: CLIP image encoding,
/// multi-view U-Net denoising, and VAE decoding.
pub use oxigaf_diffusion as diffusion;

/// Differentiable 3D Gaussian Splatting rasterizer.
///
/// GPU-accelerated rasterizer using wgpu compute shaders with
/// forward and backward passes for gradient computation.
pub use oxigaf_render as render;

/// Optimization / training pipeline.
///
/// Provides Gaussian initialization, Adam optimizer, loss computation,
/// adaptive density control, and checkpoint management.
pub use oxigaf_trainer as trainer;

// ============================================================================
// Unified Error Type
// ============================================================================

/// Unified error type wrapping all sub-crate errors.
///
/// This enum provides a single error type for the entire OxiGAF ecosystem,
/// allowing seamless error propagation across crate boundaries.
///
/// # Example
///
/// ```rust,ignore
/// use oxigaf::prelude::*;
///
/// fn process() -> oxigaf::Result<()> {
///     let model = FlameModel::load("path/to/model")?;  // FlameError → OxigafError
///     let rasterizer = Rasterizer::new(&config)?;      // RenderError → OxigafError
///     Ok(())
/// }
/// ```
#[derive(Debug, Error)]
pub enum OxigafError {
    /// Error from the FLAME parametric model.
    #[error("FLAME error: {0}")]
    Flame(#[from] flame::FlameError),

    /// Error from the diffusion pipeline.
    #[error("Diffusion error: {0}")]
    Diffusion(#[from] diffusion::DiffusionError),

    /// Error from the GPU rasterizer.
    #[error("Render error: {0}")]
    Render(#[from] render::RenderError),

    /// Error from the training pipeline.
    #[error("Trainer error: {0}")]
    Trainer(#[from] trainer::TrainerError),
}

/// A specialized `Result` type for OxiGAF operations.
///
/// This type alias provides a convenient way to handle errors across
/// the OxiGAF ecosystem without explicitly specifying the error type.
///
/// # Example
///
/// ```rust,ignore
/// use oxigaf::prelude::*;
///
/// fn load_and_render() -> oxigaf::Result<()> {
///     let model = FlameModel::load("path/to/flame")?;
///     // ... processing ...
///     Ok(())
/// }
/// ```
pub type Result<T> = std::result::Result<T, OxigafError>;

/// Returns the crate version string.
///
/// # Example
///
/// ```rust
/// let version = oxigaf::version();
/// assert!(!version.is_empty());
/// ```
#[inline]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ============================================================================
// Prelude
// ============================================================================

/// Convenience re-exports of the most commonly used types.
///
/// Import everything from the prelude for quick access to the full API:
///
/// ```rust
/// use oxigaf::prelude::*;
/// ```
///
/// This module re-exports:
///
/// ## FLAME Model Types
/// - [`FlameModel`](crate::flame::FlameModel) — FLAME parametric head model
/// - [`FlameParams`](crate::flame::FlameParams) — Parameters (shape, expression, pose)
/// - [`FlameParamsBuilder`](crate::flame::FlameParamsBuilder) — Builder for constructing parameters
/// - [`FlameError`](crate::flame::FlameError) — FLAME-specific errors
/// - [`Mesh`](crate::flame::Mesh) — Triangle mesh representation
/// - [`Camera`](crate::flame::Camera) — Camera parameters for rendering
/// - [`NormalMapRenderer`](crate::flame::NormalMapRenderer) — CPU normal map renderer
/// - [`SurfacePoint`](crate::flame::SurfacePoint) — Sampled point on mesh surface
/// - [`sample_mesh_surface`](crate::flame::sample_mesh_surface) — Sample points from mesh
///
/// ## Diffusion Types
/// - [`MultiViewDiffusionPipeline`](crate::diffusion::MultiViewDiffusionPipeline) — Full diffusion pipeline
/// - [`MultiViewOutput`](crate::diffusion::MultiViewOutput) — Output from diffusion inference
/// - [`DiffusionConfig`](crate::diffusion::DiffusionConfig) — Pipeline configuration
/// - [`DdimScheduler`](crate::diffusion::DdimScheduler) — DDIM noise scheduler
/// - [`PredictionType`](crate::diffusion::PredictionType) — Noise prediction type
/// - [`ClipImageEncoder`](crate::diffusion::ClipImageEncoder) — CLIP image encoder
/// - [`Vae`](crate::diffusion::Vae) — Variational autoencoder
/// - [`MultiViewUNet`](crate::diffusion::MultiViewUNet) — Multi-view U-Net
/// - [`DiffusionError`](crate::diffusion::DiffusionError) — Diffusion-specific errors
///
/// ## Render Types
/// - [`Rasterizer`](crate::render::Rasterizer) — GPU rasterizer
/// - [`RasterConfig`](crate::render::RasterConfig) — Rasterizer configuration
/// - [`RenderOutput`](crate::render::RenderOutput) — Rendered image output
/// - [`RenderCamera`](crate::render::RenderCamera) — Camera for rendering
/// - [`GaussianModel`](crate::render::gaussian::GaussianModel) — Collection of Gaussians
/// - [`GaussianGradients`](crate::render::GaussianGradients) — Gradients from backward pass
/// - [`RenderError`](crate::render::RenderError) — Render-specific errors
///
/// ## Trainer Types
/// - [`Trainer`](crate::trainer::Trainer) — Training orchestrator
/// - [`TrainingConfig`](crate::trainer::TrainingConfig) — Full training configuration
/// - [`OptimizerConfig`](crate::trainer::OptimizerConfig) — Per-parameter learning rates
/// - [`LossConfig`](crate::trainer::LossConfig) — Loss function weights
/// - [`DensityConfig`](crate::trainer::DensityConfig) — Density control settings
/// - [`InitConfig`](crate::trainer::InitConfig) — Gaussian initialization settings
/// - [`TrainerError`](crate::trainer::TrainerError) — Trainer-specific errors
///
/// ## Unified Types
/// - [`OxigafError`] — Unified error type
/// - [`Result`] — Type alias for `std::result::Result<T, OxigafError>`
pub mod prelude {
    // ---- FLAME types ----
    pub use oxigaf_flame::{
        sample_mesh_surface, Camera, FlameError, FlameModel, FlameParams, FlameParamsBuilder, Mesh,
        NormalMapRenderer, SurfacePoint,
    };

    // ---- Diffusion types ----
    pub use oxigaf_diffusion::{
        ClipImageEncoder, DdimScheduler, DiffusionConfig, DiffusionError,
        MultiViewDiffusionPipeline, MultiViewOutput, MultiViewUNet, PredictionType, Vae,
    };

    // ---- Render types ----
    pub use oxigaf_render::{
        gaussian::GaussianModel, GaussianGradients, RasterConfig, Rasterizer, RenderCamera,
        RenderError, RenderOutput,
    };

    // ---- Trainer types ----
    pub use oxigaf_trainer::{
        DensityConfig, InitConfig, LossConfig, OptimizerConfig, Trainer, TrainerError,
        TrainingConfig,
    };

    // ---- Unified types ----
    pub use super::{OxigafError, Result};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
        assert!(v.contains('.'));
    }

    #[test]
    fn test_error_conversion_flame() {
        let flame_err = flame::FlameError::ModelDir("test".to_string());
        let oxigaf_err: OxigafError = flame_err.into();
        assert!(matches!(oxigaf_err, OxigafError::Flame(_)));
    }

    #[test]
    fn test_error_conversion_diffusion() {
        let diff_err = diffusion::DiffusionError::ModelLoad("test".to_string());
        let oxigaf_err: OxigafError = diff_err.into();
        assert!(matches!(oxigaf_err, OxigafError::Diffusion(_)));
    }

    #[test]
    fn test_error_conversion_render() {
        let render_err = render::RenderError::GpuInit("test".to_string());
        let oxigaf_err: OxigafError = render_err.into();
        assert!(matches!(oxigaf_err, OxigafError::Render(_)));
    }

    #[test]
    fn test_error_conversion_trainer() {
        let trainer_err = trainer::TrainerError::Init("test".to_string());
        let oxigaf_err: OxigafError = trainer_err.into();
        assert!(matches!(oxigaf_err, OxigafError::Trainer(_)));
    }

    #[test]
    fn test_prelude_exports() {
        // Verify all prelude types are accessible
        use prelude::*;

        // Just verify the types exist and are usable
        let _: fn() -> FlameParams = FlameParams::neutral;

        // Verify config defaults work
        let _config = TrainingConfig::default();
        let _opt = OptimizerConfig::default();
        let _loss = LossConfig::default();
        let _density = DensityConfig::default();
        let _init = InitConfig::default();
    }

    #[test]
    fn test_result_type_alias() {
        fn example_fn() -> Result<i32> {
            Ok(42)
        }
        assert_eq!(example_fn().ok(), Some(42));
    }
}
