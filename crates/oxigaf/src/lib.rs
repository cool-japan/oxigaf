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
//! ```
//!
//! There is no `cuda` or `metal` Cargo feature on `oxigaf` itself. GPU
//! acceleration for the [`render`] rasterizer is automatic (wgpu picks
//! Vulkan/Metal/DX12 at runtime); GPU acceleration for [`diffusion`]
//! inference is opted into by depending on `candle-core` directly and
//! enabling its own `cuda` or `metal` feature.
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
//! GPU acceleration is not an `oxigaf` Cargo feature (there is no `cuda` or
//! `metal` feature on this crate — see the Quick Start note above). The
//! flags below only affect CPU-side behaviour.
//!
//! ### Performance Optimization Features
//!
//! | Feature | Description | Speedup |
//! |---------|-------------|---------|
//! | `simd` | SIMD-accelerated FLAME operations (requires nightly Rust) | 3-4× faster |
//! | `parallel` | Parallel batch processing with rayon | Near-linear with cores |
//! | `flash_attention` | Memory-efficient O(N) attention (enabled by default) | 2-4× less memory |
//! | `mixed_precision` | Switches the default `MixedPrecisionConfig::mode` to BF16 (rounding-simulation helpers only; nothing outside `oxigaf_diffusion::mixed_precision` reads this config yet, so no inference/training path is actually affected) | 2× faster on GPUs (once wired) |
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
//! | `all_features` | `simd`, `parallel`, `flash_attention`, `mixed_precision`, `gpu_debug` |
//!
//! ### Examples
//!
//! ```toml
//! # CPU-only with all optimizations (requires nightly for SIMD)
//! oxigaf = { version = "0.1", features = ["full_performance"] }
//!
//! # Development with GPU debugging enabled
//! oxigaf = { version = "0.1", features = ["gpu_debug"] }
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
//! | Rust      | 1.85+           | 1.85.0         |
//! | wgpu      | 30.x            | 30.x           |
//! | candle    | 0.11.x          | 0.11.x         |
//! | nalgebra  | 0.35.x          | 0.35.x         |
//! | glam      | 0.33.x          | 0.33.x         |
//!
//! Note: the workspace declares `rust-version = "1.85"`, but at the time of
//! writing it also calls `usize::is_multiple_of` (stabilized in Rust 1.87)
//! from `oxigaf-bridge/src/precision.rs` and from
//! `oxigaf-render`'s gradient-verification tests — verified to fail to
//! compile on 1.85.0/1.86.0 and succeed on 1.87.0. Until those call sites
//! are replaced (e.g. with `% 2 == 0` / `% 4 == 0`) or `rust-version` is
//! raised to match, 1.87 is the true floor.
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

pub mod pipeline;

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

    /// Invalid configuration value or combination.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// A required path does not exist.
    #[error("path not found: {0}")]
    PathNotFound(String),

    /// GPU / wgpu adapter error.
    #[error("GPU error: {0}")]
    GpuError(String),

    /// I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The pipeline or component has not been initialized.
    #[error("not initialized")]
    NotInitialized,

    /// Error with an added contextual message wrapping another error.
    ///
    /// Created by [`ErrorContext::with_ctx`] / [`ErrorContext::with_ctx_fn`].
    #[error("{context}: {source}")]
    Context {
        /// Human-readable description of what was being attempted.
        context: String,
        /// The underlying error that occurred.
        #[source]
        source: Box<OxigafError>,
    },
}

pub use pipeline::{
    check_gpu, detect_best_backend, export, quick_train, render_from_file, validate_config,
    verify_assets, ExportFormat, GpuInfo, PipelineBuilder, PipelineConfig,
};

// ============================================================================
// Error Extension Trait
// ============================================================================

/// Extension trait for adding contextual messages to `Result` values.
///
/// Wraps an inner [`OxigafError`] with a human-readable context string,
/// which is included in the [`Display`](std::fmt::Display) output of the
/// resulting error.
///
/// # Example
///
/// ```rust,no_run
/// use oxigaf::prelude::*;
///
/// fn load() -> oxigaf::Result<()> {
///     std::fs::read("missing.bin")
///         .map_err(|e| OxigafError::Io(e))
///         .with_ctx("loading FLAME model weights")?;
///     Ok(())
/// }
/// ```
pub trait ErrorContext<T> {
    /// Wrap an `Err` value with a static context string.
    ///
    /// Returns `Ok(value)` unchanged on success; on error wraps the inner
    /// [`OxigafError`] with `OxigafError::Context { context, source }`.
    fn with_ctx(self, ctx: impl Into<String>) -> std::result::Result<T, OxigafError>;

    /// Wrap an `Err` value using a lazily-evaluated closure.
    ///
    /// The closure `f` is **only called on error**, avoiding any allocation
    /// cost on the success path.
    fn with_ctx_fn<F: FnOnce() -> String>(self, f: F) -> std::result::Result<T, OxigafError>;
}

impl<T> ErrorContext<T> for std::result::Result<T, OxigafError> {
    #[inline]
    fn with_ctx(self, ctx: impl Into<String>) -> std::result::Result<T, OxigafError> {
        self.map_err(|e| OxigafError::Context {
            context: ctx.into(),
            source: Box::new(e),
        })
    }

    #[inline]
    fn with_ctx_fn<F: FnOnce() -> String>(self, f: F) -> std::result::Result<T, OxigafError> {
        self.map_err(|e| OxigafError::Context {
            context: f(),
            source: Box::new(e),
        })
    }
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

/// Convenience `Result` alias for FLAME-related operations.
pub type FlameResult<T> = std::result::Result<T, OxigafError>;

/// Convenience `Result` alias for diffusion-pipeline operations.
pub type DiffusionResult<T> = std::result::Result<T, OxigafError>;

/// Convenience `Result` alias for rasterizer/rendering operations.
pub type RenderResult<T> = std::result::Result<T, OxigafError>;

/// Convenience `Result` alias for CLI operations.
pub type CliResult<T> = std::result::Result<T, OxigafError>;

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
/// - [`ErrorContext`] — Extension trait providing `.with_ctx()` / `.with_ctx_fn()`
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
    pub use super::{
        check_gpu, detect_best_backend, export, quick_train, render_from_file, validate_config,
        verify_assets, ErrorContext, ExportFormat, GpuInfo, OxigafError, PipelineBuilder,
        PipelineConfig, Result,
    };
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

    // --- ErrorContext tests ---

    #[test]
    fn test_with_ctx_ok_passthrough() {
        let r: std::result::Result<i32, OxigafError> = Ok(42);
        let result = r.with_ctx("context should not appear");
        assert_eq!(result.ok(), Some(42));
    }

    #[test]
    fn test_with_ctx_err_wraps() {
        let r: std::result::Result<i32, OxigafError> = Err(OxigafError::NotInitialized);
        let result = r.with_ctx("loading model");
        assert!(matches!(result, Err(OxigafError::Context { .. })));
    }

    #[test]
    fn test_context_display_contains_context_string() {
        let r: std::result::Result<i32, OxigafError> = Err(OxigafError::NotInitialized);
        let result = r.with_ctx("rasterizer setup");
        assert!(result.is_err());
        if let Err(err) = result {
            let display = format!("{err}");
            assert!(
                display.contains("rasterizer setup"),
                "display was: {display}"
            );
        }
    }

    #[test]
    fn test_with_ctx_fn_not_called_on_ok() {
        let r: std::result::Result<i32, OxigafError> = Ok(99);
        let mut called = false;
        let result = r.with_ctx_fn(|| {
            called = true;
            "should not be called".to_string()
        });
        assert!(!called, "closure must not be called on Ok");
        assert_eq!(result.ok(), Some(99));
    }

    #[test]
    fn test_with_ctx_fn_called_on_err() {
        let r: std::result::Result<i32, OxigafError> = Err(OxigafError::NotInitialized);
        let mut called = false;
        let _ = r.with_ctx_fn(|| {
            called = true;
            "lazy context".to_string()
        });
        assert!(called, "closure must be called on Err");
    }

    // Regression test for `ErrorContext` being reachable via
    // `oxigaf::prelude::*` alone (the crate-level `with_ctx` doc example
    // relies on exactly this). This lives in its own submodule with no
    // `use super::*`, so it only compiles if `prelude::*` re-exports
    // `ErrorContext` directly — it would have failed to compile before that
    // re-export was added.
    mod prelude_error_context_regression {
        use crate::prelude::*;

        #[test]
        fn with_ctx_resolves_via_prelude_glob_alone() {
            let r: std::result::Result<i32, OxigafError> = Err(OxigafError::NotInitialized);
            let wrapped = r.with_ctx("prelude regression check");
            assert!(matches!(wrapped, Err(OxigafError::Context { .. })));
        }
    }

    #[test]
    fn test_flame_result_alias() {
        let _r: FlameResult<i32> = Ok(1);
        let _r2: FlameResult<i32> = Err(OxigafError::NotInitialized);
    }

    #[test]
    fn test_diffusion_result_alias() {
        let _r: DiffusionResult<i32> = Ok(1);
    }

    #[test]
    fn test_render_result_alias() {
        let _r: RenderResult<i32> = Ok(1);
    }

    #[test]
    fn test_cli_result_alias() {
        let _r: CliResult<i32> = Ok(1);
    }

    #[test]
    fn test_nested_context_display_contains_both_messages() {
        let inner: std::result::Result<i32, OxigafError> = Err(OxigafError::NotInitialized);
        let once_wrapped = inner.with_ctx("inner context");
        assert!(once_wrapped.is_err());
        if let Err(inner_err) = once_wrapped {
            let twice_wrapped = Err::<i32, OxigafError>(inner_err).with_ctx("outer context");
            assert!(twice_wrapped.is_err());
            if let Err(outer_err) = twice_wrapped {
                let display = format!("{outer_err}");
                assert!(display.contains("outer context"), "display was: {display}");
                assert!(display.contains("inner context"), "display was: {display}");
            }
        }
    }
}
