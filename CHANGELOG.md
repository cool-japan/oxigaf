# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-24

### Added

#### Core Libraries

- **oxigaf-flame** — FLAME parametric 3D head model implementation
  - Linear Blend Skinning (LBS) with SIMD optimizations
  - Rodrigues rotation formula for joint transformations
  - Normal map generation from mesh geometry (CPU rasterizer)
  - Blend shapes application for facial expressions
  - Mesh sampling with barycentric coordinates
  - Support for `.npy` FLAME model files
  - **safetensors I/O** — load/save FLAME model in `.safetensors` format
  - **FlameSequence** — video frame processing with LRU caching and interpolation
  - Property-based testing with proptest
  - SIMD feature flag for vectorized operations (2-4× speedup)
  - Parallel feature flag for rayon-based parallelism

- **oxigaf-diffusion** — Multi-view diffusion pipeline
  - Multi-view U-Net architecture with cross-view attention for geometric consistency
  - **Latent Upsampler** — 32×32 → 64×64 latent upsampling for 512×512 output resolution
  - **IP-Adapter** — identity-preserving image conditioning for consistent face generation
  - **Classifier-Free Guidance (CFG)** — quality improvement with configurable guidance scale (1.0–20.0)
  - Camera pose conditioning via explicit camera pose embeddings
  - Flash Attention implementation for memory-efficient attention
  - VAE encoder/decoder for latent space operations
  - DDPM/DDIM noise scheduler support
  - CLIP text encoder integration
  - Mixed precision training support
  - CUDA and Metal GPU backend support

- **oxigaf-render** — 3D Gaussian Splatting rasterizer
  - GPU-accelerated rasterization using wgpu
  - Spherical harmonics (SH) evaluation — specialized shaders for degrees 0–3 (10× speedup)
  - Tile-based radix sort with Fuchsia-based 64-bit key sorting
  - Alpha blending with depth-ordered front-to-back traversal
  - Full backward pass: ∂L/∂color, ∂L/∂alpha, ∂L/∂conic, ∂L/∂mean2D, ∂L/∂SH, ∂L/∂scale, ∂L/∂rotation
  - **FLAME mesh binding** — barycentric coordinate binding with TBN projection
  - **FLAME binding backward pass** — ∂L/∂position → ∂L/∂local_offset for mesh-regularized training
  - **35 gradient verification tests** — numerical vs analytical comparison with <1e-3 relative error
  - Buffer pool for memory-efficient workload management (90%+ allocation reduction)
  - CPU reference rasterizer for gradient validation
  - GPU debug feature flag for validation layers

- **oxigaf-trainer** — Training and optimization pipeline
  - Gaussian model initialization from FLAME mesh surface
  - Adam optimizer with per-parameter group learning rates
  - Comprehensive loss functions:
    - L1 photometric loss
    - MS-SSIM (Multi-Scale SSIM) perceptual loss
    - LPIPS (Learned Perceptual Image Patch Similarity) — Pure Rust VGG network
    - Score Distillation Sampling (SDS) for diffusion guidance
    - Normal consistency loss
    - Opacity and scale regularization
    - Position regularization (binding to FLAME mesh)
  - Adaptive density control (split/prune/clone operations)
  - Checkpoint saving and loading
  - **TensorBoard integration** — scalar, image, histogram, and graph logging
  - Diffusion target generation for pseudo-GT supervision
  - Pipeline orchestration with modular stages and progress tracking

- **oxigaf-bridge** *(new crate)* — PyTorch ↔ OxiGAF weight conversion
  - Bidirectional weight conversion: PyTorch → OxiGAF and OxiGAF → PyTorch
  - Layer name mapping with custom overrides
  - Precision conversion (FP32 ↔ FP16 ↔ BF16)
  - Safetensors-based checkpoint interoperability
  - Validation utilities for conversion correctness
  - CLI examples: `convert_gaf_checkpoint`, `batch_convert`, `validate_conversion`

- **oxigaf-cli** — Command-line interface
  - `convert` — Convert between 3D formats and weight formats
  - `train` — Train Gaussian Avatar models
  - `render` — Render images from trained models
  - `export` — Export to standard formats
  - `benchmark` — Performance benchmarking
  - `doctor` — System diagnostics
  - Configuration hierarchy (file, environment, CLI args)
  - Progress bars and interactive prompts
  - JSON output mode for scripting
  - Log rotation and verbosity control
  - HuggingFace Hub integration
  - Asset caching system with LRU eviction

- **oxigaf** — Unified meta-crate
  - Re-exports all core APIs via `pub use`
  - Comprehensive `prelude` module with 40+ re-exported types
  - Unified `OxigafError` enum wrapping all sub-crate errors
  - Feature flag orchestration (pass-through to all sub-crates)
  - Extensive documentation: Quick Start, Data Flow diagram, Migration guide from Python GAF
  - 4 runnable examples: `basic_flame`, `gaussian_render`, `training_loop`, `diffusion_inference`

#### Development Infrastructure

- Comprehensive test suite (796 tests across all crates, all passing)
- Property-based testing with proptest
- Benchmark suite using criterion
- Examples demonstrating core workflows
- GitHub Actions CI/CD configuration
- Documentation with rustdoc examples

#### Documentation

- Crate-level documentation for all modules
- API documentation with examples
- README.md with installation and usage
- CHANGELOG.md following Keep a Changelog format
- Licensing: Apache-2.0

### Technical Highlights

- **Pure Rust Implementation** — 100% Pure Rust (no C/Fortran dependencies by default)
- **COOLJAPAN Ecosystem** — Uses oxiarc-archive instead of zip; oxiblas instead of openblas; OxiFFT instead of rustfft
- **512×512 Multi-View Generation** — Latent upsampler + IP-Adapter + CFG pipeline fully integrated
- **Verified Gradients** — 35 gradient verification tests; numerical and analytical gradients match (<1e-3 relative error) across all parameters
- **PyTorch Interoperability** — Bidirectional weight conversion via `oxigaf-bridge` crate
- **Performance Optimizations**:
  - SIMD acceleration for FLAME operations (2-4× speedup, feature-gated)
  - Flash Attention for memory-efficient diffusion
  - Specialized SH shaders (10× speedup via compile-time degree specialization)
  - Buffer pool for GPU memory efficiency (90%+ allocation reduction)
  - Parallel batch processing with rayon (near-linear with CPU cores)
  - GPU-accelerated rendering with wgpu 28
- **Code Quality**:
  - Zero `unwrap()` in production code
  - Zero warnings policy (clippy + rustc)
  - Workspace-based dependency management
  - All files under 2000 lines
  - 796 tests — 100% passing

### Statistics

- **Total Lines of Code**: ~38,000 (Rust + WGSL)
- **Crates**: 7 publishable crates
- **Test Coverage**: 796 tests (100% passing)
- **Gradient Verification Tests**: 35 (all passing, <1e-3 relative error)
- **Development Effort**: ~15 months (COCOMO estimate)

### Dependencies

- Linear Algebra: nalgebra 0.34, glam 0.31
- Deep Learning: candle-core 0.9, candle-nn 0.9, candle-transformers 0.9
- GPU Compute: wgpu 28
- Image Processing: image 0.25 (with EXR support)
- Serialization: serde 1, safetensors 0.7
- Error Handling: anyhow 1, thiserror 2
- CLI: clap 4 (with derive, env features)
- Async: tokio 1 (full features)
- Testing: approx 0.5, proptest 1, criterion 0.8

[0.1.0]: https://github.com/cool-japan/oxigaf/releases/tag/v0.1.0
