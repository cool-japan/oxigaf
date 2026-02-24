# OxiGAF — Implementation Plan

> **📋 Design Document Status**
>
> **Last Updated:** 2026-02-09
> **Implementation Status:** ~75% complete
> **Current Status:** See individual `crates/*/TODO.md` files for up-to-date progress
>
> **✨ Key Achievements Beyond Original Plan:**
> - **SIMD Acceleration** (oxigaf-flame): 2-4× speedup for Rodrigues rotation and blend shapes
> - **Flash Attention** (oxigaf-diffusion): 50% memory reduction, 30-50% faster than standard
> - **Specialized SH Shaders** (oxigaf-render): 10× speedup through compile-time degree optimization
> - **TensorBoard Integration** (oxigaf-trainer): 1,181 lines of comprehensive logging (not in plan)
> - **LPIPS in Pure Rust** (oxigaf-trainer): 689 lines, no Python dependency
> - **Unified Error Handling** (oxigaf meta crate): OxigafError enum wrapping all sub-crates
> - **Comprehensive CLI** (oxigaf-cli): 161 tests, benchmark/doctor/cache commands, HuggingFace Hub integration
> - **Buffer Pool** (oxigaf-render): Memory-efficient buffer reuse (90%+ allocation reduction)
>
> **⚠️ Significant Deviations:**
> - **No video extraction yet**: ffmpeg-next integration deferred (feature-gated in plan)
> - **No real-time preview yet**: winit integration deferred
> - **safetensors instead of .npz**: Using .npy files for FLAME model (easier conversion)
>
> **❌ Not Yet Implemented (CRITICAL for v1.0):**
> - **Diffusion Module (35% gap)**:
>   - Latent Upsampler (0%) - CRITICAL for 512×512 output
>   - IP-Adapter conditioning (0%) - CRITICAL for identity preservation
>   - Classifier-Free Guidance (0%) - CRITICAL for quality
>   - Weight conversion script (0%)
> - **Render Module (20% gap)**:
>   - Gradient verification (0%) - CRITICAL for training
>   - FLAME binding backward shader (partial)
> - **CLI (15% gap)**:
>   - Video frame extraction (0%)
>   - Real-time preview window (0%)
>   - glTF export implementation (0%)
>
> **📊 Module Status Summary:**
> - ✅ oxigaf (meta crate): 100% — Exemplary, exceeds plan
> - ✅ oxigaf-flame: 85% — Production-ready, missing sequence loading
> - ⚠️ oxigaf-diffusion: 65% — Core done, missing 3 critical GAF features
> - ⚠️ oxigaf-render: 80% — Forward pass excellent, backward needs verification
> - ✅ oxigaf-trainer: 90% — Comprehensive, blocked by diffusion gaps
> - ✅ oxigaf-cli: 85% — Feature-rich, missing video/preview
>
> **🎯 To reach v1.0 (estimated 3-4 weeks):**
> 1. Diffusion: Implement Latent Upsampler + IP-Adapter + CFG (~2 weeks)
> 2. Render: Gradient verification + FLAME binding backward (~1 week)
> 3. CLI: Video extraction + Preview window (~1 week)
>
> **📈 Test Coverage: 573 tests (all passing)**
> - oxigaf: 7 tests
> - oxigaf-flame: 43 tests
> - oxigaf-diffusion: 41 tests
> - oxigaf-render: 78 tests
> - oxigaf-trainer: 231 tests
> - oxigaf-cli: 161 tests
> - Total: **24,661 lines** of Rust code

---

> Comprehensive plan for implementing GAF (Gaussian Avatar Reconstruction from Monocular Videos via Multi-View Diffusion) in Pure Rust.

---

## Table of Contents

1. [Workspace & Project Structure](#1-workspace--project-structure)
2. [oxigaf-flame — FLAME Integration](#2-oxigaf-flame--flame-integration)
3. [oxigaf-diffusion — Multi-View Diffusion Inference](#3-oxigaf-diffusion--multi-view-diffusion-inference)
4. [oxigaf-render — Differentiable 3DGS Rasterizer](#4-oxigaf-render--differentiable-3dgs-rasterizer)
5. [oxigaf-trainer — Optimization Pipeline](#5-oxigaf-trainer--optimization-pipeline)
6. [oxigaf (Meta Crate) & oxigaf-cli (Binary)](#6-oxigaf-meta-crate--oxigaf-cli-binary)
7. [Development Roadmap & Milestones](#7-development-roadmap--milestones)
8. [Risk Registry](#8-risk-registry)

---

## 1. Workspace & Project Structure

### 1.1 Layout — Virtual Cargo Workspace

```
oxigaf/
├── Cargo.toml                    # [workspace] — virtual manifest
├── Cargo.lock
├── oxigaf.md                     # Design document
├── IMPLEMENTATION_PLAN.md        # This file
├── README.md
├── LICENSE
├── .github/
│   └── workflows/
│       ├── ci.yml                # Lint + test (CPU / llvmpipe)
│       └── release.yml           # Tagged binary builds
├── assets/
│   └── models/                   # .gitignored — downloaded at runtime
├── scripts/
│   ├── convert_flame.py          # .pkl → .npz conversion
│   └── convert_weights.py        # PyTorch → SafeTensors conversion
├── crates/
│   ├── oxigaf-flame/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── model.rs          # FlameModel, BlendShapes, LBS
│   │   │   ├── params.rs         # FlameParams (shape, expr, pose)
│   │   │   ├── mesh.rs           # Mesh topology, vertex ops
│   │   │   ├── normal_map.rs     # Normal map rasterizer
│   │   │   ├── sampler.rs        # Surface point sampling for Gaussians
│   │   │   └── io.rs             # .npz / .json file loading
│   │   └── tests/
│   │       └── flame_tests.rs
│   │
│   ├── oxigaf-diffusion/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── unet.rs           # MultiViewUNet (modified SD 2.1)
│   │   │   ├── attention.rs      # MultiViewTransformerBlock (cross-view attn)
│   │   │   ├── vae.rs            # VAE encoder/decoder
│   │   │   ├── clip.rs           # CLIP image encoder
│   │   │   ├── camera.rs         # CameraEmbedding MLP
│   │   │   ├── scheduler.rs      # DDIM scheduler (v-prediction)
│   │   │   ├── upsampler.rs      # Latent Upsampler (sd-x2-latent-upscaler)
│   │   │   ├── pipeline.rs       # MultiViewDiffusionPipeline
│   │   │   └── config.rs         # DiffusionConfig
│   │   └── tests/
│   │       └── diffusion_tests.rs
│   │
│   ├── oxigaf-render/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── gaussian.rs       # GaussianModel, attributes
│   │   │   ├── rasterizer.rs     # Rasterizer (forward + backward)
│   │   │   ├── pipeline.rs       # GPU pipeline setup, dispatches
│   │   │   ├── buffers.rs        # GPU buffer management
│   │   │   ├── sort.rs           # Radix sort wrapper
│   │   │   ├── binding.rs        # FLAME mesh binding / regularization
│   │   │   └── config.rs         # RasterConfig
│   │   ├── shaders/
│   │   │   ├── preprocess.wgsl       # Projection, cov3D→cov2D
│   │   │   ├── prefix_sum.wgsl       # Inclusive prefix sum
│   │   │   ├── tile_assign.wgsl      # Assign Gaussians to tiles
│   │   │   ├── radix_sort.wgsl       # GPU radix sort
│   │   │   ├── tile_ranges.wgsl      # Compute per-tile ranges
│   │   │   ├── rasterize_fwd.wgsl    # Forward alpha-blending
│   │   │   ├── rasterize_bwd.wgsl    # Backward rasterization
│   │   │   ├── cov2d_bwd.wgsl        # Cov2D backward
│   │   │   ├── preprocess_bwd.wgsl   # Preprocess backward
│   │   │   ├── sh_eval.wgsl          # Spherical harmonics evaluation
│   │   │   ├── flame_binding.wgsl    # FLAME mesh binding forward
│   │   │   └── flame_binding_bwd.wgsl # FLAME mesh binding backward
│   │   └── tests/
│   │       └── render_tests.rs
│   │
│   ├── oxigaf-trainer/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── trainer.rs        # Main Trainer loop
│   │   │   ├── config.rs         # TrainingConfig
│   │   │   ├── optimizer.rs      # Per-parameter Adam optimizer
│   │   │   ├── loss.rs           # Loss functions (photometric, reg, perceptual)
│   │   │   ├── init.rs           # Gaussian initialization on FLAME mesh
│   │   │   ├── density.rs        # Adaptive density control (split/clone/prune)
│   │   │   ├── checkpoint.rs     # Save/load training state
│   │   │   └── metrics.rs        # PSNR, SSIM, LPIPS tracking
│   │   └── tests/
│   │       └── trainer_tests.rs
│   │
│   ├── oxigaf/                   # Meta crate — re-exports unified public API
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs            # pub use oxigaf_{flame,diffusion,render,trainer}
│   │
│   └── oxigaf-cli/               # CLI binary
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           ├── cli.rs            # clap command definitions
│           ├── config.rs         # TOML config loading
│           ├── pipeline.rs       # End-to-end orchestration
│           ├── video.rs          # Video frame extraction
│           ├── preview.rs        # Real-time wgpu+winit viewer
│           ├── export.rs         # PLY / glTF / image export
│           └── assets.rs         # Model download & cache
```

### 1.2 Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "crates/oxigaf-flame",
    "crates/oxigaf-diffusion",
    "crates/oxigaf-render",
    "crates/oxigaf-trainer",
    "crates/oxigaf",         # meta crate (library)
    "crates/oxigaf-cli",     # CLI binary
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
repository = "https://github.com/cool-japan/oxigaf"

[workspace.dependencies]
# Linear Algebra
nalgebra = "0.34"
glam = "0.31"

# AI / Deep Learning (Pure Rust)
candle-core = { version = "0.9", features = [] }  # cuda/metal via feature flags
candle-nn = "0.9"
candle-transformers = "0.9"
safetensors = "0.5"

# GPU Graphics & Compute
wgpu = "28"
bytemuck = { version = "1.21", features = ["derive"] }

# Image & Video
image = "0.25"
ffmpeg-next = "7"

# Async & Utilities
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"
anyhow = "1.0"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = "0.3"
indicatif = "0.17"
clap = { version = "4", features = ["derive"] }

# Numerical
ndarray = "0.16"
ndarray-npy = "0.9"
kiddo = "4"

# Windowing (optional, for preview)
winit = "0.30"

# Internal crates
oxigaf-flame = { path = "crates/oxigaf-flame" }
oxigaf-diffusion = { path = "crates/oxigaf-diffusion" }
oxigaf-render = { path = "crates/oxigaf-render" }
oxigaf-trainer = { path = "crates/oxigaf-trainer" }
oxigaf = { path = "crates/oxigaf" }

# Testing
approx = "0.5"
```

### 1.3 Feature Flag Design

Feature flags propagate **top-down** — the user enables at `oxigaf-cli` or `oxigaf` (meta) level:

| Feature | Propagates To | Effect |
|---------|--------------|--------|
| `cuda` | oxigaf → oxigaf-diffusion → candle-core | Enable CUDA backend for diffusion inference |
| `metal` | oxigaf → oxigaf-diffusion → candle-core | Enable Metal backend for diffusion inference |
| `video` | oxigaf-cli → ffmpeg-next | Enable video input (requires system FFmpeg libs) |
| `preview` | oxigaf-cli → winit, wgpu surface | Enable real-time preview window |
| `wgpu-normal` | oxigaf → oxigaf-flame | GPU-accelerated normal map rendering (vs CPU default) |
| `full` | oxigaf (meta) | Enable all sub-crate features — convenience for library consumers |

Default features for `oxigaf-cli`: `["video", "preview"]`
Default features for `oxigaf` (meta): `[]` (consumers pick what they need)

### 1.4 CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - cargo fmt --check
      - cargo clippy --all-targets -- -D warnings

  test-cpu:
    runs-on: ubuntu-latest
    env:
      WGPU_BACKEND: gl                     # Mesa llvmpipe software renderer
    steps:
      - apt-get install mesa-vulkan-drivers mesa-utils libgl1-mesa-dri
      - cargo test --workspace --no-default-features  # No video/preview

  test-gpu:
    runs-on: [self-hosted, gpu]             # Optional GPU runner
    steps:
      - cargo test --workspace --features cuda
```

**Key CI decisions:**
- `llvmpipe` (Mesa software OpenGL) enables wgpu compute shader tests on standard runners via `WGPU_BACKEND=gl`
- `ffmpeg-next` is feature-gated; CI tests run without it by default
- GPU integration tests on self-hosted runners (optional, for CUDA path)

---

## 2. oxigaf-flame — FLAME Integration

### 2.1 FLAME Model Background

FLAME (Faces Learned with an Articulated Model and Expressions) is a parametric 3D head model:
- **Mesh**: 5,023 vertices, 9,976 triangle faces
- **Joint tree**: 5 joints — root → neck → {jaw, left_eye, right_eye}
- **Parameter spaces**:
  - Shape (β): 300 PCA components (typically use first 100)
  - Expression (ψ): 100 PCA components (typically use first 50)
  - Pose (θ): 5 joints × 3 axis-angle = 15 DoF + 3D translation
- **Model weights** (stored in `.pkl`): `v_template` [5023,3], `shapedirs` [5023,3,400], `posedirs` [5023,3,36], `J_regressor` [5,5023] sparse, `kintree_table` [2,5], `weights` [5023,5]

### 2.2 Data Format Strategy

**Offline Python conversion** (preferred over implementing a Rust pickle parser):

```
scripts/convert_flame.py:  FLAME2023.pkl → flame_model.npz
```

The `.npz` file contains named arrays loaded via `ndarray-npy` in Rust. Alternatively, convert to a set of `.npy` files or a single `.safetensors` file.

### 2.3 Key Structs

```rust
/// Loaded FLAME model (immutable after load)
pub struct FlameModel {
    pub v_template: Array2<f32>,        // [5023, 3]
    pub faces: Array2<u32>,             // [9976, 3]
    pub shapedirs: Array3<f32>,         // [5023, 3, 300]
    pub posedirs: Array3<f32>,          // [5023, 3, 36]
    pub j_regressor: CsrMatrix<f32>,    // [5, 5023] sparse
    pub kintree_table: Array2<i32>,     // [2, 5]
    pub lbs_weights: Array2<f32>,       // [5023, 5]
}

/// Per-frame FLAME parameters
pub struct FlameParams {
    pub shape: Vec<f32>,       // [n_shape] — identity coefficients
    pub expression: Vec<f32>,  // [n_expr]  — expression coefficients
    pub pose: Vec<f32>,        // [15]      — axis-angle per joint
    pub translation: Vec3,     // [3]       — global translation
}

/// Triangle mesh with computed positions + normals
pub struct Mesh {
    pub vertices: Vec<Vec3>,   // [5023]
    pub normals: Vec<Vec3>,    // [5023] per-vertex normals
    pub faces: Vec<[u32; 3]>,  // [9976]
}
```

### 2.4 Key Functions

```rust
impl FlameModel {
    pub fn load(path: &Path) -> Result<Self>;

    /// Core LBS pipeline: params → posed mesh
    pub fn forward(&self, params: &FlameParams) -> Mesh;

    // Internal steps:
    fn apply_blend_shapes(&self, params: &FlameParams) -> Array2<f32>;
    fn compute_joints(&self, vertices: &Array2<f32>) -> Vec<Mat4>;
    fn rodrigues(axis_angle: &Vec3) -> Mat3;
    fn apply_lbs(&self, vertices: &Array2<f32>, transforms: &[Mat4]) -> Vec<Vec3>;
}

/// CPU software rasterizer for normal maps (Phase 1)
pub struct NormalMapRenderer {
    pub width: u32,
    pub height: u32,
}

impl NormalMapRenderer {
    pub fn render(&self, mesh: &Mesh, camera: &Camera) -> Image<Rgb<f32>>;
}

/// Trait for oxigaf-diffusion integration
pub trait NormalMapProvider {
    fn generate(&self, params: &FlameParams, camera: &Camera) -> Image<Rgb<f32>>;
}

/// Trait for oxigaf-render Gaussian initialization
pub trait MeshSurfaceSampler {
    fn sample_points(&self, mesh: &Mesh, count: usize) -> Vec<SurfacePoint>;
}

pub struct SurfacePoint {
    pub position: Vec3,
    pub normal: Vec3,
    pub face_index: u32,
    pub barycentric: Vec3,  // (u, v, w)
}
```

### 2.5 Normal Map Generation

**Phase 1: CPU Software Rasterizer** (sufficient for 10K triangles at 512×512 — <5ms):
1. Transform vertices to camera space
2. Project to screen space
3. Rasterize triangles with depth buffer (scanline or half-space)
4. Output per-pixel world-space normal as RGB

**Phase 2 (optional): wgpu render pipeline** behind `wgpu-normal` feature flag for batch rendering.

### 2.6 Validation Strategy

- Run the original Python `FLAME_PyTorch` code to generate ground-truth vertex positions for a set of parameter inputs
- Compare Rust output with tolerance < 1e-4
- Visual comparison of generated normal maps

---

## 3. oxigaf-diffusion — Multi-View Diffusion Inference

### 3.1 Model Architecture

GAF's multi-view diffusion model is based on **Stable Diffusion 2.1** (via ImageDream architecture):

| Aspect | Detail |
|--------|--------|
| Base | SD 2.1 U-Net (v-prediction) |
| Input channels | 8 (4 latent + 4 VAE-encoded normal map, concatenated) |
| Views | N=4 simultaneous views |
| Multi-view consistency | Cross-view attention in every transformer block |
| Reference conditioning | CLIP image embedding via IP cross-attention |
| Camera conditioning | Camera pose MLP → added to timestep embedding |
| Resolution | 256×256 (latent 32×32), upsampled to 512×512 |

**Architecture per transformer block:**
1. Self-attention (within each view)
2. Cross-view attention (across N=4 views) — **new**
3. Cross-attention to CLIP text embedding (from SD 2.1)
4. IP cross-attention to reference image features — **new**
5. Feed-forward network

### 3.2 Candle Integration Strategy

| Component | Status | Strategy |
|-----------|--------|----------|
| VAE (encode/decode) | ✅ Exists in candle-transformers | Direct reuse |
| ResnetBlock2D | ✅ Exists | Direct reuse |
| DDIM Scheduler | ✅ Exists | Reuse, switch to v-prediction |
| Timestep Embedding | ✅ Exists | Direct reuse |
| CrossAttnDownBlock / UpBlock | ⚠️ Must modify | Replace SpatialTransformer with MultiViewSpatialTransformer |
| MultiViewUNet | 🔨 New | Custom — wraps modified SD 2.1 UNet |
| MultiViewTransformerBlock | 🔨 New | 4 attention layers + FFN per block |
| CameraEmbedding MLP | 🔨 New | 2-layer MLP: [12] → [1280] |
| CLIP Image Encoder | 🔨 New | ViT-H/14 image encoder |

### 3.3 Weight Loading Pipeline

**Offline conversion** (Python script):
```
scripts/convert_weights.py:
  GAF PyTorch checkpoint → 4 SafeTensors files:
    unet.safetensors      (~1.7 GB fp16)
    vae.safetensors        (~320 MB fp16)
    clip.safetensors       (~1.2 GB fp16)
    upsampler.safetensors  (~200 MB fp16)
```

Layer name mapping from PyTorch (`input_blocks.{i}.{j}`) → candle VarBuilder paths (`down_blocks.{i}.resnets.{j}`). Memory-mapped I/O via `safetensors::MmapedSafetensors`.

### 3.4 Inference Pipeline (7 Steps)

```
Input: reference_image [H,W,3], normal_maps [N,H,W,3], cameras [N,12]
  │
  ├─ Step 1: CLIP encode reference_image → clip_embeds [1,257,1280]
  ├─ Step 2: VAE encode normal_maps → normal_latents [N,4,32,32]
  ├─ Step 3: VAE encode reference_image → ref_latents [1,4,32,32]
  │
  ├─ Step 4: DDIM Denoising Loop (50 steps)
  │    │  noise [N,4,32,32] ← randn
  │    │  For t in scheduler.timesteps:
  │    │    latent_input = concat(noisy_latent, normal_latents)  → [N,8,32,32]
  │    │    cam_embed = camera_mlp(cameras)                      → [N,1280]
  │    │    t_embed = timestep_embed(t) + cam_embed
  │    │    noise_pred = unet(latent_input, t_embed, clip_embeds, ref_latents)
  │    │    # CFG: noise_pred = uncond + guidance_scale * (cond - uncond)
  │    │    noisy_latent = scheduler.step(noise_pred, t, noisy_latent)
  │    └─ denoised_latents [N,4,32,32]
  │
  ├─ Step 5: Latent Upsampler (10-step DDIM)
  │    denoised_latents [N,4,32,32] → upsampled_latents [N,4,64,64]
  │
  └─ Step 6: VAE decode → multi_view_images [N,3,512,512]

Output: N=4 multi-view consistent images at 512×512
```

### 3.5 Latent Upsampler

Based on `stabilityai/sd-x2-latent-upscaler`:
- 10-step DDIM denoising
- Input: bilinear-upsampled latent 32×32 → 64×64 as conditioning
- Fallback: `BilinearVae` mode (skip upsampler, bilinear upsample + VAE decode) for speed during development

### 3.6 Key Structs

```rust
pub struct MultiViewDiffusionPipeline {
    unet: MultiViewUNet,
    vae: AutoEncoderKL,
    clip_encoder: ClipImageEncoder,
    scheduler: DDIMScheduler,
    upsampler: Option<LatentUpsampler>,
    device: Device,
}

pub struct DiffusionConfig {
    pub num_views: usize,           // default: 4
    pub guidance_scale: f64,        // default: 3.0
    pub num_inference_steps: usize, // default: 50
    pub upsampler_steps: usize,     // default: 10
    pub image_size: usize,          // default: 256 (before upscale)
}

pub struct MultiViewOutput {
    pub images: Vec<Tensor>,        // [N] × [3, H, W]
    pub latents: Vec<Tensor>,       // [N] × [4, h, w] (optional, for loss)
}

/// Trait for trainer integration
pub trait MultiViewGenerator {
    fn generate(
        &self,
        reference: &Tensor,
        normal_maps: &[Tensor],
        cameras: &[CameraParams],
    ) -> Result<MultiViewOutput>;
}

impl MultiViewDiffusionPipeline {
    pub fn load(config: &DiffusionConfig, weights_dir: &Path, device: Device) -> Result<Self>;
    pub fn generate(&self, input: &DiffusionInput) -> Result<MultiViewOutput>;
}
```

### 3.7 Performance Estimates

| Metric | Value |
|--------|-------|
| Model size (fp16) | ~3.4 GB total |
| Peak GPU memory | ~5.2 GB |
| Inference time (A100) | ~8 seconds |
| Inference time (RTX 3090) | ~15 seconds |
| Inference time (CPU) | ~10 minutes |

**Optimization strategies:**
- Attention slicing (chunked attention for lower memory)
- Flash attention (if available in candle)
- Sequential VAE encoding (one view at a time)
- fp16 throughout (candle `DType::F16`)

### 3.8 Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Weight name mapping errors | 🔴 High | Layer-by-layer validation script comparing candle vs PyTorch outputs |
| Cross-view attention correctness | 🔴 High | Unit test with known input/output pairs from Python reference |
| V-prediction vs ε-prediction mismatch | 🟡 Medium | Verify scheduler config matches original training |
| Missing candle ops | 🟡 Medium | Audit required ops before implementation; contribute upstream if needed |
| fp16 numerical precision | 🟡 Medium | Compare with fp32 reference; use fp32 for sensitive ops |

---

## 4. oxigaf-render — Differentiable 3DGS Rasterizer

### 4.1 3D Gaussian Splatting Algorithm Summary

Each 3D Gaussian is parameterized by:
- **Position** μ ∈ ℝ³
- **Covariance** Σ ∈ ℝ³ˣ³ (represented as rotation quaternion q + scale s)
- **Opacity** α ∈ [0,1]
- **Color** — Spherical Harmonics coefficients (degree 0–3, up to 48 coefficients)

Forward pipeline:
1. **Frustum culling** — discard Gaussians outside view frustum
2. **Projection** — 3D Gaussian → 2D Gaussian (project mean, compute 2D covariance via J·Σ·Jᵀ)
3. **Tile assignment** — assign each 2D Gaussian to overlapping 16×16 pixel tiles
4. **Depth sorting** — radix sort by (tile_id, depth) key
5. **Rasterization** — per-tile front-to-back alpha blending

### 4.2 Compute Shader Pipeline — 11 Dispatches Per Iteration

#### Forward Pass (6 dispatches):

| # | Shader | Workgroups | Description |
|---|--------|-----------|-------------|
| 1 | `preprocess.wgsl` | ceil(N/256) | Project 3D→2D, compute cov2D, SH→RGB, frustum cull |
| 2 | `prefix_sum.wgsl` | varies | Inclusive prefix sum of per-Gaussian tile counts |
| 3 | `tile_assign.wgsl` | ceil(N/256) | Write (tile_id, depth) keys for each Gaussian-tile pair |
| 4 | `radix_sort.wgsl` | varies | GPU radix sort of keys (adapted from web-splat/Fuchsia) |
| 5 | `tile_ranges.wgsl` | ceil(K/256) | Find start/end indices per tile in sorted array |
| 6 | `rasterize_fwd.wgsl` | (W/16)×(H/16) | Per-tile alpha blending, output color + depth + transmittance |

#### Backward Pass (4 dispatches):

| # | Shader | Description |
|---|--------|-------------|
| 7 | `rasterize_bwd.wgsl` | Reverse-order tile traversal, dL/d(color, opacity, cov2D) |
| 8 | `cov2d_bwd.wgsl` | dL/d(cov2D) → dL/d(cov3D) → dL/d(quaternion, scale) |
| 9 | `preprocess_bwd.wgsl` | dL/d(mean3D), chain through projection Jacobian |
| 10 | `flame_binding_bwd.wgsl` | dL/d(binding offsets) for FLAME regularization |

Plus radix sort internal passes (4–8 additional dispatches).

### 4.3 GPU Buffer Layout

```
Gaussian Attribute Buffers (read in forward, write gradients in backward):
  positions:   Buffer<[f32; 3]>    × N    (12 bytes/Gaussian)
  rotations:   Buffer<[f32; 4]>    × N    (16 bytes — quaternion)
  scales:      Buffer<[f32; 3]>    × N    (12 bytes)
  opacities:   Buffer<f32>         × N    (4 bytes)
  sh_coeffs:   Buffer<[f32; 48]>   × N    (192 bytes — degree 3)
                                           Total: ~236 bytes/Gaussian

Intermediate Buffers (per-frame, transient):
  cov2d:       Buffer<[f32; 3]>    × N    (upper-tri 2×2)
  means2d:     Buffer<[f32; 2]>    × N
  depths:      Buffer<f32>         × N
  radii:       Buffer<i32>         × N
  tile_counts: Buffer<u32>         × N
  sort_keys:   Buffer<u64>         × K    (K = total Gaussian-tile pairs)
  sort_vals:   Buffer<u32>         × K
  tile_ranges: Buffer<[u32; 2]>    × T    (T = number of tiles)

Output Buffers:
  color:       Buffer<[f32; 4]>    × W×H
  depth:       Buffer<f32>         × W×H
  transmit:    Buffer<f32>         × W×H  (final transmittance, for backward)

Gradient Buffers (same layout as attribute buffers):
  grad_positions, grad_rotations, grad_scales, grad_opacities, grad_sh_coeffs
```

**Memory estimate at 100K Gaussians, 512×512:**
- Attribute buffers: ~23 MB
- Intermediate: ~15 MB (assuming avg 4 tiles/Gaussian → K=400K)
- Output: ~5 MB
- Gradient buffers: ~23 MB
- **Total: ~66 MB** — well within GPU budget

### 4.4 Backward Pass Strategy

**Recommended: Hand-written WGSL backward shaders** (Approach A)

Rationale:
- The rasterization backward kernel **must** run on GPU (per-pixel × per-Gaussian)
- `burn-autodiff` operates at the tensor level, not at the per-pixel shader level
- Hand-written backward matches the original 3DGS CUDA implementation approach
- The rasterizer exposes a `forward()` → `RenderOutput` and `backward(grad_output)` → `GaussianGradients` API that can integrate with burn-autodiff as a custom op for the outer optimization loop

**f32 atomicAdd workaround**: WGSL lacks native f32 atomicAdd. Use `atomicCompareExchangeWeak` CAS loop (standard WebGPU practice):
```wgsl
fn atomic_add_f32(addr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old = atomicLoad(addr);
    loop {
        let new_val = bitcast<f32>(old) + val;
        let result = atomicCompareExchangeWeak(addr, old, bitcast<u32>(new_val));
        if result.exchanged { break; }
        old = result.old_value;
    }
}
```

### 4.5 FLAME Mesh Binding

GAF binds Gaussians to the FLAME mesh surface:

- **Rigid Gaussians**: Transform directly with the nearest FLAME vertex/face. Position = face_transform × local_offset
- **Flexible Gaussians**: Soft binding with learned offset from nearest face, regularized to stay close

**Initialization**:
1. Sample N points on FLAME mesh faces (barycentric sampling via `MeshSurfaceSampler`)
2. Assign each Gaussian a `face_index` and `barycentric_coords`
3. Initial scale ∝ face area, initial opacity = 0.5, initial rotation = face normal orientation

**Regularization losses** (computed in `flame_binding.wgsl`):
- Position reg: $L_{pos} = \sum_i \| \mu_i - T_i(\Delta_i) \|^2$ where $T_i$ is the face transform and $\Delta_i$ is the learned offset
- Scale reg: $L_{scale} = \sum_i \max(0, s_i - s_{max})^2$ to prevent Gaussians from growing too large

### 4.6 Key Structs

```rust
pub struct GaussianModel {
    pub count: usize,
    pub positions: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,    // quaternions
    pub scales: Vec<[f32; 3]>,
    pub opacities: Vec<f32>,
    pub sh_coeffs: Vec<[f32; 48]>,   // SH degree 3
    // FLAME binding
    pub face_indices: Vec<u32>,
    pub barycentric: Vec<[f32; 3]>,
    pub local_offsets: Vec<[f32; 3]>,
}

pub struct RasterConfig {
    pub image_width: u32,
    pub image_height: u32,
    pub tile_size: u32,              // default: 16
    pub sh_degree: u32,              // 0–3
    pub near_plane: f32,
    pub far_plane: f32,
    pub background: [f32; 3],
}

pub struct RenderOutput {
    pub color: Texture,              // [H, W, 4]
    pub depth: Texture,              // [H, W]
    // Retained for backward pass:
    transmittance: Buffer,
    sorted_indices: Buffer,
    tile_ranges: Buffer,
}

pub struct GaussianGradients {
    pub grad_positions: Vec<[f32; 3]>,
    pub grad_rotations: Vec<[f32; 4]>,
    pub grad_scales: Vec<[f32; 3]>,
    pub grad_opacities: Vec<f32>,
    pub grad_sh_coeffs: Vec<[f32; 48]>,
    pub grad_local_offsets: Vec<[f32; 3]>,
}

pub struct Rasterizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: RasterConfig,
    pipelines: RasterPipelines,  // compiled compute pipelines
    buffers: GpuBuffers,
}

impl Rasterizer {
    pub fn new(device: &wgpu::Device, config: RasterConfig) -> Result<Self>;
    pub fn forward(&mut self, model: &GaussianModel, camera: &Camera) -> Result<RenderOutput>;
    pub fn backward(&mut self, grad_output: &Tensor, render_ctx: &RenderOutput) -> Result<GaussianGradients>;
    pub fn upload_gaussians(&mut self, model: &GaussianModel);
    pub fn download_image(&self, output: &RenderOutput) -> Image<Rgba<f32>>;
}
```

### 4.7 Existing Work to Leverage

- **web-splat** (MIT): Fuchsia radix sort port, SH evaluation WGSL, covariance projection — directly reusable
- **gsplat** (Python reference): Algorithm reference for forward/backward kernels
- **3D Gaussian Splatting (original)**: CUDA kernel logic to port to WGSL
- **burn-autodiff**: For outer-loop gradient tracking (Gaussian params → loss)

### 4.8 Performance Targets

| Metric | Target |
|--------|--------|
| Forward pass (512×512, 100K Gaussians) | < 20 ms |
| Backward pass | < 30 ms |
| Forward + backward | < 50 ms |
| Max Gaussians | 500K |
| Memory budget | < 2 GB GPU |

### 4.9 Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| wgpu compute shader limitations (no shared memory atomics on some backends) | 🔴 High | Test on Vulkan first; fallback to multiple passes |
| Backward shader numerical stability | 🔴 High | Finite-difference gradient verification |
| Radix sort performance on non-Vulkan backends | 🟡 Medium | Benchmark; consider bitonic sort fallback |
| WGSL f32 atomicAdd CAS loop performance | 🟡 Medium | Profile; consider tiled gradient accumulation |
| WebGPU spec changes affecting wgpu API | 🟡 Medium | Pin wgpu version; update periodically |

---

## 5. oxigaf-trainer — Optimization Pipeline

### 5.1 Training Algorithm Overview

GAF uses **Iterative Denoising Distillation** — a variant of Score Distillation Sampling (SDS) that generates explicit pseudo ground-truth images instead of using noisy gradient signals.

**Per-iteration breakdown:**
1. Select a batch of camera viewpoints (N=4 views)
2. Compute FLAME mesh for current frame → generate Normal Maps
3. Render current Gaussians from N viewpoints (oxigaf-render forward)
4. Feed rendered images + Normal Maps to diffusion model → Pseudo-GT images
5. Compute losses between rendered and Pseudo-GT
6. Backpropagate through rasterizer (oxigaf-render backward)
7. Update Gaussian attributes via Adam optimizer
8. (Every K iterations) Adaptive density control

**Training schedule:**
- Total iterations: ~15,000–30,000 per identity
- Warmup: 1,000 iterations (lower learning rate, no density control)
- Density control: every 500 iterations from iteration 1,000 to 12,000
- Diffusion guidance scale annealing: start high (7.5), decay to 3.0

### 5.2 Gaussian Initialization

```rust
pub struct GaussianInitializer;

impl GaussianInitializer {
    /// Initialize Gaussians on FLAME mesh surface
    pub fn initialize(
        mesh: &Mesh,
        sampler: &dyn MeshSurfaceSampler,
        config: &InitConfig,
    ) -> GaussianModel {
        // 1. Sample N_rigid + N_flexible points on mesh
        // 2. Rigid: fixed barycentric binding, small local_offset = 0
        // 3. Flexible: learnable offset, initialized to 0
        // 4. Initial scale ∝ sqrt(face_area) * 0.5
        // 5. Initial opacity = inverse_sigmoid(0.1)
        // 6. Initial rotation = quaternion_from_normal(face_normal)
        // 7. Initial SH = [mean_color, 0, 0, ...] (DC only)
    }
}

pub struct InitConfig {
    pub num_rigid: usize,       // default: 50,000
    pub num_flexible: usize,    // default: 50,000
    pub initial_scale_factor: f32,
    pub initial_opacity: f32,
}
```

**Rigid vs Flexible split:**
- **Rigid** (~50%): Positions are strictly determined by FLAME mesh transform. Only color (SH), opacity, and small scale adjustments are optimized. Good for smooth skin areas.
- **Flexible** (~50%): Positions have a learnable offset from the mesh surface. Can capture hair, ears, fine details that deviate from FLAME topology.

### 5.3 Loss Functions

```rust
pub struct LossComputer {
    lpips: Option<LpipsNetwork>,  // candle-based VGG feature extractor
}

impl LossComputer {
    pub fn compute(&self, rendered: &Tensor, pseudo_gt: &Tensor, model: &GaussianModel, mesh: &Mesh) -> LossBundle;
}

pub struct LossBundle {
    pub total: f32,
    pub photometric: f32,     // L1 + λ_ssim * (1 - SSIM)
    pub perceptual: f32,      // LPIPS
    pub position_reg: f32,    // binding regularization
    pub scale_reg: f32,       // prevent oversized Gaussians
    pub opacity_reg: f32,     // encourage sparse opacity
    pub normal_consistency: f32,  // rendered normals vs FLAME normals
}
```

**Loss formulation:**

$$L_{total} = \lambda_1 L_{photo} + \lambda_2 L_{LPIPS} + \lambda_3 L_{pos} + \lambda_4 L_{scale} + \lambda_5 L_{opacity} + \lambda_6 L_{normal}$$

| Loss | Formula | Weight |
|------|---------|--------|
| Photometric | $(1-\lambda_{ssim}) \cdot L_1 + \lambda_{ssim} \cdot (1 - \text{SSIM})$ | 1.0 |
| Perceptual (LPIPS) | VGG feature distance | 0.1 |
| Position Reg | $\sum \|\mu_i - T_i(\Delta_i)\|^2$ | 0.01 |
| Scale Reg | $\sum \max(0, s_i - s_{max})^2$ | 0.01 |
| Opacity Reg | $\sum \|o_i\|$ (L1 sparsity) | 0.001 |
| Normal Consistency | $\sum (1 - n_{rendered} \cdot n_{flame})$ | 0.05 |

### 5.4 Optimizer Design

**Per-parameter Adam** with different learning rates (following original 3DGS):

```rust
pub struct GaussianOptimizer {
    position_lr: LearningRateSchedule,  // 1.6e-4 → 1.6e-6 (exponential decay)
    rotation_lr: f32,                    // 1e-3
    scale_lr: f32,                       // 5e-3
    opacity_lr: f32,                     // 5e-2
    sh_lr: f32,                          // 2.5e-3 (DC), 2.5e-3 / 20 (higher)
    offset_lr: f32,                      // 1e-4 (flexible binding offsets)

    // Adam state per parameter group
    states: HashMap<ParamGroup, AdamState>,
}

pub struct AdamState {
    pub m: Vec<f32>,   // first moment
    pub v: Vec<f32>,   // second moment
    pub step: u64,
    pub beta1: f32,    // 0.9
    pub beta2: f32,    // 0.999
    pub eps: f32,      // 1e-15
}

pub struct LearningRateSchedule {
    pub initial: f32,
    pub final_lr: f32,
    pub warmup_steps: u32,
    pub decay_steps: u32,
}

impl GaussianOptimizer {
    pub fn step(&mut self, model: &mut GaussianModel, grads: &GaussianGradients, iteration: u32);
}
```

### 5.5 Adaptive Density Control

Every 500 iterations (from iteration 1,000 to 12,000):

```rust
pub struct DensityController {
    pub grad_threshold: f32,        // 0.0002
    pub min_opacity: f32,           // 0.005
    pub max_screen_size: f32,       // 20 pixels
    pub split_scale_factor: f32,    // 1.6
    pub clone_scale_threshold: f32, // scene_extent * 0.01
}

impl DensityController {
    /// Run density control on the model
    pub fn step(&self, model: &mut GaussianModel, grad_accum: &GradientAccumulator) {
        // 1. PRUNE: Remove Gaussians with opacity < min_opacity
        //    or screen-space size > max_screen_size
        // 2. SPLIT: Gaussians with large position gradient AND large scale
        //    → replace with 2 smaller Gaussians (scale / split_scale_factor)
        // 3. CLONE: Gaussians with large position gradient AND small scale
        //    → duplicate and shift slightly along gradient direction
        // 4. Reset optimizer state for new/modified Gaussians
        // 5. (Every 3000 iter) Reset all opacities to 0.01 to re-evaluate
    }
}

pub struct GradientAccumulator {
    pub position_grad_norm: Vec<f32>,  // accumulated ||∇μ||
    pub count: Vec<u32>,               // number of accumulations
}
```

### 5.6 Training Loop

```rust
pub struct Trainer {
    flame: FlameModel,
    diffusion: MultiViewDiffusionPipeline,
    rasterizer: Rasterizer,
    optimizer: GaussianOptimizer,
    density_ctrl: DensityController,
    loss_computer: LossComputer,
    config: TrainingConfig,
}

pub struct TrainingConfig {
    pub total_iterations: u32,          // 15,000
    pub views_per_step: usize,          // 4
    pub density_control_interval: u32,  // 500
    pub density_control_start: u32,     // 1,000
    pub density_control_end: u32,       // 12,000
    pub checkpoint_interval: u32,       // 1,000
    pub log_interval: u32,              // 50
    pub guidance_scale_start: f32,      // 7.5
    pub guidance_scale_end: f32,        // 3.0
    pub guidance_anneal_steps: u32,     // 10,000
}

impl Trainer {
    pub fn train(&mut self, flame_params: &[FlameParams]) -> Result<GaussianModel> {
        let mut model = GaussianInitializer::initialize(...);

        for iter in 0..self.config.total_iterations {
            // 1. Sample camera viewpoints
            let cameras = self.sample_cameras(iter);

            // 2. Get FLAME mesh & normal maps for current frame
            let frame_idx = iter % flame_params.len();
            let mesh = self.flame.forward(&flame_params[frame_idx]);
            let normal_maps = cameras.iter()
                .map(|cam| self.flame.render_normal(&mesh, cam))
                .collect();

            // 3. Render current Gaussians
            let renders: Vec<RenderOutput> = cameras.iter()
                .map(|cam| self.rasterizer.forward(&model, cam))
                .collect()?;

            // 4. Generate Pseudo-GT via diffusion
            let pseudo_gt = self.diffusion.generate(
                &renders[0].color, &normal_maps, &cameras
            )?;

            // 5. Compute losses
            let losses = self.loss_computer.compute(
                &renders, &pseudo_gt.images, &model, &mesh
            );

            // 6. Backward pass through rasterizer
            let grads = self.rasterizer.backward(&loss_grads, &renders)?;

            // 7. Optimizer step
            self.optimizer.step(&mut model, &grads, iter);

            // 8. Density control
            if iter >= self.config.density_control_start
                && iter <= self.config.density_control_end
                && iter % self.config.density_control_interval == 0
            {
                self.density_ctrl.step(&mut model, &grad_accum);
            }

            // 9. Checkpoint
            if iter % self.config.checkpoint_interval == 0 {
                self.save_checkpoint(&model, iter)?;
            }
        }
        Ok(model)
    }
}
```

### 5.7 Checkpointing & Export

```rust
pub struct CheckpointManager;

impl CheckpointManager {
    /// Save full training state (model + optimizer + iteration)
    pub fn save(path: &Path, model: &GaussianModel, optimizer: &GaussianOptimizer, iter: u32) -> Result<()>;
    pub fn load(path: &Path) -> Result<(GaussianModel, GaussianOptimizer, u32)>;

    /// Export final model
    pub fn export_ply(path: &Path, model: &GaussianModel) -> Result<()>;
    pub fn export_safetensors(path: &Path, model: &GaussianModel) -> Result<()>;
}
```

Format: Custom `.safetensors` checkpoint with all Gaussian attributes + Adam state. PLY export for visualization in external tools.

### 5.8 Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Training divergence | 🔴 High | Gradient clipping, careful LR scheduling, validate with simple scenes first |
| Memory during long training | 🟡 Medium | GPU buffer reuse, periodic CPU offload of non-active state |
| LPIPS in Pure Rust | 🟡 Medium | Port VGG feature extractor to candle; fallback to L1+SSIM only |
| Diffusion inference bottleneck | 🟡 Medium | Cache Pseudo-GT, reuse across nearby iterations |
| Density control instability | 🟡 Medium | Conservative thresholds, validate Gaussian count growth |

---

## 6. oxigaf (Meta Crate) & oxigaf-cli (Binary)

### 6.0 oxigaf — Meta Crate (Library)

The `oxigaf` crate is a **thin library crate** that re-exports all sub-crates under a single unified namespace. This gives downstream consumers (the CLI, tests, third-party integrations) a single dependency instead of four.

**Purpose:**
- Library users `cargo add oxigaf` and get the full API
- The CLI binary (`oxigaf-cli`) depends only on `oxigaf`
- Feature flags on `oxigaf` propagate to the correct sub-crate

```rust
// crates/oxigaf/src/lib.rs

pub use oxigaf_flame as flame;
pub use oxigaf_diffusion as diffusion;
pub use oxigaf_render as render;
pub use oxigaf_trainer as trainer;

/// Convenience re-exports of commonly used types
pub mod prelude {
    pub use oxigaf_flame::{FlameModel, FlameParams, Mesh};
    pub use oxigaf_diffusion::{MultiViewDiffusionPipeline, DiffusionConfig};
    pub use oxigaf_render::{GaussianModel, Rasterizer, RasterConfig};
    pub use oxigaf_trainer::{Trainer, TrainingConfig};
}
```

**Cargo.toml for `oxigaf` (meta crate):**
```toml
[package]
name = "oxigaf"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "Pure Rust Gaussian Avatar Reconstruction — unified API"

[dependencies]
oxigaf-flame = { path = "../oxigaf-flame" }
oxigaf-diffusion = { path = "../oxigaf-diffusion" }
oxigaf-render = { path = "../oxigaf-render" }
oxigaf-trainer = { path = "../oxigaf-trainer" }

[features]
default = []
cuda = ["oxigaf-diffusion/cuda"]
metal = ["oxigaf-diffusion/metal"]
wgpu-normal = ["oxigaf-flame/wgpu-normal"]
full = ["cuda", "wgpu-normal"]
```

**Cargo.toml for `oxigaf-cli` (binary):**
```toml
[package]
name = "oxigaf-cli"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "CLI for OxiGAF — Gaussian Avatar Reconstruction"

[[bin]]
name = "oxigaf"
path = "src/main.rs"

[dependencies]
oxigaf = { path = "../oxigaf" }
clap = { workspace = true }
anyhow = { workspace = true }
serde = { workspace = true }
toml = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
indicatif = { workspace = true }
tokio = { workspace = true }
image = { workspace = true }
ffmpeg-next = { workspace = true, optional = true }
winit = { version = "0.30", optional = true }

[features]
default = ["video", "preview"]
video = ["dep:ffmpeg-next"]
preview = ["dep:winit"]
cuda = ["oxigaf/cuda"]
metal = ["oxigaf/metal"]
```

> **Key benefit**: Third-party Rust projects can `depend on oxigaf` (the library) without pulling in CLI dependencies (clap, indicatif, winit, ffmpeg-next). The binary name is still `oxigaf` via `[[bin]]`.

### 6.1 CLI Command Structure (clap)

```rust
#[derive(Parser)]
#[command(name = "oxigaf", about = "Pure Rust Gaussian Avatar Reconstruction")]
pub enum Cli {
    /// Reconstruct a 3D avatar from monocular video
    Reconstruct {
        #[arg(short, long)]
        input: PathBuf,                    // Video file or frame directory
        #[arg(short, long)]
        output: PathBuf,                   // Output directory
        #[arg(short, long, default_value = "config.toml")]
        config: PathBuf,                   // Training config
        #[arg(long)]
        flame_params: PathBuf,             // Pre-computed FLAME tracking .npz
        #[arg(long, default_value = "0")]
        device: usize,                     // GPU device index
        #[arg(long)]
        resume: Option<PathBuf>,           // Resume from checkpoint
    },

    /// Render an avatar from novel viewpoints
    Render {
        #[arg(short, long)]
        model: PathBuf,                    // .safetensors or .ply avatar
        #[arg(short, long)]
        output: PathBuf,                   // Output image directory
        #[arg(long, default_value = "512")]
        width: u32,
        #[arg(long, default_value = "512")]
        height: u32,
        #[arg(long)]
        cameras: Option<PathBuf>,          // Camera trajectory JSON
        #[arg(long)]
        flame_params: Option<PathBuf>,     // Animate with FLAME params
    },

    /// Export avatar to standard formats
    Export {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "ply")]
        format: ExportFormat,              // ply | safetensors
    },

    /// Real-time interactive preview
    Preview {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(long, default_value = "800")]
        width: u32,
        #[arg(long, default_value = "600")]
        height: u32,
    },

    /// Download required model weights
    Setup {
        #[arg(long, default_value = "~/.cache/oxigaf")]
        cache_dir: PathBuf,
    },
}
```

### 6.2 Configuration File Format

```toml
# oxigaf.toml

[model]
flame_model_path = "~/.cache/oxigaf/flame2023.npz"
diffusion_weights_dir = "~/.cache/oxigaf/weights/"

[device]
backend = "vulkan"       # vulkan | metal | dx12 | gl
gpu_index = 0

[training]
total_iterations = 15000
views_per_step = 4
image_size = 512
guidance_scale_start = 7.5
guidance_scale_end = 3.0
num_inference_steps = 50

[training.init]
num_rigid_gaussians = 50000
num_flexible_gaussians = 50000

[training.optimizer]
position_lr = 1.6e-4
position_lr_final = 1.6e-6
rotation_lr = 1e-3
scale_lr = 5e-3
opacity_lr = 5e-2
sh_lr = 2.5e-3

[training.density_control]
interval = 500
start_iteration = 1000
end_iteration = 12000
grad_threshold = 0.0002
min_opacity = 0.005

[training.loss]
lambda_photometric = 1.0
lambda_ssim = 0.2
lambda_lpips = 0.1
lambda_position_reg = 0.01
lambda_scale_reg = 0.01
lambda_normal = 0.05

[output]
checkpoint_interval = 1000
log_interval = 50
export_format = "ply"
```

### 6.3 End-to-End Pipeline Flow (`reconstruct`)

```
oxigaf reconstruct --input video.mp4 --flame-params tracking.npz --output ./avatar/

Step 1: Video Extraction
  ├─ ffmpeg-next: decode video → frame images in memory
  └─ Or: load pre-extracted frames from directory

Step 2: Load Pre-computed FLAME Tracking
  ├─ Load per-frame FlameParams from .npz
  └─ (Future: integrate FLAME fitting — out of scope for v1)

Step 3: Initialize
  ├─ Load FLAME model → compute rest-pose mesh
  ├─ Load diffusion model weights → MultiViewDiffusionPipeline
  ├─ Initialize wgpu device → Rasterizer
  └─ Sample Gaussians on mesh → GaussianModel

Step 4: Training Loop
  └─ (See Section 5.6 — Trainer.train())

Step 5: Export
  ├─ Save final GaussianModel as .ply and .safetensors
  ├─ Render showcase images from canonical viewpoints
  └─ Print summary statistics (Gaussian count, PSNR, training time)
```

### 6.4 Video Handling

```rust
pub struct VideoExtractor;

impl VideoExtractor {
    /// Extract frames from video file (requires 'video' feature / ffmpeg-next)
    pub fn extract_frames(video_path: &Path, config: &VideoConfig) -> Result<Vec<DynamicImage>>;
}

pub struct VideoConfig {
    pub max_frames: usize,       // default: 300
    pub target_fps: f32,         // default: 10.0 (subsample from source)
    pub resize: Option<(u32, u32)>,
}
```

**FLAME parameter input**: For v1.0, assume pre-computed FLAME tracking parameters (from external tools like DECA, MICA, or the original GAF Python code). Future versions may integrate a Rust FLAME fitter.

### 6.5 Real-Time Preview

```rust
pub struct PreviewWindow {
    window: winit::Window,
    surface: wgpu::Surface,
    rasterizer: Rasterizer,
    camera: ArcballCamera,      // interactive orbit camera
}

impl PreviewWindow {
    pub fn open(model: &GaussianModel) -> Result<()> {
        // winit event loop:
        //   - Mouse drag → orbit camera
        //   - Scroll → zoom
        //   - Arrow keys → translate
        //   - Space → toggle animation (if FLAME params loaded)
        //   - S → screenshot
        //   - Q / Esc → quit
    }
}
```

### 6.6 Asset Management

```rust
pub struct AssetManager {
    cache_dir: PathBuf,   // ~/.cache/oxigaf/
}

impl AssetManager {
    /// Download and cache all required model files
    pub fn setup(&self) -> Result<()> {
        // 1. FLAME model: download from MPI (requires agreement) or bundled converted .npz
        // 2. Diffusion weights: download from HuggingFace Hub (SafeTensors)
        // 3. CLIP weights
        // 4. Latent Upsampler weights
        // 5. LPIPS VGG weights (for perceptual loss)
    }

    pub fn get_path(&self, asset: Asset) -> PathBuf;
    pub fn verify_checksums(&self) -> Result<()>;
}
```

### 6.7 Logging & Progress

- **tracing**: Structured logging with levels (ERROR, WARN, INFO, DEBUG, TRACE)
- **indicatif**: Progress bars for training iterations, model loading, video extraction
- **Error handling**: `anyhow::Result` at CLI level, `thiserror` for typed errors in library crates

---

## 7. Development Roadmap & Milestones

### Phase 1: Foundation Building (v0.1.0) — Weeks 1–4

| Task | Module | Priority | Estimated Effort |
|------|--------|----------|------------------|
| Set up Cargo workspace with all 6 crates | workspace | P0 | 1 day |
| Implement FLAME model loading (.npz) | oxigaf-flame | P0 | 3 days |
| Implement LBS forward pass | oxigaf-flame | P0 | 5 days |
| CPU Normal Map renderer | oxigaf-flame | P0 | 3 days |
| Python conversion scripts (FLAME .pkl, weights) | scripts | P0 | 2 days |
| Candle VAE encode/decode smoke test | oxigaf-diffusion | P1 | 3 days |
| Candle CLIP image encoder | oxigaf-diffusion | P1 | 3 days |
| CI pipeline setup | .github | P1 | 1 day |
| **Milestone**: FLAME mesh renders correct Normal Maps | | | |

### Phase 2: Renderer Implementation (v0.2.0) — Weeks 5–10

| Task | Module | Priority | Estimated Effort |
|------|--------|----------|------------------|
| wgpu device setup, buffer management | oxigaf-render | P0 | 3 days |
| Preprocess shader (projection, cov2D) | oxigaf-render | P0 | 5 days |
| Radix sort (port from web-splat) | oxigaf-render | P0 | 5 days |
| Tile assignment + tile ranges shaders | oxigaf-render | P0 | 3 days |
| Forward rasterization shader | oxigaf-render | P0 | 5 days |
| SH evaluation shader | oxigaf-render | P0 | 2 days |
| MultiViewUNet implementation in candle | oxigaf-diffusion | P0 | 10 days |
| Cross-view attention blocks | oxigaf-diffusion | P0 | 5 days |
| DDIM scheduler (v-prediction) | oxigaf-diffusion | P1 | 3 days |
| Weight loading & validation | oxigaf-diffusion | P0 | 5 days |
| **Milestone**: Forward-only 3DGS renders a scene; diffusion model produces multi-view images | | | |

### Phase 3: Optimization Implementation (v0.3.0) — Weeks 11–16

| Task | Module | Priority | Estimated Effort |
|------|--------|----------|------------------|
| Backward rasterization shader | oxigaf-render | P0 | 10 days |
| Cov2D + preprocess backward shaders | oxigaf-render | P0 | 5 days |
| Gradient verification (finite differences) | oxigaf-render | P0 | 3 days |
| FLAME mesh binding (forward + backward) | oxigaf-render | P0 | 5 days |
| Adam optimizer (per-parameter) | oxigaf-trainer | P0 | 3 days |
| Loss functions (L1, SSIM) | oxigaf-trainer | P0 | 3 days |
| LPIPS port to candle | oxigaf-trainer | P1 | 5 days |
| Single-image Gaussian fitting test | oxigaf-trainer | P0 | 3 days |
| Latent Upsampler | oxigaf-diffusion | P1 | 5 days |
| **Milestone**: Fit Gaussians to a single image via gradient descent | | | |

### Phase 4: GAF Integration (v1.0.0) — Weeks 17–22

| Task | Module | Priority | Estimated Effort |
|------|--------|----------|------------------|
| Full training loop (iterative denoising) | oxigaf-trainer | P0 | 5 days |
| Adaptive density control | oxigaf-trainer | P0 | 5 days |
| Gaussian initialization on FLAME mesh | oxigaf-trainer | P0 | 3 days |
| oxigaf meta crate (re-exports + features) | oxigaf | P0 | 1 day |
| Video frame extraction | oxigaf-cli | P0 | 2 days |
| CLI commands (reconstruct, render, export) | oxigaf-cli | P0 | 5 days |
| TOML configuration system | oxigaf-cli | P0 | 2 days |
| Checkpointing (save/load) | oxigaf-trainer | P0 | 2 days |
| PLY export | oxigaf-cli | P0 | 1 day |
| Real-time preview (winit) | oxigaf-cli | P2 | 5 days |
| Asset download manager | oxigaf-cli | P1 | 3 days |
| End-to-end testing on real videos | all | P0 | 5 days |
| **Milestone**: Full pipeline from monocular video → animatable 3D avatar | | | |

### Total Estimated Timeline: ~22 weeks (5.5 months)

---

## 8. Risk Registry

| # | Risk | Severity | Probability | Impact | Mitigation |
|---|------|----------|-------------|--------|------------|
| R1 | Diffusion weight mapping errors causing incorrect generation | 🔴 Critical | High | Blocks pipeline | Layer-by-layer output comparison with Python reference |
| R2 | Backward rasterization shader bugs | 🔴 Critical | High | Blocks optimization | Finite-difference gradient checks; start with simple scenes |
| R3 | wgpu compute shader limitations on certain GPUs | 🔴 High | Medium | Platform incompatibility | Test on Vulkan/Metal/DX12; provide software fallback |
| R4 | Training divergence / poor convergence | 🟡 Medium | Medium | Quality issues | Match hyperparameters exactly to paper; ablation tests |
| R5 | candle missing required ops | 🟡 Medium | Medium | Blocks diffusion module | Audit ops early; contribute upstream or implement custom ops |
| R6 | FLAME model licensing restrictions | 🟡 Medium | Low | Distribution issues | Use FLAME 2023 Open (CC-BY-4.0); document license requirements |
| R7 | GPU memory exhaustion during training | 🟡 Medium | Medium | Limits resolution/quality | Attention slicing, gradient checkpointing, reduce batch size |
| R8 | Performance gap vs CUDA reference | 🟡 Medium | High | Slower training | Accept 2-3x overhead for wgpu; optimize hot paths |
| R9 | ffmpeg-next breaking Pure Rust goal | 🟢 Low | Low | Purist concern | Feature-gate; support pre-extracted frames as primary input |
| R10 | Cross-view attention numerical precision in fp16 | 🟡 Medium | Medium | Image artifacts | Selective fp32 for attention computation |
