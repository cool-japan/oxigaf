# oxigaf-render

Differentiable 3D Gaussian Splatting rasterizer using wgpu compute shaders.

## Overview

This crate implements a GPU-accelerated, differentiable 3D Gaussian Splatting (3DGS) rasterizer for real-time rendering and gradient-based optimization:

- **Forward pass**: Project 3D Gaussians → Sort by depth → Tile-based alpha blending
- **Backward pass**: Compute gradients for positions, rotations, scales, opacities, and colors
- **FLAME binding**: Anchor Gaussians to a parametric head model for expression control
- **wgpu backend**: Cross-platform GPU compute (Vulkan, Metal, DX12, WebGPU)
- **Tile-based rendering**: Efficient parallel rasterization with minimal memory overhead

**v0.1.2 — what's included:**
- **Two gradient-correctness fixes in the backward shaders** — a wrong-Gaussian gradient accumulation bug in the backward tile kernel, and a missing position gradient through view-dependent SH color for `sh_degree >= 1` models. Both affected every 0.1.1 training run — see "Upgrading from 0.1.1" below before you retrain anything
- **glTF export** — new `gltf` module (`write_gltf`/`GltfError`), a single spec-conformant glTF 2.0 writer consolidating what were three independently-written, mutually-incompatible glTF emitters in the workspace
- **GPU timestamp profiler** — `profiler::GpuTimestampProfiler` + `Rasterizer::enable_gpu_timestamps()`, pass-level GPU-side timing (via `wgpu::Features::TIMESTAMP_QUERY`) alongside the existing CPU-side `PassProfiler`
- `Rasterizer::from_device` now validates a caller-supplied device's limits up front (`rasterizer::rasterizer_device_limits`) instead of failing later as an opaque pipeline-validation error
- 2,903 tests, all passing (2,887 run by default + 16 GPU-hardware `#[ignore]`d tests verified separately)

## Upgrading from 0.1.1

- **Retrain any `sh_degree >= 1` model.** `preprocess_bwd.wgsl` only differentiated the projection path from Gaussian position to screen space, omitting the position gradient through view-dependent SH color — so every 0.1.1 model with `sh_degree >= 1` trained against a systematically incomplete position gradient. There is no way to recover the missing signal from an already-trained model; retraining is the only remedy.
- **The backward tile kernel could attribute a tile's gradient sum to the wrong Gaussian.** `rasterize_bwd.wgsl`'s reverse-traversal loop bound was read per-pixel, so threads in the same 16×16 workgroup ran a different number of loop iterations around `workgroupBarrier()` calls — non-uniform control flow, which WGSL requires to be uniform. Fixed by computing a single workgroup-uniform loop bound. This also affected every 0.1.1 training run.
- **Re-export any `.ply` with `sh_degree >= 1` written by `GaussianModel::save_ply` before this release.** The `f_rest_*` property order changed to channel-major (matching the reference 3DGS Python convention); older files load with correctly-valued but permuted higher-order SH coefficients.
- **API renames to watch for**: `hdr_tone_mapping::lottes_approx` → `generalized_reinhard` (a real, non-approximating `lottes()` was added alongside it); `MbStats.mean_samples_used` → `estimated_sample_utilization`; `RadixSorter::sort` no longer takes `device`/`keys`/`values` (that setup moved to a new `prepare()` step); `Rasterizer::from_device` now errors on `tile_size != 16` instead of silently accepting it.

See `CHANGELOG.md` for the complete list of changes, including two more minor backward-pass fixes (a missing background-alpha gradient contribution, and NaN gradients for culled Gaussians).

## Installation

```toml
[dependencies]
oxigaf-render = "0.1"
```

## Features

| Feature | Description |
|---------|-------------|
| `default` | Minimal configuration (production-ready) |
| `gpu_debug` | GPU validation layers with detailed error messages (10-100× slower) |

### Feature Details

- **`gpu_debug`**: Enables comprehensive GPU debugging
  - Vulkan validation layers (Linux/Windows)
  - Metal API validation (macOS)
  - DirectX debug layer (Windows)
  - Enhanced error messages and warnings
  - Performance profiling with detailed traces
  - **Warning**: Adds significant runtime overhead (10-100× slower)
  - Recommended for development/debugging only

### Example Usage

```toml
# Production use (fast, minimal validation)
oxigaf-render = "0.1"

# Development/debugging (slow, extensive validation)
oxigaf-render = { version = "0.1", features = ["gpu_debug"] }
```

## Usage

All the examples below use `pollster` to block on the async GPU setup calls
from a plain `fn main`; add `pollster = "1"` to your own `[dependencies]`
to run them as-is (any other async executor works too).

### Basic Gaussian Rendering

```rust
use oxigaf_render::{
    keyframe_to_render_camera, CameraKeyframe, GaussianAttributes, GaussianModel,
    RasterConfig, Rasterizer,
};

// `Rasterizer::new` is async because wgpu device creation is async.
fn main() -> Result<(), oxigaf_render::RenderError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), oxigaf_render::RenderError> {
    // Create a simple Gaussian model (SH degree 0, one Gaussian).
    let gaussians = vec![GaussianAttributes {
        position: [0.0, 0.0, 0.0],
        _pad0: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion (x, y, z, w)
        scale: [-4.0, -4.0, -4.0],      // log-scale: exp(-4) ≈ 0.018
        opacity: 2.0,                   // inverse-sigmoid: sigmoid(2.0) ≈ 0.88
    }];

    // SH coefficients for color (DC component only, degree 0).
    // The DC term is scaled by the Y_0^0 basis constant (≈0.2821) when
    // evaluated, so pre-dividing by it gives ≈ RGB [1.0, 0.5, 0.5] on screen.
    let sh_coeffs = vec![3.545, 1.772, 1.772];

    let model = GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree: 0,
        // Only meaningful for FLAME-bound avatars; unused here.
        face_indices: vec![0],
        barycentric: vec![[1.0, 0.0, 0.0]],
        local_offsets: vec![[0.0, 0.0, 0.0]],
        is_rigid: vec![true],
    };

    // Initialize the GPU rasterizer.
    let config = RasterConfig::new()
        .with_resolution(512, 512)
        .with_sh_degree(0);

    let mut rasterizer = Rasterizer::new(config.clone()).await?;

    // Set up a look-at camera.
    let camera = keyframe_to_render_camera(
        &CameraKeyframe::look_from_to(0.0, [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]),
        config.image_width as usize,
        config.image_height as usize,
    );

    // Upload Gaussians once, then render.
    rasterizer.upload_gaussians(&model);
    let output = rasterizer.forward(&model, &camera)?;

    // Convert to an `image::RgbaImage` and save it.
    let image = rasterizer.download_image(&output);
    image
        .save("output.png")
        .map_err(|e| oxigaf_render::RenderError::ImageSaveFailed(e.to_string()))?;

    println!("Rendered {} Gaussians", model.len());

    Ok(())
}
```

### Differentiable Rendering with Gradients

```rust
use oxigaf_render::{
    keyframe_to_render_camera, CameraKeyframe, GaussianGradients, GaussianModel, RasterConfig,
    Rasterizer, RenderCamera,
};

fn main() -> Result<(), oxigaf_render::RenderError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), oxigaf_render::RenderError> {
    let config = RasterConfig::default();
    let mut rasterizer = Rasterizer::new(config.clone()).await?;
    let mut model = create_gaussian_model(); // Your model
    let camera = create_camera(config.image_width, config.image_height);

    // Forward pass.
    rasterizer.upload_gaussians(&model);
    let output = rasterizer.forward(&model, &camera)?;

    // Compute the per-pixel loss gradient against a target (e.g. L1).
    let target = load_target_image(output.color_data.len());
    let grad_image = l1_gradient(&output.color_data, &target);

    // Backward pass: chains the image-space gradient back to per-Gaussian gradients.
    let gradients = rasterizer.backward(&model, &grad_image)?;

    // Access gradients for optimization.
    println!("Position gradients: {} values", gradients.grad_positions.len());
    println!("Rotation gradients: {} values", gradients.grad_rotations.len());
    println!("Scale gradients: {} values", gradients.grad_scales.len());
    println!("Opacity gradients: {} values", gradients.grad_opacities.len());
    println!("SH gradients: {} values", gradients.grad_sh_coeffs.len());

    // Apply gradients with a simple step (swap in Adam, etc. for real training).
    apply_gradients(&mut model, &gradients, 0.001);

    Ok(())
}

// --- Helper functions (fill in for your training loop) ---

fn create_gaussian_model() -> GaussianModel {
    // e.g. GaussianModel::load_ply(Path::new("avatar.ply"))
    //      or GaussianModel::load_safetensors(...)
    unimplemented!("build or load a GaussianModel")
}

fn create_camera(width: u32, height: u32) -> RenderCamera {
    keyframe_to_render_camera(
        &CameraKeyframe::look_from_to(0.0, [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]),
        width as usize,
        height as usize,
    )
}

fn load_target_image(len: usize) -> Vec<f32> {
    // Load your ground-truth image as RGBA f32 pixels matching
    // `output.color_data`'s layout (H * W * 4, row-major).
    vec![0.0; len]
}

/// L1 loss gradient: sign(rendered - target).
fn l1_gradient(rendered: &[f32], target: &[f32]) -> Vec<f32> {
    rendered
        .iter()
        .zip(target)
        .map(|(r, t)| (r - t).signum())
        .collect()
}

fn apply_gradients(model: &mut GaussianModel, gradients: &GaussianGradients, learning_rate: f32) {
    for (g, grad) in model
        .gaussians
        .iter_mut()
        .zip(gradients.grad_positions.iter())
    {
        g.position[0] -= learning_rate * grad[0];
        g.position[1] -= learning_rate * grad[1];
        g.position[2] -= learning_rate * grad[2];
    }
}
```

### Custom Camera Trajectories

```rust
use oxigaf_render::{turntable_path, GaussianModel, RasterConfig, Rasterizer};
use std::f32::consts::FRAC_PI_4;

fn main() -> Result<(), oxigaf_render::RenderError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), oxigaf_render::RenderError> {
    let config = RasterConfig::new().with_resolution(512, 512);
    let mut rasterizer = Rasterizer::new(config.clone()).await?;
    let model = load_avatar_model();

    // Generate a 360-frame turntable orbit around the origin.
    let num_frames: usize = 360;
    let path = turntable_path(
        [0.0, 0.0, 0.0], // center
        2.0,             // radius
        0.0,             // elevation
        num_frames,      // keyframes
        FRAC_PI_4,       // vertical FOV
    );
    let cameras = path.to_render_cameras(
        num_frames,
        config.image_width as usize,
        config.image_height as usize,
    );

    rasterizer.upload_gaussians(&model);

    for (frame, camera) in cameras.iter().enumerate() {
        let output = rasterizer.forward(&model, camera)?;
        let image = rasterizer.download_image(&output);
        image.save(format!("frame_{frame:04}.png")).map_err(|e| {
            oxigaf_render::RenderError::ImageSaveFailed(format!(
                "Failed to save frame {frame}: {e}"
            ))
        })?;
    }

    println!("Rendered {} frames", cameras.len());

    Ok(())
}

fn load_avatar_model() -> GaussianModel {
    // e.g. GaussianModel::load_ply(Path::new("avatar.ply")).unwrap_or_else(...)
    unimplemented!("load your trained avatar model")
}
```

### High-Quality Rendering Configuration

```rust
use oxigaf_render::{RasterConfig, Rasterizer};

fn main() -> Result<(), oxigaf_render::RenderError> {
    // High-quality rendering configuration.
    let config = RasterConfig::new()
        .with_resolution(1920, 1080) // Full HD
        .with_sh_degree(3)           // Maximum spherical harmonics degree
        .with_depth_output(true);    // Enable depth buffer output

    println!("Initialized high-quality rasterizer config:");
    println!(
        "  Resolution: {}x{}",
        config.image_width, config.image_height
    );
    println!("  Tile size: {}x{}", config.tile_size, config.tile_size);
    println!("  Max SH degree: {}", config.sh_degree);

    // `Rasterizer::new` is async because wgpu device creation is async.
    let _rasterizer = pollster::block_on(Rasterizer::new(config))?;

    Ok(())
}
```

### Buffer Pool for Memory Efficiency

```rust
use oxigaf_render::BufferPool;

fn main() {
    // Create a buffer pool for efficient GPU memory reuse.
    let pool = BufferPool::new(10 * 1024 * 1024); // 10 MB budget

    // Get pool statistics.
    let stats = pool.stats();
    println!("Buffer pool statistics:");
    println!("  Total allocations: {}", stats.total_allocations);
    println!("  Total acquisitions: {}", stats.total_acquisitions);
    println!("  Hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("  Available buffers: {}", stats.available_count);
    println!("  In-use buffers: {}", stats.in_use_count);
    println!("  Total allocated: {} bytes", stats.total_allocated_bytes);

    // Buffers are automatically returned to the pool when dropped.
}
```

### Exporting to glTF

```rust
use oxigaf_render::gltf::write_gltf;
use oxigaf_render::{GaussianAttributes, GaussianModel};
use std::path::Path;

fn main() -> Result<(), oxigaf_render::gltf::GltfError> {
    let model = GaussianModel {
        gaussians: vec![GaussianAttributes {
            position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-4.0, -4.0, -4.0],
            opacity: 2.0,
        }],
        sh_coeffs: vec![3.545, 1.772, 1.772],
        sh_degree: 0,
        face_indices: vec![0],
        barycentric: vec![[1.0, 0.0, 0.0]],
        local_offsets: vec![[0.0, 0.0, 0.0]],
        is_rigid: vec![true],
    };

    // Spec-conformant glTF 2.0: `POSITION` is a real mesh-primitive
    // attribute (with the mandatory min/max glTF requires on it).
    // Rotation/scale/opacity/SH have no standard glTF per-vertex semantic,
    // so each gets its own accessor + buffer view too, referenced by index
    // from the `OXIGAF_gaussian_splat` node extension instead.
    write_gltf(&model, Path::new("avatar.gltf"))?;
    println!("Wrote {} Gaussians to glTF", model.len());

    Ok(())
}
```

### Building a Rasterizer from a Custom Device

`Rasterizer::new` creates its own wgpu instance/adapter/device. To reuse an
existing `wgpu::Device` (e.g. one shared with a windowing/UI layer), raise
its limits with `rasterizer_device_limits` before requesting it, then call
`Rasterizer::from_device`. This needs `wgpu` directly in your own
`[dependencies]` — add `wgpu = "30"`, matching the version this crate
depends on, since `from_device` takes a `wgpu::Device` and a version
mismatch there is a type error, not a warning:

```rust
use oxigaf_render::rasterizer::rasterizer_device_limits;
use oxigaf_render::{RasterConfig, Rasterizer};

fn main() -> Result<(), oxigaf_render::RenderError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), oxigaf_render::RenderError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| oxigaf_render::RenderError::GpuInit(format!("no adapter: {e}")))?;

    // `Rasterizer::from_device` validates the device's limits and rejects
    // `tile_size != RASTERIZE_TILE_SIZE`; a hand-built device must raise the
    // storage-buffer and workgroup-memory ceilings itself first.
    let required_limits = rasterizer_device_limits(wgpu::Limits::default());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("oxigaf_render_custom"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| oxigaf_render::RenderError::GpuInit(e.to_string()))?;

    let config = RasterConfig::new().with_resolution(512, 512);
    let mut rasterizer = Rasterizer::from_device(device, queue, config)?;

    // Succeeds only when the device was created with
    // `wgpu::Features::TIMESTAMP_QUERY` (as above, this one wasn't, so this
    // returns `Err` instead of panicking); `Rasterizer::new` requests the
    // feature automatically whenever the adapter supports it.
    match rasterizer.enable_gpu_timestamps() {
        Ok(()) => println!("GPU timestamp profiling enabled"),
        Err(e) => println!("GPU timestamps unavailable on this device: {e}"),
    }

    Ok(())
}
```

## Architecture

### 3D Gaussian Splatting

Each Gaussian primitive is defined by:

- **Position** (3 floats): 3D center point (x, y, z)
- **Rotation** (4 floats): Quaternion (x, y, z, w) defining orientation
- **Scale** (3 floats): Log-scale values (sx, sy, sz) — exponentiated before use
- **Opacity** (1 float): Inverse-sigmoid opacity — passed through sigmoid(x)
- **SH Coefficients** (N floats): Spherical harmonics for view-dependent color
  - Degree 0: 3 coeffs (1 band × 3 RGB)
  - Degree 1: 12 coeffs (4 bands × 3 RGB)
  - Degree 2: 27 coeffs (9 bands × 3 RGB)
  - Degree 3: 48 coeffs (16 bands × 3 RGB)

### Rendering Pipeline

1. **Preprocess**: Project Gaussians to screen space, compute covariance matrices
2. **Sort**: Sort Gaussians by depth (front-to-back for alpha blending)
3. **Rasterize**: Tile-based parallel alpha blending
4. **Output**: RGB image + depth map + alpha mask

### Shader Specialization

The rasterizer includes specialized shaders for different SH degrees:

- `preprocess_sh0.wgsl` — DC component only (fastest)
- `preprocess_sh1.wgsl` — Degree 1 (4 bands)
- `preprocess_sh2.wgsl` — Degree 2 (9 bands)
- `preprocess_sh3.wgsl` — Degree 3 (16 bands)

Specialization provides **10× speedup** over generic shader.

## Performance

Rendering performance on various hardware (512×512 resolution):

| Hardware | Gaussians | FPS | Memory |
|----------|-----------|-----|--------|
| NVIDIA RTX 4090 | 100K | 240+ | 1.2 GB |
| NVIDIA RTX 3080 | 100K | 120+ | 1.5 GB |
| Apple M2 Max | 100K | 60+ | 1.8 GB |
| Apple M1 | 50K | 60+ | 1.2 GB |

## Differentiability

**Backward pass** — all parameters have verified gradients (two gradient-correctness bugs were fixed in 0.1.2 — see "Upgrading from 0.1.1" above):

- ∂L/∂color, ∂L/∂alpha, ∂L/∂conic, ∂L/∂mean2D (rasterize_bwd), including the background's contribution to ∂L/∂alpha through final transmittance
- ∂L/∂SH, ∂L/∂scale, ∂L/∂rotation, ∂L/∂mean3D (preprocess_bwd), including the position gradient through view-dependent SH color for `sh_degree >= 1`
- ∂L/∂local_offset for FLAME mesh-bound Gaussians (binding backward pass)

**Gradient verification** — 78 tests (see `tests/gpu_gradient_verify.rs` + `tests/gpu_gradient_verify/sh.rs` and `tests/gradient_verification/`): finite-difference parameter checks plus the harness's own NaN/empty-input guard tests, checked with a median-error metric robust to per-pixel outliers: ≤5e-2 median relative error for most parameters, relaxed to ≤2.5e-1 for position (the tiled rasterizer's forward pass has tile-boundary discontinuities that finite differences can't model, so position gradients need a wider tolerance). These tests skip themselves at runtime (not via `#[ignore]`) when no GPU adapter is available, so a headless run without a GPU still reports them as passing without actually exercising the shaders — a real GPU adapter was present and exercised when the numbers below were last measured.

## Statistics

- **Tests**: 2,903 `#[test]`-attributed tests in source (`src/` + `tests/`), all passing as of this release:
  - 2,887 run by default: `cargo nextest run -p oxigaf-render --all-features`
  - 16 more are `#[ignore]`d because they need real GPU hardware (4 `deform`, 5 `multi_view`, 4 `cpu_gpu_compare`, 1 `pipeline`, 1 `sort`, plus 1 slow 100-Gaussian position-gradient check); run them with `--run-ignored ignored-only` — all 16 pass when a GPU adapter is available
  - 23 doc-tests passing, 3 `#[ignore]`d
  - 78 of the above are in the gradient-verification suite (`gpu_gradient_verify.rs` + `tests/gpu_gradient_verify/sh.rs` and `tests/gradient_verification/`) — see "Differentiability" above
- **Shaders** (17 `.wgsl` files): `preprocess` and its degree-specialized variants `preprocess_{sh0,sh1,sh2,sh3}`, `preprocess_bwd`, `rasterize_{fwd,bwd}`, `deform_gaussians`, `flame_binding_bwd`, `tile_assign`, `tile_ranges`, `prefix_sum{,_add}`, `radix_{histogram,scatter}`, `atomic_to_f32`. `cov2d_bwd` documents the 2D-covariance backward math but is not itself compiled/dispatched — the live GPU path is inline in `preprocess_bwd`.

Run benchmarks with:

```bash
cargo bench -p oxigaf-render
```

## GPU Backend Selection

wgpu automatically selects the best available backend:

| Platform | Primary Backend | Fallback |
|----------|----------------|----------|
| Linux | Vulkan | — |
| Windows | Vulkan | DX12 |
| macOS | Metal | — |
| Web | WebGPU | WebGL 2 |

## Documentation

- [API Documentation](https://docs.rs/oxigaf-render)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Repository](https://github.com/cool-japan/oxigaf)
- [Crate](https://crates.io/crates/oxigaf-render)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE))
