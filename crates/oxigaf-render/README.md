# oxigaf-render

Differentiable 3D Gaussian Splatting rasterizer using wgpu compute shaders.

## Overview

This crate implements a GPU-accelerated, differentiable 3D Gaussian Splatting (3DGS) rasterizer for real-time rendering and gradient-based optimization:

- **Forward pass**: Project 3D Gaussians → Sort by depth → Tile-based alpha blending
- **Backward pass**: Compute gradients for positions, rotations, scales, opacities, and colors
- **FLAME binding**: Anchor Gaussians to a parametric head model for expression control
- **wgpu backend**: Cross-platform GPU compute (Vulkan, Metal, DX12, WebGPU)
- **Tile-based rendering**: Efficient parallel rasterization with minimal memory overhead

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

**Backward pass** — all parameters have verified gradients:

- ∂L/∂color, ∂L/∂alpha, ∂L/∂conic, ∂L/∂mean2D (rasterize_bwd)
- ∂L/∂SH, ∂L/∂scale, ∂L/∂rotation, ∂L/∂mean3D (preprocess_bwd)
- ∂L/∂local_offset for FLAME mesh-bound Gaussians (binding backward pass)

**Gradient verification** — 35 finite-difference tests with <1e-3 relative error across all parameters (see `tests/gpu_gradient_verify.rs`).

## Statistics

- **Tests**: 140 (all passing)
  - 18 unit tests (in `src/`), 122 integration tests
  - 35 gradient verification tests (`gpu_gradient_verify.rs`)
- **Shaders**: `preprocess_{sh0,sh1,sh2,sh3}`, `rasterize_fwd`, `rasterize_bwd`, `preprocess_bwd`, radix sort suite, `tile_ranges`, `binding`

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
