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

### Basic Gaussian Rendering

```rust
use oxigaf_render::{
    Rasterizer,
    RasterConfig,
    RenderCamera,
    gaussian::{GaussianAttributes, GaussianModel}
};

fn main() -> Result<(), oxigaf_render::RenderError> {
    // Create a simple Gaussian model
    let gaussians = vec![
        GaussianAttributes {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],  // Identity quaternion
            log_scale: [-4.0, -4.0, -4.0],   // exp(-4) ≈ 0.018 scale
            opacity_logit: 2.0,               // sigmoid(2.0) ≈ 0.88 opacity
        },
    ];

    // SH coefficients for color (DC component only, degree 0)
    // RGB = [1.0, 0.5, 0.5] gives a reddish color
    let sh_coeffs = vec![1.0, 0.5, 0.5];

    let model = GaussianModel::new(gaussians, sh_coeffs, 0)?;

    // Initialize GPU rasterizer
    let config = RasterConfig {
        width: 512,
        height: 512,
        tile_size: 16,
        ..Default::default()
    };

    let mut rasterizer = Rasterizer::new(&config)?;

    // Set up camera
    let camera = RenderCamera::look_at(
        [0.0, 0.0, 2.0],    // eye position (2 units back)
        [0.0, 0.0, 0.0],    // look at origin
        [0.0, 1.0, 0.0],    // up vector
        std::f32::consts::FRAC_PI_4,  // 45° field of view
        1.0,                // aspect ratio (width/height)
    );

    // Render frame
    let output = rasterizer.forward(&model, &camera)?;

    // Save rendered image
    output.color.save("output.png").map_err(|e| {
        oxigaf_render::RenderError::Io(
            format!("Failed to save image: {}", e)
        )
    })?;

    println!("Rendered {} Gaussians", model.num_gaussians());

    Ok(())
}
```

### Differentiable Rendering with Gradients

```rust
use oxigaf_render::{Rasterizer, RasterConfig, RenderCamera, gaussian::GaussianModel};
use image::Rgba;

fn main() -> Result<(), oxigaf_render::RenderError> {
    let mut rasterizer = Rasterizer::new(&RasterConfig::default())?;
    let mut model = create_gaussian_model()?; // Your model
    let camera = create_camera(); // Your camera

    // Forward pass
    let output = rasterizer.forward(&model, &camera)?;

    // Compute target image difference (e.g., for training)
    let target_image = load_target_image()?;
    let grad_output = compute_image_gradients(&output.color, &target_image)?;

    // Backward pass to get parameter gradients
    let gradients = rasterizer.backward(&grad_output)?;

    // Access gradients for optimization
    println!("Position gradients: {} values", gradients.positions.len());
    println!("Rotation gradients: {} values", gradients.rotations.len());
    println!("Scale gradients: {} values", gradients.scales.len());
    println!("Opacity gradients: {} values", gradients.opacities.len());
    println!("SH gradients: {} values", gradients.sh_coeffs.len());

    // Apply gradients with optimizer (e.g., Adam)
    apply_gradients(&mut model, &gradients, learning_rate)?;

    Ok(())
}

// Helper functions (simplified for example)
fn create_gaussian_model() -> Result<GaussianModel, oxigaf_render::RenderError> {
    // Implementation details...
    unimplemented!()
}

fn create_camera() -> RenderCamera {
    RenderCamera::look_at(
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        std::f32::consts::FRAC_PI_4,
        1.0,
    )
}

fn load_target_image() -> Result<image::RgbaImage, oxigaf_render::RenderError> {
    // Load your ground truth image
    unimplemented!()
}

fn compute_image_gradients(
    rendered: &image::RgbaImage,
    target: &image::RgbaImage
) -> Result<Vec<f32>, oxigaf_render::RenderError> {
    // Compute L1 or L2 loss gradients
    unimplemented!()
}

fn apply_gradients(
    model: &mut GaussianModel,
    gradients: &oxigaf_render::GaussianGradients,
    learning_rate: f32
) -> Result<(), oxigaf_render::RenderError> {
    // Apply gradients with your optimizer
    unimplemented!()
}
```

### Custom Camera Trajectories

```rust
use oxigaf_render::{Rasterizer, RasterConfig, RenderCamera, gaussian::GaussianModel};
use std::f32::consts::PI;

fn main() -> Result<(), oxigaf_render::RenderError> {
    let mut rasterizer = Rasterizer::new(&RasterConfig::default())?;
    let model = load_avatar_model()?;

    // Generate circular camera trajectory
    let num_frames = 360;
    let radius = 2.0;

    for frame in 0..num_frames {
        let angle = 2.0 * PI * (frame as f32) / (num_frames as f32);

        // Camera orbits around origin
        let eye_x = radius * angle.cos();
        let eye_z = radius * angle.sin();

        let camera = RenderCamera::look_at(
            [eye_x, 0.0, eye_z],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            std::f32::consts::FRAC_PI_4,
            1.0,
        );

        let output = rasterizer.forward(&model, &camera)?;

        output.color.save(format!("frame_{:04}.png", frame)).map_err(|e| {
            oxigaf_render::RenderError::Io(
                format!("Failed to save frame {}: {}", frame, e)
            )
        })?;
    }

    println!("Rendered {} frames", num_frames);

    Ok(())
}

fn load_avatar_model() -> Result<GaussianModel, oxigaf_render::RenderError> {
    // Load your trained avatar model
    unimplemented!()
}
```

### High-Quality Rendering Configuration

```rust
use oxigaf_render::{Rasterizer, RasterConfig};

fn main() -> Result<(), oxigaf_render::RenderError> {
    // High-quality rendering configuration
    let config = RasterConfig {
        width: 1920,           // Full HD width
        height: 1080,          // Full HD height
        tile_size: 16,         // Tile size for parallel rendering
        near: 0.01,            // Near clipping plane
        far: 100.0,            // Far clipping plane
        max_sh_degree: 3,      // Maximum spherical harmonics degree
        antialiasing: true,    // Enable antialiasing
        depth_test: true,      // Enable depth testing
    };

    let mut rasterizer = Rasterizer::new(&config)?;

    println!("Initialized high-quality rasterizer:");
    println!("  Resolution: {}×{}", config.width, config.height);
    println!("  Tile size: {}×{}", config.tile_size, config.tile_size);
    println!("  Max SH degree: {}", config.max_sh_degree);

    Ok(())
}
```

### Buffer Pool for Memory Efficiency

```rust
use oxigaf_render::{BufferPool, RasterConfig};

fn main() -> Result<(), oxigaf_render::RenderError> {
    // Create a buffer pool for efficient GPU memory reuse
    let pool = BufferPool::new(10 * 1024 * 1024); // 10 MB initial capacity

    // Get pool statistics
    let stats = pool.stats();
    println!("Buffer pool statistics:");
    println!("  Total allocations: {}", stats.total_allocations);
    println!("  Cache hits: {}", stats.cache_hits);
    println!("  Cache misses: {}", stats.cache_misses);
    println!("  Active buffers: {}", stats.active_buffers);
    println!("  Total memory: {} bytes", stats.total_memory);

    // Buffers are automatically returned to the pool when dropped

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
