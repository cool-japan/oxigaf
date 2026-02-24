# oxigaf

Pure Rust Gaussian Avatar Reconstruction — unified API for the OxiGAF ecosystem.

OxiGAF implements [GAF (Gaussian Avatars Reconstructed from Multi-view Images using Feed-forward)](https://www.microsoft.com/en-us/research/project/gaf/) in pure Rust with no Python or C/C++ dependencies in the default build.

## Overview

OxiGAF provides a complete pipeline for reconstructing photorealistic 3D head avatars from monocular images:

- **FLAME parametric head model** — 5023 vertices, Linear Blend Skinning (LBS)
- **Multi-view diffusion** — novel view synthesis via CLIP + U-Net + VAE
- **Differentiable 3DGS rasterizer** — wgpu compute shaders with FLAME binding
- **Full training pipeline** — Adam optimizer, density control, checkpointing

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxigaf = "0.1"
```

## Features

### GPU Backend Features

| Feature | Description |
|---------|-------------|
| `default` | Pure CPU inference (no CUDA/Metal dependencies) |
| `cuda` | NVIDIA GPU acceleration via candle CUDA backend (requires CUDA toolkit) |
| `metal` | Apple Silicon acceleration via Metal (automatic on macOS with M1/M2/M3) |

### Performance Optimization Features

| Feature | Description | Speedup |
|---------|-------------|---------|
| `simd` | SIMD-accelerated FLAME operations (requires nightly Rust) | 3-4× faster |
| `parallel` | Parallel batch processing with rayon | Near-linear with cores |
| `flash_attention` | Memory-efficient O(N) attention (enabled by default) | 2-4× less memory |
| `mixed_precision` | FP16/BF16 inference (planned, not yet implemented) | 2× faster on GPUs |

### Debug Features

| Feature | Description |
|---------|-------------|
| `gpu_debug` | Enable GPU validation layers (adds 10-100× overhead) |

### Convenience Feature Bundles

| Feature | Enables |
|---------|---------|
| `full_performance` | `simd`, `parallel`, `flash_attention` |
| `all_features` | All available features (except GPU backends) |

### Examples

```toml
# CPU-only with all optimizations (requires nightly for SIMD)
oxigaf = { version = "0.1", features = ["full_performance"] }

# Apple Silicon with Metal acceleration
oxigaf = { version = "0.1", features = ["metal", "parallel", "flash_attention"] }

# NVIDIA GPU with all optimizations
oxigaf = { version = "0.1", features = ["cuda", "full_performance"] }
```

## Usage

### Load FLAME and Generate a Mesh

```rust
use oxigaf::prelude::*;

fn main() -> oxigaf::Result<()> {
    // Load FLAME model from directory containing .npy files
    let model = FlameModel::load("path/to/flame/model")?;

    // Create neutral parameters (zero shape, expression, pose)
    let params = FlameParams::neutral();

    // Run forward pass to get posed mesh
    let mesh = model.forward(&params);

    println!("Generated mesh with {} vertices", mesh.vertices.len());
    Ok(())
}
```

### Render 3D Gaussians

```rust
use oxigaf::prelude::*;
use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

fn main() -> oxigaf::Result<()> {
    // Create a simple Gaussian model
    let gaussians = vec![
        GaussianAttributes {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],  // Identity quaternion
            log_scale: [-4.0, -4.0, -4.0],   // Small Gaussian
            opacity_logit: 2.0,               // High opacity
        },
    ];

    let sh_coeffs = vec![1.0, 0.5, 0.5];  // Reddish color (SH DC component)
    let model = GaussianModel::new(gaussians, sh_coeffs, 0)?;

    // Initialize GPU rasterizer
    let config = RasterConfig::default();
    let mut rasterizer = Rasterizer::new(&config)?;

    // Set up camera
    let camera = RenderCamera::look_at(
        [0.0, 0.0, 2.0],    // eye position
        [0.0, 0.0, 0.0],    // look at target
        [0.0, 1.0, 0.0],    // up vector
        std::f32::consts::FRAC_PI_4,  // fov_y
        1.0,                // aspect ratio
    );

    // Render frame
    let output = rasterizer.forward(&model, &camera)?;

    // Save image
    output.color.save("output.png").map_err(|e| {
        oxigaf::OxigafError::Render(
            oxigaf::render::RenderError::Io(
                format!("Failed to save image: {}", e)
            )
        )
    })?;

    Ok(())
}
```

## Project Structure

```text
oxigaf (meta-crate)
├── oxigaf-flame     — FLAME model + mesh utilities + safetensors I/O
├── oxigaf-diffusion — Multi-view diffusion: Latent Upsampler, IP-Adapter, CFG
├── oxigaf-render    — wgpu 3DGS rasterizer with verified gradients
├── oxigaf-trainer   — Training orchestration + TensorBoard + LPIPS
├── oxigaf-bridge    — PyTorch ⇔ OxiGAF weight conversion
└── oxigaf-cli       — Command-line interface
```

## Documentation

- [API Documentation](https://docs.rs/oxigaf)
- [Repository](https://github.com/cool-japan/oxigaf)
- [Crate](https://crates.io/crates/oxigaf)

## Version Compatibility

| Component | Minimum Version | Tested Version |
|-----------|-----------------|----------------|
| Rust      | 1.75+           | 1.85.0         |
| wgpu      | 27.x            | 28.x           |
| candle    | 0.9.x           | 0.9.x          |
| nalgebra  | 0.34.x          | 0.34.x         |
| glam      | 0.31.x          | 0.31.x         |

### GPU Requirements

- **wgpu**: Any Vulkan 1.1+ / Metal 2.0+ / DX12 GPU
- **candle CUDA**: NVIDIA compute capability 7.0+ (Volta and newer)
- **candle Metal**: Apple M1+ chips

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE))
