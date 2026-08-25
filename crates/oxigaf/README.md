# oxigaf

Pure Rust Gaussian Avatar Reconstruction — unified API for the OxiGAF ecosystem.

OxiGAF implements [GAF: Gaussian Avatar Reconstruction from Monocular Videos via Multi-View Diffusion](https://arxiv.org/abs/2412.10209) (Tang et al., CVPR 2025) in pure Rust.

## What "pure Rust" means here

It describes the **Rust build and runtime**: the default feature set pulls in no C/C++/Fortran code and no Python interpreter, so building and running this crate needs nothing but a Rust toolchain.

It does *not* mean the project is Python-free end to end. Turning the upstream FLAME `.pkl` and PyTorch `.pt` releases into the `.npy` / `.safetensors` files these crates read is still a one-time offline step run by `scripts/convert_flame.py` and `scripts/convert_weights.py`, which require Python and PyTorch. Once those converted files exist, nothing in the Rust runtime touches Python again.

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

### GPU acceleration is not a Cargo feature

There is **no `cuda` and no `metal` feature on `oxigaf`**. GPU acceleration comes from two independent places, neither of which is switched on here:

- The 3DGS rasterizer (`oxigaf-render`) is always GPU-backed through wgpu, which picks Vulkan / Metal / DX12 **at runtime**. Nothing to enable at compile time.
- Diffusion inference (`oxigaf-diffusion`) runs on candle. To move it onto a GPU, depend on candle directly and turn on *its* backend feature:

  ```toml
  # macOS GPU
  candle-core = { package = "oxicandle-core", version = "0.11.0", features = ["metal"] }
  # NVIDIA GPU
  candle-core = { package = "oxicandle-core", version = "0.11.0", features = ["cuda"] }
  ```

  Note that those backends are **not** pure Rust, so they are outside this workspace's default policy.

The features below only affect CPU-side behaviour.

### Performance Optimization Features

| Feature | Description | Speedup |
|---------|-------------|---------|
| `simd` | SIMD-accelerated FLAME operations (requires nightly Rust; accepted but compiled out on stable) | 3-4× faster |
| `parallel` | Parallel batch processing with rayon | Near-linear with cores |
| `flash_attention` | Memory-efficient O(N) attention. **Off by default** — `oxigaf-diffusion`'s own default feature set is empty, so `DiffusionConfig::use_flash_attention` starts `false` unless this is enabled | 2-4× less memory |
| `mixed_precision` | Implemented in `oxigaf-diffusion::mixed_precision`; flips `MixedPrecisionConfig::default().mode` to BF16. Currently rounding-simulation helpers only — no inference or training path reads that config yet, so enabling it does not change numerics | 2× faster on GPUs (once wired) |

### Data Format Features

| Feature | Description |
|---------|-------------|
| `npz` | Forwards to `oxigaf-flame/npz`, which makes `FlameSequence::from_npz` actually read `.npz` archives instead of returning `FlameError::InvalidParams("NPZ support not enabled…")`. Off by default because `.npz` is a niche input format whose reader is slated to be rebuilt on `oxiarc-archive`. It does **not** currently gate the `zip` crate: the workspace declares `ndarray-npy = "0.10"` without `default-features = false`, and ndarray-npy's own default set already activates its `npz` feature, so `zip 6.0.0` is in the dependency graph with or without this flag |

### Debug Features

| Feature | Description |
|---------|-------------|
| `gpu_debug` | Enable wgpu validation layers (adds 10-100× overhead) **and** `oxigaf-diffusion`'s NaN/Inf debug hooks (`debug_hooks::DebugConfig::default().enabled`) |

### Convenience Feature Bundles

| Feature | Enables |
|---------|---------|
| `full_performance` | `simd`, `parallel`, `flash_attention` |
| `all_features` | `simd`, `parallel`, `flash_attention`, `mixed_precision`, `gpu_debug`, `npz` |

`crates/oxigaf/tests/feature_forwarding_tests.rs` asserts that `parallel`, `flash_attention`, `mixed_precision`, `npz` and the `oxigaf-diffusion` half of `gpu_debug` really reach the sub-crate they name, in both the on and off configuration. `simd` and the `oxigaf-render` half of `gpu_debug` are unasserted — neither exposes a marker observable without nightly Rust or a live GPU adapter; the test file's header explains why.

### Examples

```toml
# CPU-only with all optimizations (requires nightly for SIMD)
oxigaf = { version = "0.1", features = ["full_performance"] }

# Stable Rust, everything except the nightly-only SIMD paths
oxigaf = { version = "0.1", features = ["parallel", "flash_attention"] }

# Development build with GPU validation layers and NaN/Inf hooks
oxigaf = { version = "0.1", features = ["gpu_debug"] }

# FLAME tracking sequences stored as .npz
oxigaf = { version = "0.1", features = ["npz"] }
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

`Rasterizer::new` is `async` (wgpu device creation is), takes the config **by value**, and `RenderOutput` hands back raw `Vec<f32>` buffers rather than an `image` type — encoding is left to the caller. `oxigaf` does not re-export an async runtime, so add your own (`pollster = "1"` below).

This is an excerpt, not a standalone file: `create_camera` is defined in [`examples/gaussian_render.rs`](examples/gaussian_render.rs).

```rust,ignore
use oxigaf::prelude::*;
use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

async fn render() -> oxigaf::Result<()> {
    // One opaque Gaussian at the origin, SH degree 0 (DC term only).
    let gaussians = vec![GaussianAttributes {
        position: [0.0, 0.0, 0.0],
        _pad0: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
        scale: [-4.0, -4.0, -4.0],      // Log-scale: exp(-4) ≈ 0.018
        opacity: 2.0,                   // Inverse-sigmoid: sigmoid(2) ≈ 0.88
    }];

    let model = GaussianModel {
        gaussians,
        sh_coeffs: vec![1.0, 0.5, 0.5], // Reddish DC component
        sh_degree: 0,
        // No FLAME binding in this minimal example.
        face_indices: vec![0],
        barycentric: vec![[1.0, 0.0, 0.0]],
        local_offsets: vec![[0.0, 0.0, 0.0]],
        is_rigid: vec![true],
    };

    let config = RasterConfig::new()
        .with_resolution(512, 512)
        .with_sh_degree(0)
        .with_background([0.1, 0.1, 0.15]);

    // Async: requests a wgpu adapter and device.
    let mut rasterizer = Rasterizer::new(config.clone()).await?;

    // `RenderCamera` holds explicit column-major `view_matrix` / `proj_matrix`
    // arrays (`[f32; 16]`) plus `position` and `focal`. There is no `look_at`
    // constructor — build the matrices yourself; `examples/gaussian_render.rs`
    // has a complete, correct look-at + perspective derivation, deliberately
    // not abridged here because a placeholder matrix renders a blank frame
    // with no error.
    let camera = create_camera(config.image_width, config.image_height);

    let output = rasterizer.forward(&model, &camera)?;

    // `output.color_data` is RGBA f32 in [0, 1], length width * height * 4.
    assert_eq!(
        output.color_data.len(),
        (output.width * output.height * 4) as usize
    );

    Ok(())
}

fn main() -> oxigaf::Result<()> {
    pollster::block_on(render())
}
```

A complete, runnable version (camera matrices included, plus PNG encoding) lives in [`examples/gaussian_render.rs`](examples/gaussian_render.rs):

```bash
cargo run -p oxigaf --example gaussian_render
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
| Rust      | 1.87+           | 1.87.0         |
| wgpu      | 30.x            | 30.x           |
| candle    | 0.11.x          | 0.11.x         |
| nalgebra  | 0.35.x          | 0.35.x         |
| glam      | 0.33.x          | 0.33.x         |

`candle` here means the COOLJAPAN fork published as `oxicandle-core` / `oxicandle-nn`, pinned in the workspace at `0.11.0`. It is API-identical to upstream candle 0.11 but selects `fancy-regex` instead of the C Oniguruma library that upstream pulls in through `tokenizers`; the dependency keys stay `candle-core` / `candle-nn`, so source code is unchanged.

The `rust-version = "1.87"` floor is real, not aspirational: the workspace calls `usize::is_multiple_of` (stabilized in 1.87) from `oxigaf-bridge/src/precision.rs` and from `oxigaf-render`'s gradient-verification tests, plus several other APIs stabilized in 1.87.0 across `oxigaf-flame` that `clippy::incompatible_msrv` flags on anything older. Rust 1.85 will **not** build this workspace.

### GPU Requirements

- **wgpu**: Any Vulkan 1.1+ / Metal 2.0+ / DX12 GPU
- **candle CUDA**: NVIDIA compute capability 7.0+ (Volta and newer) — enabled on candle directly, not through an `oxigaf` feature
- **candle Metal**: Apple M1+ chips — likewise enabled on candle directly

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE))
