# oxigaf-cli

Command-line interface for OxiGAF — Gaussian Avatar Reconstruction.

## Overview

The OxiGAF CLI provides a complete toolkit for working with Gaussian head avatars:

- **Train** — End-to-end avatar reconstruction from monocular images
- **Render** — Render existing avatars from novel viewpoints
- **Export** — Export avatars to PLY, glTF, safetensors, JSON, point cloud, or mesh
- **Convert** — Convert FLAME model files (.pkl to .npy format)
- **Benchmark** — Run performance benchmarks
- **Doctor** — Check system configuration and dependencies
- **Setup** — Download and cache required model weights
- **Cache** — Manage cached assets (list, clean, verify, path)
- **Info** — Inspect a model or data file (`.ply`, `.safetensors`, `.json`) and print its metadata
- **Compare** — Compare two model files and report structural/statistical differences
- **Config-cmd** — Manage `oxigaf.toml` configuration files (init, validate, show)
- **Completions** — Generate shell completion scripts (bash, zsh, fish, PowerShell)

## Installation

```toml
[dependencies]
oxigaf-cli = "0.1"
```

Or install as a binary:

```bash
cargo install oxigaf-cli
```

## Features

| Feature | Description |
|---------|-------------|
| `default` | Minimal configuration (CPU-only) |
| `simd` | SIMD optimizations (requires nightly Rust) |
| `parallel` | Parallel processing with rayon |
| `flash_attention` | Memory-efficient attention |
| `mixed_precision` | FP16/BF16 inference (planned) |
| `gpu_debug` | GPU validation layers |
| `full_performance` | All performance optimizations |
| `all_features` | All available features |

There is no `cuda` or `metal` feature on this crate — GPU rendering itself
always goes through `wgpu` (Vulkan/Metal/DX12/GL, chosen via the `[device]`
section of `oxigaf.toml` or the `OXIGAF_DEVICE_BACKEND` env var; `--device`
on `oxigaf train` selects a GPU *index*, not a backend), and CUDA/Metal
acceleration for the diffusion backend is configured on `candle-core`
directly (see the `oxigaf-diffusion` README).

## Usage

### Train an Avatar

Reconstruct a 3D avatar from a monocular video:

```bash
oxigaf train \
  --input input_video.mp4 \
  --output avatar_output/ \
  --flame-model path/to/flame \
  --max-iterations 1000
```

`--output` is a directory: it receives `final_model.ply`, a `checkpoints/`
subdirectory, and (unless `--no-preview`) a `preview/` turntable render.

### Render from Novel Views

Render an existing avatar from custom viewpoints:

```bash
oxigaf render \
  --model avatar_output/final_model.ply \
  --output renders/ \
  --cameras camera_trajectory.json \
  --width 512 \
  --height 512
```

### Export to Standard Formats

Export avatar to PLY for use in other tools:

```bash
# Export as PLY (3D Gaussian point cloud)
oxigaf export \
  --model avatar_output/final_model.ply \
  --output avatar.ply \
  --format ply

# Export as glTF (textured mesh)
oxigaf export \
  --model avatar_output/final_model.ply \
  --output avatar.gltf \
  --format gltf
```

### Convert FLAME Model

Convert FLAME model from pickle format to numpy:

```bash
oxigaf convert \
  --input generic_model.pkl \
  --output flame_model/ \
  --version 2023
```

### Inspect and Compare Models

```bash
# Print metadata (Gaussian count, SH degree, tensor shapes, ...)
oxigaf info avatar_output/final_model.ply

# Compare two model files
oxigaf compare model_a.ply model_b.ply --threshold 0.85
```

### Check System Configuration

Verify GPU support and dependencies:

```bash
oxigaf doctor
```

### Benchmark Performance

Run performance benchmarks:

```bash
oxigaf benchmark \
  --flame-model path/to/flame \
  --iterations 100 \
  --output benchmark_results.json
```

### Manage Configuration Files

```bash
# Write a fully-populated default oxigaf.toml
oxigaf config-cmd init --output oxigaf.toml

# Parse and validate a config file
oxigaf config-cmd validate oxigaf.toml

# Pretty-print all resolved fields
oxigaf config-cmd show oxigaf.toml
```

### Shell Completions

```bash
oxigaf completions bash > ~/.local/share/bash-completion/completions/oxigaf
oxigaf completions zsh > ~/.zsh/completion/_oxigaf
oxigaf completions fish > ~/.config/fish/completions/oxigaf.fish
```

### Programmatic Usage

The CLI's interactive keyboard-control layer (pause/resume/quit while
training) can also be driven from your own binary that links `oxigaf-cli`
and `oxigaf` directly (add `oxigaf`, `oxigaf-cli`, `wgpu`, `anyhow`, and
`pollster` to that binary's own `Cargo.toml`). `Trainer::new` takes the
Gaussian model, rasterizer config, and GPU device/queue explicitly — see
[`crates/oxigaf-trainer/examples/checkpoint_resume.rs`](../oxigaf-trainer/examples/checkpoint_resume.rs)
for a complete, runnable `setup_gpu()` (the async wgpu adapter/device
request is elided below for brevity):

```rust
use std::sync::atomic::Ordering;

use oxigaf::prelude::*;
use oxigaf_cli::InteractiveController;

// See checkpoint_resume.rs (linked above) for a full implementation:
// requests a wgpu::Instance -> Adapter -> (Device, Queue).
async fn setup_gpu() -> anyhow::Result<(wgpu::Device, wgpu::Queue)> {
    unimplemented!("wgpu adapter/device setup — see checkpoint_resume.rs")
}

async fn run() -> anyhow::Result<()> {
    // Load configuration
    let config = TrainingConfig::default();

    // Build (or load) the initial Gaussian model and rasterizer config.
    let model: GaussianModel = todo!("initial Gaussians, e.g. from a FLAME mesh surface");
    let raster_config = RasterConfig::new().with_resolution(512, 512);
    let (device, queue) = setup_gpu().await?;

    // Trainer::new takes 6 arguments: config, model, raster config,
    // GPU device, GPU queue, and an RNG seed.
    let mut trainer = Trainer::new(config.clone(), model, raster_config, device, queue, 42)?;

    // Interactive controller: `paused`, `lr_adjustment`, `verbose_toggle`,
    // `save_requested`, and `quit_requested` are all `Arc<Atomic*>` fields
    // a keyboard-listener thread and the training loop share.
    let controller = InteractiveController::new();
    controller.start_keyboard_listener();

    for _ in 0..config.total_iterations {
        if controller.quit_requested.load(Ordering::Relaxed) {
            break;
        }

        // Training step
        let output = trainer.train_step()?;

        // Report progress
        println!(
            "Iteration {}/{} | loss: {:.6}",
            output.iteration, config.total_iterations, output.loss.total
        );
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}
```

## Configuration

The CLI supports project configuration files in TOML format (default path:
`./oxigaf.toml`, overridable with `--config`). Run `oxigaf config-cmd init`
to generate a fully-populated file; the excerpt below shows the real section
structure with a few commonly-tuned fields (any field not listed keeps its
built-in default):

```toml
# oxigaf.toml
[model]
flame_model_path = "~/.cache/oxigaf/flame2023"
diffusion_weights_dir = "~/.cache/oxigaf/weights"

[device]
backend = "vulkan"   # vulkan, metal, dx12, or gl
gpu_index = 0

[training]
total_iterations = 15000
views_per_step = 4
image_size = 512
guidance_scale_start = 7.5
guidance_scale_end = 3.0

[training.init]
num_rigid_gaussians = 50000
num_flexible_gaussians = 10000
sh_degree = 3

[training.optimizer]
position_lr = 0.00016
rotation_lr = 0.001
scale_lr = 0.005
opacity_lr = 0.05
sh_lr = 0.0025
beta1 = 0.9
beta2 = 0.999

# [training.density_control] and [training.loss] are also nested under
# [training] — see `oxigaf config-cmd show oxigaf.toml` for every field.

[output]
checkpoint_interval = 1000
log_interval = 50
export_format = "ply"
```

Load configuration with:

```bash
oxigaf train --config oxigaf.toml --input video.mp4 --output avatar_output/ --flame-model path/to/flame
```

Configuration is resolved with the following priority (highest to lowest):
CLI arguments > `OXIGAF_*` environment variables > `--config` file (or
`./oxigaf.toml`) > `~/.config/oxigaf/config.toml` > built-in defaults.

## Output Formats

### Training Output

`oxigaf train` writes into the `--output` directory:

- `final_model.ply` — Trained Gaussian model (PLY point cloud)
- `checkpoints/ckpt_<iteration>.json` — Periodic training checkpoints
- `checkpoints/final.json` — Final checkpoint
- `preview/view_<i>.png` — Turntable preview renders (unless `--no-preview`)

### Render Output

- `*.png` — RGB images (default)
- `*.jpg` — JPEG images (lossy compression)
- `*.exr` — High dynamic range (HDR) images

### Export Formats (`oxigaf export --format <fmt>`)

- `ply` — Point cloud (3D Gaussian Splatting), ASCII or binary (`--ply-format`)
- `gltf` — glTF 2.0 JSON document plus a companion `.bin` buffer (there is no
  packed `.glb` output)
- `safetensors` — Native format (all Gaussian parameters)
- `json` — JSON checkpoint format
- `pointcloud` — Colored PLY point cloud (xyzirgb) from SH DC coefficients
- `mesh` — Surface Nets triangle mesh, written as binary little-endian PLY

## Logging

Control verbosity with `-v` flags:

```bash
# Quiet (errors only)
oxigaf train --quiet ...

# Normal (info)
oxigaf train ...

# Verbose (debug)
oxigaf train -v ...

# Very verbose (trace)
oxigaf train -vv ...
```

Save logs to file:

```bash
oxigaf train \
  --log-file training.log \
  --log-rotation daily \
  --log-max-files 7 \
  ...
```

## Documentation

- [API Documentation](https://docs.rs/oxigaf-cli)
- [Repository](https://github.com/cool-japan/oxigaf)
- [Crate](https://crates.io/crates/oxigaf-cli)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE))
