# oxigaf-cli

Command-line interface for OxiGAF — Gaussian Avatar Reconstruction.

## Overview

The OxiGAF CLI provides a complete toolkit for working with Gaussian head avatars:

- **Train** — End-to-end avatar reconstruction from monocular images
- **Render** — Render existing avatars from novel viewpoints
- **Export** — Export avatars to PLY, glTF, or safetensors
- **Convert** — Convert FLAME model files (.pkl to .npy format)
- **Benchmark** — Run performance benchmarks
- **Doctor** — Check system configuration and dependencies
- **Setup** — Download and cache required model weights
- **Cache** — Manage cached assets (list, clean, verify, path)

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
| `cuda` | NVIDIA GPU acceleration |
| `metal` | Apple Silicon GPU acceleration |
| `simd` | SIMD optimizations (requires nightly Rust) |
| `parallel` | Parallel processing with rayon |
| `flash_attention` | Memory-efficient attention |
| `mixed_precision` | FP16/BF16 inference (planned) |
| `gpu_debug` | GPU validation layers |
| `full_performance` | All performance optimizations |
| `all_features` | All available features |

## Usage

### Train an Avatar

Reconstruct a 3D avatar from a monocular video:

```bash
oxigaf train \
  --input input_video.mp4 \
  --output avatar.safetensors \
  --flame-model path/to/flame \
  --iterations 1000
```

### Render from Novel Views

Render an existing avatar from custom viewpoints:

```bash
oxigaf render \
  --avatar avatar.safetensors \
  --output-dir renders/ \
  --camera-path camera_trajectory.json \
  --width 512 \
  --height 512
```

### Export to Standard Formats

Export avatar to PLY for use in other tools:

```bash
# Export as PLY (3D Gaussian point cloud)
oxigaf export \
  --avatar avatar.safetensors \
  --output avatar.ply \
  --format ply

# Export as glTF (textured mesh)
oxigaf export \
  --avatar avatar.safetensors \
  --output avatar.gltf \
  --format gltf
```

### Convert FLAME Model

Convert FLAME model from pickle format to numpy:

```bash
oxigaf convert \
  --input generic_model.pkl \
  --output flame_model/ \
  --format npy
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
  --report benchmark_results.json
```

### Programmatic Usage

You can also use the CLI components as a library:

```rust
use oxigaf_cli::{InteractiveController};
use oxigaf::prelude::*;

fn main() -> anyhow::Result<()> {
    // Load configuration
    let config = TrainingConfig::default();

    // Create trainer
    let trainer = Trainer::new(config)?;

    // Run training loop with interactive controls
    let controller = InteractiveController::new()?;

    for iteration in 0..1000 {
        // Check for user commands (pause, stop, etc.)
        if controller.should_stop() {
            break;
        }

        // Training step
        // trainer.step(...)?;

        // Report progress
        println!("Iteration {}/1000", iteration + 1);
    }

    Ok(())
}
```

## Configuration

The CLI supports configuration files in TOML format:

```toml
# config.toml
[training]
iterations = 1000
learning_rate = 0.001
batch_size = 4

[render]
width = 512
height = 512
tile_size = 16

[optimizer]
lr_positions = 0.00016
lr_rotations = 0.001
lr_scales = 0.005
lr_opacities = 0.05
lr_sh = 0.0025
```

Load configuration with:

```bash
oxigaf train --config config.toml --input video.mp4
```

## Output Formats

### Training Output

- `*.safetensors` — Trained Gaussian model (recommended)
- `checkpoint_*.json` — Training checkpoints with metadata

### Render Output

- `*.png` — RGB images (default)
- `*.jpg` — JPEG images (lossy compression)
- `*.exr` — High dynamic range (HDR) images

### Export Formats

- `*.ply` — Point cloud (3D Gaussian Splatting)
- `*.gltf` / `*.glb` — glTF 2.0 mesh with textures
- `*.safetensors` — Native format (all Gaussian parameters)

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
