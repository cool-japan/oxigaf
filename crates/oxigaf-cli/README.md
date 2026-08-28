# oxigaf-cli

Command-line interface for OxiGAF — Gaussian Avatar Reconstruction.

**Version 0.1.2** (2026-08-28) — 32 subcommands, 3105 tests passing (`--all-features`).

## Overview

The OxiGAF CLI provides a complete toolkit for working with Gaussian head avatars.
Core pipeline commands (worked examples below):

- **Train** — End-to-end avatar reconstruction from a directory of frames
- **Render** — Render existing avatars from novel viewpoints
- **Export** — Export avatars to PLY, glTF, safetensors, JSON, point cloud, mesh, or all of them at once
- **Convert** — Convert FLAME model files (.pkl to .npy format)
- **Benchmark** — Run performance benchmarks
- **Doctor** — Check system configuration and dependencies
- **Setup** — Download and cache required model weights
- **Cache** — Manage cached assets (list, clean, verify, path)
- **Info** — Inspect a model or data file (`.ply`, `.safetensors`, `.json`) and print its metadata
- **Compare** — Compare two model files and report structural/statistical differences
- **Config** — Manage `oxigaf.toml` configuration files (init, validate, show); the
  earlier name `config-cmd` still works as an alias
- **Completions** — Generate shell completion scripts (bash, zsh, fish, PowerShell, elvish)

Beyond the core pipeline, the CLI ships 20 additional read-only/tooling subcommands
(`oxigaf <command> --help` for full details on any of them):

| Command | What it does |
|---------|--------------|
| `anim` | Inspect and transform per-frame Gaussian animation sequences |
| `analyze` | Read-only inspection: colour calibration, model diffs, image metrics |
| `batch` | Run many model conversions as one dependency-ordered batch |
| `camera` | Author camera paths and evaluate arcball navigation |
| `dataset` | Scan, validate and split training datasets |
| `inspect` | Read-only interrogation of models, PLY files and memory budgets |
| `monitor` | Render a training run's metrics stream as a live dashboard |
| `perf` | Micro-benchmark the CPU-side numeric kernels |
| `preset` | Inspect and apply named training hyper-parameter presets |
| `preview` | Drive a model's camera and re-render it to a live image file |
| `pipeline` | Run the reconstruction workflow one composable stage at a time (`plan`/`track`/`diffuse`/`export`/`status`) |
| `profile` | Turn a phase-timing log into a bottleneck report |
| `quality` | Quality-gate rendered images against references and hunt for artefacts |
| `report` | Build comparison reports across training runs |
| `runs` | Create, list, prune and retire training run workspaces |
| `scene` | Whole-scene operations: alignment, analysis, merging, filtering, optimisation, LOD, compression and streaming plans (14 sub-subcommands) |
| `sweep` | Plan and score hyper-parameter sweeps |
| `training` | Analyse a finished training run: summary, smoothing, reports, resume recommendations and timing traces |
| `video` | Turn a directory of rendered frames into a GIF, a frame sequence, a manifest, or a self-contained HTML viewer |
| `workspace` | Browse and compare the checkpoints of a run directory |

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
| `default` | No optional features enabled (all core commands function; `rayon`-based `--parallel` rendering is always available) |
| `simd` | SIMD optimizations (requires nightly Rust) |
| `parallel` | Parallel processing with rayon (forwarded to `oxigaf/parallel`) |
| `flash_attention` | Memory-efficient attention (forwarded to `oxigaf/flash_attention`; opt-in as of 0.1.2 — no longer pulled in by default via `oxigaf-diffusion`) |
| `mixed_precision` | FP16/BF16 inference (planned) |
| `npz` | Forwards to `oxigaf-flame/npz` (`FlameSequence::from_npz`); the CLI's own `convert` subcommand reads `.npz` through its own reader and does not need this |
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

Reconstruct a 3D avatar from a directory of frames extracted from a monocular
video. OxiGAF is pure Rust and bundles no video demuxer, so `--input` must be
a directory of images (or a single frame) — a `.mp4`/`.mov` container is
rejected with an actionable error message; extract frames first with an
external tool (e.g. `ffmpeg -i clip.mp4 frames/%05d.png`):

```bash
oxigaf train \
  --input frames/ \
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

`--width`/`--height` are optional — omitting them derives a resolution from
`--quality` (`low`/`medium`/`high`/`ultra`) instead of a hardcoded default;
an explicit value always wins over the preset. Add `--parallel N` to render
with a dedicated `rayon` thread pool (`0`, the default, uses the global pool
sized to all CPU cores).

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

# Export every format at once into a directory (model.ply, model.safetensors,
# model.glb, model.json)
oxigaf export \
  --model avatar_output/final_model.ply \
  --output avatar_export/ \
  --format all
```

### Convert FLAME Model

Convert FLAME model from pickle format to numpy:

```bash
oxigaf convert \
  --input generic_model.pkl \
  --output flame_model/ \
  --version 2023
```

This writes the 8 files `oxigaf_flame::load_flame_model` reads:
`v_template.npy`, `shapedirs.npy`, `posedirs.npy`, `expressiondirs.npy`,
`j_regressor.npy`, `kintree_table.npy`, `lbs_weights.npy`, `faces.npy` —
plus `uv.npy`/`uv_faces.npy` when the source model carries UV data, and any
landmark index/barycentric `.npy` files present in the source. There is no
`parents.npy`: older releases wrote one (a verbatim copy of
`kintree_table`), but the FLAME loader never read it and the converter no
longer writes it — `oxigaf convert --force` over a directory from an older
release leaves a stale one behind, which `oxigaf convert` flags as a hint.

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

Every target drives a real component — `flame` runs `FlameModel::forward` on
the model at `--flame-model`, `raster`/`train` need a working GPU adapter,
and `export` writes a real PLY to a temp directory. `--flame-model` is
**required** to benchmark the `flame` target: with no explicit `--target`
it is skipped (and noted in the report) when the flag is missing, but
`--target flame` on its own turns a missing `--flame-model` into a hard
error instead of a skip.

### Manage Configuration Files

```bash
# Write a fully-populated default oxigaf.toml
oxigaf config init --output oxigaf.toml

# Parse and validate a config file
oxigaf config validate oxigaf.toml

# Pretty-print all resolved fields
oxigaf config show oxigaf.toml
```

The command is named `config`; the earlier kebab-case name `config-cmd` is
kept as an alias so scripts written against it keep working. `config init
--interactive` also runs a non-interactive hardware-detection wizard that
queries the real `wgpu` adapter and picks VRAM-bound defaults for
`sh_degree`, `views_per_step`, `image_size`, and `max_gaussians`.

### Shell Completions

```bash
oxigaf completions bash > ~/.local/share/bash-completion/completions/oxigaf
oxigaf completions zsh > ~/.zsh/completion/_oxigaf
oxigaf completions fish > ~/.config/fish/completions/oxigaf.fish
```

`powershell` and `elvish` are also supported (`oxigaf completions --help`
lists installation instructions for every shell).

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
`./oxigaf.toml`, overridable with `--config`). Run `oxigaf config init`
to generate a fully-populated file; the excerpt below shows the real section
structure with a few commonly-tuned fields (any field not listed keeps its
built-in default). The example paths below use the macOS cache directory —
the actual default (when a path isn't set explicitly) comes from
`dirs::cache_dir()`, which resolves to `~/Library/Caches/oxigaf` on macOS,
`~/.cache/oxigaf` (or `$XDG_CACHE_HOME/oxigaf`) on Linux, and
`%LOCALAPPDATA%\oxigaf` on Windows; override it with `OXIGAF_CACHE_DIR` on
any platform:

```toml
# oxigaf.toml
[model]
flame_model_path = "~/Library/Caches/oxigaf/flame2023"
diffusion_weights_dir = "~/Library/Caches/oxigaf/weights"

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
# [training] — see `oxigaf config show oxigaf.toml` for every field.

[output]
checkpoint_interval = 1000
log_interval = 50
export_format = "ply"
```

Load configuration with:

```bash
oxigaf train --config oxigaf.toml --input frames/ --output avatar_output/ --flame-model path/to/flame
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
  packed `.glb` output from this command). As of 0.1.2 this is written by
  the workspace's single spec-conformant glTF writer
  (`oxigaf_render::gltf::write_gltf`); earlier 0.1.x output put every
  accessor on one buffer view with no `byteStride`, which glTF 2.0 forbids.
- `safetensors` — Native format (all Gaussian parameters)
- `json` — JSON checkpoint format
- `point-cloud` — Colored PLY point cloud (xyzirgb) from SH DC coefficients
  (note the hyphen — `--format pointcloud` is rejected by the parser)
- `mesh` — Surface Nets triangle mesh, written as binary little-endian PLY
- `all` — PLY, safetensors, glTF and JSON checkpoint written concurrently:
  treats `--output` as a directory and writes `model.ply`,
  `model.safetensors`, `model.glb`, and `model.json`. Its glTF component
  goes through a third, separate `.glb` writer
  (`OXIGAF_gaussians` extension) rather than the spec-conformant one used by
  standalone `--format gltf` above — a documented, deliberate scope limit,
  not an oversight.

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
