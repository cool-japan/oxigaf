# oxigaf-diffusion

Multi-view diffusion model inference for GAF.

## Overview

This crate implements the multi-view diffusion pipeline for Gaussian Avatar Framework (GAF):

- **CLIP image encoding** — Extract semantic features from input images
- **Multi-view U-Net** — Generate novel views with camera-conditioned cross-view attention
- **Latent Upsampler** — 32×32 → 64×64 latent upsampling (sd-x2-latent-upscaler) for 512×512 output
- **IP-Adapter** — Identity-preserving image conditioning for consistent face generation
- **Classifier-Free Guidance (CFG)** — Quality improvement with configurable guidance scale (1.0–20.0)
- **VAE decoding** — Decode latent representations to RGB images
- **DDIM scheduling** — Fast sampling with 50-100 steps (vs 1000 for DDPM)
- **Flash Attention** — Memory-efficient O(N) attention for large images

The pipeline takes a single input image and generates multiple novel views of
the subject, which are then used to initialize and optimize 3D Gaussians.
Output is **256×256 by default** (`DiffusionConfig::image_size`); setting
`upsampler_mode: Some(UpsamplerMode::SdX2)` (or `BilinearVae`) upsamples to
**512×512**.

**v0.1.2 — what's included:**
- Multi-view generation pipeline (Latent Upsampler + IP-Adapter + CFG) — 256×256 by default, 512×512 with the upsampler enabled
- 66 tests (all passing)
- Benchmarks: standard vs Flash Attention, sequence lengths, DDIM scheduler

## Installation

```toml
[dependencies]
oxigaf-diffusion = "0.1"
```

## Features

| Feature | Description |
|---------|-------------|
| `default` | `["flash_attention"]` — CPU with memory-efficient attention |
| `flash_attention` | Memory-efficient O(N) attention (enabled by default) |
| `mixed_precision` | FP16/BF16 inference (planned, not yet implemented) |
| `gpu_debug` | NaN/Inf debug hooks (`debug_hooks::assert_finite`, `DebugConfig`) |

### Feature Details

- **`flash_attention`**: Block-based attention computation
  - Reduces memory usage by 2-4× for large images
  - Maintains quality while being faster
  - Enabled by default (part of the `default` feature set)

- **`mixed_precision`**: FP16/BF16 inference — the flag exists but the
  implementation is not yet in place (see `Cargo.toml`: "currently a
  placeholder")

- **`gpu_debug`**: Turns on NaN/Inf assertions (`debug_hooks::assert_finite`,
  configured via `DebugConfig`) during the diffusion forward pass, useful
  for diagnosing numerical instability

### GPU / BLAS Backends

This crate has no `accelerate`, `cuda`, or `metal` feature of its own —
platform-specific BLAS/GPU backends were removed from it in v0.1.2.
`candle-core`/`candle-nn` resolve (at the workspace level) to the COOLJAPAN
fork `oxicandle-core`/`oxicandle-nn` with `default-features = false`, so by
default this crate is Pure Rust / CPU-only. To opt into a backend, enable
the feature on the same resolved package from your own `Cargo.toml`:

```toml
[dependencies]
oxigaf-diffusion = "0.1"
# candle-core/-nn resolve to the COOLJAPAN fork; opt into a backend feature
# on that same package (see oxigaf-diffusion/Cargo.toml for this note):
candle-core = { package = "oxicandle-core", version = "0.11.0", features = ["metal"] } # macOS GPU
candle-nn   = { package = "oxicandle-nn",   version = "0.11.0", features = ["metal"] }
# Swap "metal" for "accelerate" (macOS BLAS) or "cuda" (NVIDIA GPU) instead.
```

## Usage

### Basic Multi-View Inference

```rust
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use image::imageops::FilterType;
use oxigaf_diffusion::{DiffusionConfig, DiffusionError, MultiViewDiffusionPipeline};

/// Load an image and lay it out as the `(1, 3, 224, 224)` tensor that
/// `generate()` expects for CLIP conditioning (see `pipeline.rs`).
fn load_reference_image(path: &str, device: &Device) -> Result<Tensor, DiffusionError> {
    let img = image::open(path)
        .map_err(|e| DiffusionError::ImageProcessingError(format!("failed to open {path}: {e}")))?
        .resize_exact(224, 224, FilterType::Triangle)
        .into_rgb32f(); // HWC, values already in [0, 1]

    let mut chw = vec![0f32; 3 * 224 * 224];
    for y in 0..224u32 {
        for x in 0..224u32 {
            let px = img.get_pixel(x, y);
            for c in 0..3 {
                chw[c * 224 * 224 + (y * 224 + x) as usize] = px[c];
            }
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, 224, 224), device)?)
}

/// Convert a `(3, H, W)` `[0, 1]` output tensor to a PNG file.
fn save_tensor_as_png(t: &Tensor, width: u32, height: u32, path: &str) -> Result<(), DiffusionError> {
    let data = t.contiguous()?.flatten_all()?.to_vec1::<f32>()?; // CHW, row-major
    let (w, h) = (width as usize, height as usize);
    let mut buf = image::RgbImage::new(width, height);
    for y in 0..h {
        for x in 0..w {
            let px = [0usize, 1, 2].map(|c| (data[c * h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8);
            buf.put_pixel(x as u32, y as u32, image::Rgb(px));
        }
    }
    buf.save(path)
        .map_err(|e| DiffusionError::ImageProcessingError(format!("failed to save {path}: {e}")))
}

fn main() -> Result<(), DiffusionError> {
    let device = Device::Cpu;

    // Configure diffusion pipeline. DiffusionConfig has ~25 fields covering
    // U-Net architecture, attention, and CFG — override what you need and
    // take the rest from Default (num_views: 4, guidance_scale: 3.0, ...).
    let config = DiffusionConfig {
        num_views: 4,
        ..Default::default()
    };

    // Load pre-trained model weights from a directory containing
    // unet/, vae/, and image_encoder/ safetensors files.
    let mut pipeline = MultiViewDiffusionPipeline::load(
        config.clone(),
        Path::new("path/to/model/weights"),
        &device,
    )?;

    let reference_image = load_reference_image("portrait.jpg", &device)?;

    // Normal-map latents: `(num_views, latent_channels, latent_size, latent_size)`.
    // Zeroed here as a placeholder — a real caller renders per-view normal
    // maps and VAE-encodes them (see `DiffusionTargetGenerator` in
    // `oxigaf-trainer` for a worked, production example).
    let normal_map_latents = Tensor::zeros(
        (config.num_views, config.latent_channels, config.latent_size, config.latent_size),
        DType::F32,
        &device,
    )?;

    // Camera poses: `(num_views, camera_pose_dim)` flattened 4x3 extrinsics
    // per view. Zeroed here — see "Custom Camera Poses" below for how to
    // build this from real camera rotation/translation.
    let camera_poses = Tensor::zeros((config.num_views, config.camera_pose_dim), DType::F32, &device)?;

    // Generate multiple views (note: `generate` requires `&mut pipeline`).
    let output = pipeline.generate(&reference_image, &normal_map_latents, &camera_poses, 0)?;

    // output.images is Vec<Tensor>, each (3, H, W) in [0, 1].
    for (i, view) in output.images.iter().enumerate() {
        save_tensor_as_png(view, output.width, output.height, &format!("view_{i}.png"))?;
    }

    println!("Generated {} novel views", output.images.len());

    Ok(())
}
```

### Custom Camera Poses

`generate()` takes camera poses as a `(num_views, 12)` tensor — each row is a
flattened 4×3 world-to-camera extrinsics matrix (9 rotation floats, then 3
translation floats), not a struct of azimuth/elevation/distance. Build it
from `oxigaf_flame::Camera` the same way `oxigaf-trainer`'s
`cameras_to_tensor` (`src/diffusion_target.rs`) does. This needs
`oxigaf-flame` and `nalgebra` as direct dependencies too (both are already
transitive dependencies of `oxigaf-diffusion`, but Cargo does not expose a
crate's own dependencies to its downstream users):

```rust
use candle_core::{DType, Device, Tensor};
use nalgebra as na;
use oxigaf_diffusion::{DiffusionConfig, DiffusionError, MultiViewDiffusionPipeline};
use oxigaf_flame::Camera;

/// Flatten cameras into the `(N, 12)` pose tensor `generate()` expects:
/// row-major rotation (9 floats) then translation (3 floats) per view.
fn cameras_to_pose_tensor(cameras: &[Camera], device: &Device) -> Result<Tensor, DiffusionError> {
    let mut data = vec![0.0f32; cameras.len() * 12];
    for (i, cam) in cameras.iter().enumerate() {
        for r in 0..3 {
            for c in 0..3 {
                data[i * 12 + r * 3 + c] = cam.rotation[(r, c)];
            }
        }
        data[i * 12 + 9] = cam.translation.x;
        data[i * 12 + 10] = cam.translation.y;
        data[i * 12 + 11] = cam.translation.z;
    }
    Ok(Tensor::from_vec(data, (cameras.len(), 12), device)?)
}

fn main() -> Result<(), DiffusionError> {
    let device = Device::Cpu;

    // Four views around the subject: 0°, 90°, 180°, 270° yaw about Y.
    let cameras: Vec<Camera> = [0.0f32, 90.0, 180.0, 270.0]
        .into_iter()
        .map(|deg| {
            let mut cam = Camera::default_front(512, 512);
            let (s, c) = deg.to_radians().sin_cos();
            cam.rotation = na::Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
            cam.translation.z = 2.0;
            cam
        })
        .collect();
    let camera_poses = cameras_to_pose_tensor(&cameras, &device)?;

    let config = DiffusionConfig {
        num_views: cameras.len(),
        ..Default::default()
    };
    let mut pipeline = MultiViewDiffusionPipeline::load(
        config.clone(),
        std::path::Path::new("path/to/model/weights"),
        &device,
    )?;

    // See `load_reference_image` under "Basic Multi-View Inference" above.
    let reference_image = load_reference_image("portrait.jpg", &device)?;
    let normal_map_latents = Tensor::zeros(
        (config.num_views, config.latent_channels, config.latent_size, config.latent_size),
        DType::F32,
        &device,
    )?;

    // Generate views with custom cameras
    let output = pipeline.generate(&reference_image, &normal_map_latents, &camera_poses, 0)?;

    println!("Generated {} views with custom camera poses", output.images.len());

    Ok(())
}
```

### DDIM Scheduler Configuration

```rust
use oxigaf_diffusion::{DdimScheduler, PredictionType};

fn main() {
    // Create a DDIM scheduler (SD-2.1 defaults: scaled-linear beta schedule,
    // 1000 training timesteps). `new` is infallible — it only builds the
    // alpha-cumprod table; call `set_timesteps` to pick the inference stride.
    let mut scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);
    scheduler.set_timesteps(50);

    let timesteps = scheduler.timesteps();
    println!("Using {} inference steps", timesteps.len());
    println!("Timesteps: {:?}", timesteps);
}
```

### Memory-Efficient Inference with Flash Attention

```rust
use candle_core::{DType, Device, Tensor};
use oxigaf_diffusion::{DiffusionConfig, DiffusionError, MultiViewDiffusionPipeline};

fn main() -> Result<(), DiffusionError> {
    let device = Device::Cpu;

    // Enable flash attention for larger batches of views (already the
    // default when the `flash_attention` feature is active — set here
    // explicitly for clarity).
    let config = DiffusionConfig {
        num_views: 8,
        use_flash_attention: true,
        ..Default::default()
    };

    let mut pipeline = MultiViewDiffusionPipeline::load(
        config.clone(),
        std::path::Path::new("path/to/model/weights"),
        &device,
    )?;

    // See `load_reference_image` under "Basic Multi-View Inference" above.
    let reference_image = load_reference_image("high_res_portrait.jpg", &device)?;
    let normal_map_latents = Tensor::zeros(
        (config.num_views, config.latent_channels, config.latent_size, config.latent_size),
        DType::F32,
        &device,
    )?;
    let camera_poses = Tensor::zeros((config.num_views, config.camera_pose_dim), DType::F32, &device)?;

    let output = pipeline.generate(&reference_image, &normal_map_latents, &camera_poses, 0)?;

    println!(
        "Generated {} views with flash attention",
        output.images.len()
    );

    Ok(())
}
```

## Pipeline Components

### CLIP Image Encoder

Extracts semantic features from input images using CLIP ViT-H/14 (Vision Transformer):

- Input: RGB image (224×224)
- Output: 1280-dimensional feature vector (`DiffusionConfig::clip_embed_dim`)
- Loads externally-supplied pre-trained weights (`image_encoder/model.safetensors`)

### Multi-View U-Net

Denoises latent representations with camera-conditioned attention:

- Camera-conditioned cross-attention for view consistency
- Multi-scale feature pyramid (4 levels)
- Skip connections for detail preservation
- Supports batch processing of multiple views

### VAE Decoder

Decodes latent representations to RGB images:

- Latent space: 4 channels
- RGB output: 3 channels
- Upsampling factor: 8× (e.g., with the latent upsampler enabled, 64×64 latent → 512×512 RGB; by default, 32×32 latent → 256×256 RGB)

### Latent Upsampler (v0.1.1)

Upscales latent representations from 32×32 to 64×64 for 512×512 output:

- Separate U-Net (`upsampler.rs`) from `stabilityai/sd-x2-latent-upscaler`
- 10-step DDIM denoising in latent space
- Fallback: `BilinearVae` mode for CPU inference

### IP-Adapter (v0.1.1)

Adds pixel-level identity conditioning:

- Additional `attn_ip` cross-attention layer in transformer blocks
- Context = VAE-encoded reference image
- Ensures face identity consistency across all generated views

### Classifier-Free Guidance (v0.1.1)

Improves generation quality via dual forward pass:

- Conditional: full CLIP + IP embeddings
- Unconditional: zero embeddings
- `noise_pred = uncond + guidance_scale * (cond - uncond)`
- Configurable `guidance_scale` (default: 3.0, range: 1.0–20.0)

### DDIM Scheduler

Fast sampling with fewer steps than DDPM:

- **DDPM**: 1000 steps (slow)
- **DDIM**: 50-100 steps (20× faster)
- Deterministic sampling for reproducibility
- Supports both ε-prediction and v-prediction

## Performance

Inference times on various hardware (512×512 resolution — i.e. with
`upsampler_mode` enabled —, 4 views, 50 steps):

| Hardware | Time (with flash attention) | Time (without) |
|----------|----------------------------|----------------|
| CPU (Apple M2 Max) | ~12s | ~25s |
| Apple M2 Max (Metal)\* | ~3s | ~6s |
| NVIDIA RTX 4090 (CUDA)\* | ~1.5s | ~3s |
| NVIDIA RTX 3080 (CUDA)\* | ~2.5s | ~5s |

\* Requires enabling the `metal`/`cuda` feature on `candle-core`/`candle-nn`
from your own `Cargo.toml` — see "GPU / BLAS Backends" above. This crate
itself has no `metal`/`cuda` feature.

Memory usage:

| Resolution | Standard Attention | Flash Attention |
|------------|-------------------|-----------------|
| 512×512 | ~8 GB | ~4 GB |
| 1024×1024 | ~24 GB | ~8 GB |

## Statistics

- **Tests**: 66 (all passing)
- **Source files**: `attention.rs`, `camera.rs`, `clip.rs`, `flash_attention.rs`, `pipeline.rs`, `scheduler.rs`, `unet.rs`, `upsampler.rs`, `vae.rs`
- **Benchmarks**: `diffusion_bench.rs`, `flash_attention_bench.rs`

## Documentation

- [API Documentation](https://docs.rs/oxigaf-diffusion)
- [Repository](https://github.com/cool-japan/oxigaf)
- [Crate](https://crates.io/crates/oxigaf-diffusion)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE))
