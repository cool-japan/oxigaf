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

The pipeline takes a single input image and generates multiple novel views of the subject at **512×512 resolution**, which are then used to initialize and optimize 3D Gaussians.

**v0.1.2 — what's included:**
- Full 512×512 multi-view generation pipeline (Latent Upsampler + IP-Adapter + CFG)
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
| `default` | `["accelerate", "flash_attention"]` — CPU with optimizations |
| `accelerate` | Platform-native BLAS/LAPACK (Accelerate on macOS, OpenBLAS on Linux) |
| `cuda` | NVIDIA GPU acceleration (requires CUDA toolkit) |
| `metal` | Apple Silicon GPU acceleration (M1/M2/M3) |
| `flash_attention` | Memory-efficient O(N) attention (enabled by default) |
| `mixed_precision` | FP16/BF16 inference (planned, not yet implemented) |

### Feature Details

- **`accelerate`**: Uses native BLAS/LAPACK for tensor operations
  - macOS: Apple Accelerate framework
  - Linux: OpenBLAS or Intel MKL
  - Windows: OpenBLAS

- **`cuda`**: NVIDIA GPU acceleration via candle CUDA backend
  - Requires CUDA toolkit (11.8+ recommended)
  - Requires compute capability 7.0+ (Volta and newer)
  - Not available on macOS

- **`metal`**: Apple Silicon GPU acceleration via Metal
  - macOS only
  - Optimized for M1/M2/M3 chips
  - Automatic selection on compatible hardware

- **`flash_attention`**: Block-based attention computation
  - Reduces memory usage by 2-4× for large images
  - Maintains quality while being faster
  - Enabled by default for efficiency

### Example Usage

```toml
# CPU-only with flash attention (default)
oxigaf-diffusion = "0.1"

# Apple Silicon with Metal acceleration
oxigaf-diffusion = { version = "0.1", features = ["metal", "flash_attention"] }

# NVIDIA GPU with CUDA
oxigaf-diffusion = { version = "0.1", features = ["cuda", "flash_attention"] }
```

## Usage

### Basic Multi-View Inference

```rust
use oxigaf_diffusion::{
    MultiViewDiffusionPipeline,
    DiffusionConfig,
    PredictionType
};
use image;

fn main() -> Result<(), oxigaf_diffusion::DiffusionError> {
    // Load input image
    let input_image = image::open("portrait.jpg").map_err(|e| {
        oxigaf_diffusion::DiffusionError::ImageLoad(
            format!("Failed to load image: {}", e)
        )
    })?;

    // Configure diffusion pipeline
    let config = DiffusionConfig {
        num_views: 4,
        num_inference_steps: 50,
        guidance_scale: 7.5,
        use_flash_attention: true,
        prediction_type: PredictionType::VPrediction,
    };

    // Load pre-trained model weights
    let pipeline = MultiViewDiffusionPipeline::from_pretrained(
        "path/to/model/weights",
        &config,
    )?;

    // Generate multiple views
    let output = pipeline.generate(&input_image, None)?;

    // Save generated views
    for (i, view) in output.views.iter().enumerate() {
        view.save(format!("view_{}.png", i)).map_err(|e| {
            oxigaf_diffusion::DiffusionError::ImageSave(
                format!("Failed to save view {}: {}", i, e)
            )
        })?;
    }

    println!("Generated {} novel views", output.views.len());

    Ok(())
}
```

### Custom Camera Poses

```rust
use oxigaf_diffusion::{
    MultiViewDiffusionPipeline,
    DiffusionConfig,
    camera::CameraParams
};
use nalgebra as na;

fn main() -> Result<(), oxigaf_diffusion::DiffusionError> {
    let input_image = image::open("portrait.jpg").map_err(|e| {
        oxigaf_diffusion::DiffusionError::ImageLoad(
            format!("Failed to load image: {}", e)
        )
    })?;

    let config = DiffusionConfig::default();
    let pipeline = MultiViewDiffusionPipeline::from_pretrained(
        "path/to/model/weights",
        &config,
    )?;

    // Define custom camera poses (4 views around the subject)
    let camera_poses = vec![
        CameraParams {
            azimuth: 0.0,       // Front view
            elevation: 0.0,
            distance: 2.0,
        },
        CameraParams {
            azimuth: std::f32::consts::FRAC_PI_4,  // 45° right
            elevation: 0.0,
            distance: 2.0,
        },
        CameraParams {
            azimuth: -std::f32::consts::FRAC_PI_4, // 45° left
            elevation: 0.0,
            distance: 2.0,
        },
        CameraParams {
            azimuth: 0.0,
            elevation: std::f32::consts::FRAC_PI_6,  // 30° up
            distance: 2.0,
        },
    ];

    // Generate views with custom cameras
    let output = pipeline.generate(&input_image, Some(&camera_poses))?;

    println!("Generated {} views with custom camera poses", output.views.len());

    Ok(())
}
```

### DDIM Scheduler Configuration

```rust
use oxigaf_diffusion::{
    DdimScheduler,
    PredictionType
};

fn main() -> Result<(), oxigaf_diffusion::DiffusionError> {
    // Create DDIM scheduler for fast sampling
    let scheduler = DdimScheduler::new(
        1000,                        // num_train_timesteps
        50,                          // num_inference_steps
        0.0,                         // beta_start
        0.02,                        // beta_end
        PredictionType::VPrediction, // prediction_type
    )?;

    // Get timesteps for inference
    let timesteps = scheduler.timesteps();

    println!("Using {} inference steps", timesteps.len());
    println!("Timesteps: {:?}", timesteps);

    Ok(())
}
```

### Memory-Efficient Inference with Flash Attention

```rust
use oxigaf_diffusion::{MultiViewDiffusionPipeline, DiffusionConfig};

fn main() -> Result<(), oxigaf_diffusion::DiffusionError> {
    let input_image = image::open("high_res_portrait.jpg").map_err(|e| {
        oxigaf_diffusion::DiffusionError::ImageLoad(
            format!("Failed to load image: {}", e)
        )
    })?;

    // Enable flash attention for large images
    let config = DiffusionConfig {
        num_views: 8,
        num_inference_steps: 50,
        guidance_scale: 7.5,
        use_flash_attention: true,  // Reduces memory by 2-4×
        prediction_type: PredictionType::VPrediction,
    };

    let pipeline = MultiViewDiffusionPipeline::from_pretrained(
        "path/to/model/weights",
        &config,
    )?;

    let output = pipeline.generate(&input_image, None)?;

    println!(
        "Generated {} high-resolution views with flash attention",
        output.views.len()
    );

    Ok(())
}
```

## Pipeline Components

### CLIP Image Encoder

Extracts semantic features from input images using CLIP ViT (Vision Transformer):

- Input: RGB image (224×224)
- Output: 768-dimensional feature vector
- Pre-trained on 400M image-text pairs

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
- Upsampling factor: 8× (e.g., 64×64 latent → 512×512 RGB)

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
- Configurable `guidance_scale` (default: 7.5, range: 1.0–20.0)

### DDIM Scheduler

Fast sampling with fewer steps than DDPM:

- **DDPM**: 1000 steps (slow)
- **DDIM**: 50-100 steps (20× faster)
- Deterministic sampling for reproducibility
- Supports both ε-prediction and v-prediction

## Performance

Inference times on various hardware (512×512 resolution, 4 views, 50 steps):

| Hardware | Time (with flash attention) | Time (without) |
|----------|----------------------------|----------------|
| CPU (Apple M2 Max) | ~12s | ~25s |
| Apple M2 Max (Metal) | ~3s | ~6s |
| NVIDIA RTX 4090 (CUDA) | ~1.5s | ~3s |
| NVIDIA RTX 3080 (CUDA) | ~2.5s | ~5s |

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
