# oxigaf-diffusion

Multi-view diffusion model inference for GAF.

**Version 0.1.2** (2026-08-28) — **Status: Alpha.** The multi-view
generation pipeline (Latent Upsampler + IP-Adapter + CFG) and the wider
sampler/attention/conditioning/editing/analysis toolkit below (see
[API Overview](#api-overview)) are implemented and covered by 3213 passing
tests plus 51 passing doc tests (1 ignored), `--all-features`. A
PyTorch→SafeTensors weight-conversion script and Python cross-validation are
not yet built (bring your own converted weights) — see [TODO.md](TODO.md)
for the itemized state. The public API is still settling: this release alone
made ten breaking signature/semantics changes — read
["Breaking Changes in 0.1.2"](#breaking-changes-in-012) before upgrading
from 0.1.1.

## Overview

This crate implements the multi-view diffusion pipeline for Gaussian Avatar Framework (GAF):

- **CLIP image encoding** — Extract semantic features from input images
- **Multi-view U-Net** — Generate novel views with camera-conditioned cross-view attention
- **Latent Upsampler** — 32×32 → 64×64 latent upsampling (sd-x2-latent-upscaler) for 512×512 output
- **IP-Adapter** — Identity-preserving image conditioning for consistent face generation
- **Classifier-Free Guidance (CFG)** — Quality improvement with configurable guidance scale (1.0–20.0)
- **VAE decoding** — Decode latent representations to RGB images
- **DDIM scheduling** — Fast sampling with 50-100 steps (vs 1000 for DDPM)
- **Flash Attention** — Memory-efficient O(N) attention for large images (feature: `flash_attention`, opt-in)

The pipeline takes a single input image and generates multiple novel views of
the subject, which are then used to initialize and optimize 3D Gaussians.
Output is **256×256 by default** (`DiffusionConfig::image_size`); setting
`upsampler_mode: Some(UpsamplerMode::SdX2)` (or `BilinearVae`) upsamples to
**512×512**.

**v0.1.2 — what's included:**
- Multi-view generation pipeline (Latent Upsampler + IP-Adapter + CFG) — 256×256 by default, 512×512 with the upsampler enabled
- 3213 unit/integration tests (all passing) + 51 passing doc tests (1 ignored)
- Benchmarks: standard vs Flash Attention, sequence lengths, DDIM scheduler

## Breaking Changes in 0.1.2

Full detail (and the rest of the workspace's changes) is in the top-level
[CHANGELOG.md](../../CHANGELOG.md#012---2026-08-28). The three most likely
to affect existing callers of this crate:

- **`MultiViewDiffusionPipeline::begin_session_from_latents`** now takes a
  single `SessionRequest` instead of seven positional arguments (four of
  them `&Tensor`, and therefore trivially transposable at a call site) — see
  "Custom Camera Poses" below for the shape of the old-style call this
  replaces.
- **`dpm_plus_plus_2m_step`** now takes `prev_x0: Option<&[f32]>` and
  `h_prev: Option<f32>` instead of a single `prev_noise_pred` (6 → 7
  arguments) — the DPM-Solver++ 2M multistep formula needs the previous
  denoised sample and step size, not just the previous noise prediction.
- **`image_editing::sdedit::ImageEditError` was removed.** `sdedit` now
  returns the parent `ImageEditingError` directly instead of a separate enum
  bridged in via `From`; old variants map onto it as `InvalidStrength` /
  `InvalidParam` → `InvalidConfig`, `InvalidMask` → `InvalidImage`,
  `TimestepOutOfRange` → `NoiseLevelOutOfRange`, and `DimensionMismatch`'s
  `got` field is renamed to `actual`.

Also, in brief: `flash_attention` is no longer a `default` feature (see
"Features" below — this is the change most likely to surprise an existing
build); `compute_distillation_loss`, `DistillationStep::aggregate_losses`,
`pad_to_square`, `flip_horizontal`, `flip_vertical`, and `softmax_over_dim`
now return `Result` instead of an infallible value;
`InversionTrajectory::x_0()` / `x_t()` now return `Option<&[f32]>`;
`DdimInversionConfig::refinement_threshold` is now compared against an
RMS (not raw L2) reconstruction error, so a pinned threshold value means
something different and should be re-tuned; `FlowStats::divergence_estimate`
was renamed to `mean_squared_norm`; `EditMask::all_ones` now takes
`(height, width)` (matching `EditMask::new`); `fm_interpolate` /
`fm_target_velocity`'s `Linear` path now honors a non-zero
`FlowMatchingConfig::sigma_min`; and `candle-core`/`candle-nn` moved to the
`oxicandle-core`/`oxicandle-nn` COOLJAPAN fork (0.10 → 0.11 — the dependency
*keys* are unchanged, so no source changes are needed for that one).

## Installation

```toml
[dependencies]
oxigaf-diffusion = "0.1"
```

## Features

| Feature | Description |
|---------|-------------|
| `default` | `[]` — plain CPU build, no optional features (v0.1.2: `flash_attention` is opt-in, not part of `default`) |
| `flash_attention` | Memory-efficient O(N) attention, opt-in — only above the score-matrix budget (see below) |
| `mixed_precision` | FP32↔BF16/FP16 conversion utilities; not yet wired into the inference path (see below) |
| `gpu_debug` | NaN/Inf debug hooks (`debug_hooks::assert_finite`, `DebugConfig`) |
| `parallel` | Data-parallel CPU work via `rayon` (`dep:rayon`) — currently drives the per-sample ControlNet feature injection in `unet.rs`. Purely a performance switch: every parallelised routine is arithmetic-identical to its serial form |

### Feature Details

- **`flash_attention`**: Block-based attention computation. **Opt-in as of
  v0.1.2** — not part of `default`; add
  `features = ["flash_attention"]` to pull it in.
  - `FlashAttention::forward` only runs the tiled O(N) kernel when the
    un-tiled `(batch, heads, seq_q, seq_k)` score matrix would exceed the
    64 MiB budget (`flash_attention::DEFAULT_SCORE_MATRIX_BUDGET`,
    overridable via `FlashAttention::with_score_matrix_budget`). Below that
    budget it transparently falls back to the same full-materialization
    kernel standard attention uses, so **small sequences see no memory
    benefit** from turning the feature on.
  - Above the budget: reduces memory usage by 2-4× for large images while
    maintaining quality
  - `DiffusionConfig::use_flash_attention` still defaults to `true` *when
    this feature is compiled in* (see `config.rs`) — the feature itself is
    what is no longer default

- **`mixed_precision`**: FP32↔BF16/FP16 conversion and precision-loss
  simulation utilities (`mixed_precision.rs`) — real, tested code, not a
  stub. The flag only changes the default `MixedPrecisionConfig::mode`
  (`BFloat16` with the feature, `Float32` without). As of this writing,
  nothing in `unet.rs`/`vae.rs`/`pipeline.rs` calls into this module, so
  enabling the feature does **not** change what `generate()` computes — see
  the "Reachability" section of `mixed_precision.rs`'s module docs for the
  precise wiring that would be needed.

- **`gpu_debug`**: Turns on NaN/Inf assertions (`debug_hooks::assert_finite`,
  configured via `DebugConfig`) during the diffusion forward pass, useful
  for diagnosing numerical instability

- **`parallel`**: Enables `rayon`-backed data parallelism for CPU work.
  Currently wired into `unet.rs`'s per-sample ControlNet feature injection
  (the batch dimension is embarrassingly parallel: each sample reads only
  its own slice plus a shared, immutable condition cache). Every
  parallelised routine is arithmetic-identical to its serial form, so
  enabling this feature never changes results, only throughput.

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

    // Configure diffusion pipeline. DiffusionConfig has 30 fields covering
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
`cameras_to_tensor` (`src/diffusion_target.rs`) does. Neither `oxigaf-flame`
nor `nalgebra` is a dependency of `oxigaf-diffusion` itself (verified via
`cargo tree -p oxigaf-diffusion --all-features`: neither appears) — add both
as direct dependencies in your own `Cargo.toml` to build camera poses this
way, the same way `oxigaf-trainer` already does (it depends on
`oxigaf-diffusion`, `oxigaf-flame`, and `nalgebra` side by side):

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
use oxigaf_diffusion::{DdimScheduler, DiffusionError, PredictionType};

fn main() -> Result<(), DiffusionError> {
    // Create a DDIM scheduler (SD-2.1 defaults: scaled-linear beta schedule,
    // 1000 training timesteps). `new` is infallible — it only builds the
    // alpha-cumprod table; `set_timesteps` picks the inference stride and is
    // fallible (errors on 0 steps), so its `Result` needs handling.
    let mut scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);
    scheduler.set_timesteps(50)?;

    let timesteps = scheduler.timesteps();
    println!("Using {} inference steps", timesteps.len());
    println!("Timesteps: {:?}", timesteps);
    Ok(())
}
```

### Memory-Efficient Inference with Flash Attention

Requires the `flash_attention` feature (opt-in as of v0.1.2 — see
"Features" above):

```toml
oxigaf-diffusion = { version = "0.1", features = ["flash_attention"] }
```

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

## API Overview

The crate has grown to 72 source files (~65,000 lines of code) across the
areas below. `pipeline.rs` + `config.rs` are the integration surface most
callers need; everything else is available for advanced use (custom
samplers, training-time distillation losses, latent-space analysis tooling).

| Area | Key types / entry points | Modules |
|------|---------------------------|---------|
| Core pipeline | `MultiViewDiffusionPipeline`, `SessionRequest`, `MultiViewOutput`, `DiffusionConfig`, `AttentionBackend` (Standard/Flash/Sliced, config-selectable) | `pipeline.rs`, `config.rs` |
| Schedulers & samplers | `DdimScheduler`, DDPM/adaptive/consistency-model/flow-matching/rectified-flow/stochastic-interpolant samplers, `MultiStepSampler` (`dpm_plus_plus_2m_step`, `plms_step`) | `scheduler.rs`, `ddpm_sampler.rs`, `adaptive_sampling.rs`, `consistency_model.rs`, `flow_matching.rs`, `rectified_flow.rs`, `stochastic_interpolant.rs`, `multi_step_sampler/`, `step_scheduler.rs` |
| Attention | `FlashAttention` (feature-gated), `SlicedAttention`, `fused_attention`, `attention_masking`, `attention_viz`, `KVCache`, ToMe `token_merging` | `attention.rs`, `flash_attention.rs`, `sliced_attention.rs`, `fused_attention.rs`, `attention_masking.rs`, `attention_viz.rs`, `kv_cache.rs`, `token_merging.rs` |
| Conditioning & guidance | CFG, classifier guidance, identity/avatar/text conditioning, `ControlNet`, `LoRA`, prompt weighting/scheduling, unconditional dropout | `cfg_guidance.rs`, `classifier_guidance.rs`, `identity_conditioning.rs`, `avatar_conditioning.rs`, `text_conditioning.rs`, `controlnet.rs`, `lora_adapter.rs`, `guidance_rescaling.rs`, `prompt_weighting.rs`, `prompt_scheduler.rs`, `uncond_dropout.rs` |
| Image editing & inversion | SDEdit (two overlapping APIs sharing `ImageEditingError`), DDIM inversion (`InversionTrajectory`), image variations/preprocessing, style transfer | `image_editing/` (`mod.rs`, `sdedit.rs`), `inversion.rs`, `image_variations.rs`, `image_preprocessing.rs`, `style_transfer.rs` |
| Distillation & training support | Score distillation (SDS), `compute_distillation_loss` (CM/LCM-style), score matching, prior preservation | `distillation_loss.rs`, `score_distillation.rs`, `score_matching.rs`, `prior_preservation.rs` |
| Latent-space analysis | Latent blend/interp/walk/PCA analysis, denoising trajectory/visualisation, noise-schedule/-prediction analysis, view-consistency checks | `latent_blend.rs`, `latent_interp.rs`, `latent_walk.rs`, `latent_space_analysis.rs`, `denoising_trajectory.rs`, `denoising_viz.rs`, `noise_schedule_analysis.rs`, `noise_prediction_analysis.rs`, `multi_view_consistency.rs`, `cross_frame_consistency.rs` |
| Performance & serving | Mixed precision (not yet wired — see "Features"), `numerics` (stable softmax/layer-norm), `quantization`, `weight_offload`, `sequential_vae`, `streaming`, `batch_gen`, `dynamic_views`, `resolution`, `model_variants`, `profiling`, `debug_hooks` | `mixed_precision.rs`, `numerics.rs`, `quantization.rs`, `weight_offload.rs`, `sequential_vae.rs`, `streaming.rs`, `batch_gen.rs`, `dynamic_views.rs`, `resolution.rs`, `model_variants.rs`, `profiling.rs`, `debug_hooks.rs` |

## Pipeline Components

### CLIP Image Encoder

Extracts semantic features from input images using CLIP ViT-H/14 (Vision Transformer):

- Input: RGB image (224×224)
- Tower hidden width: `DiffusionConfig::clip_embed_dim` (1280 by default).
  `clip::build_clip_encoder` sizes the tower from this field via
  `ClipVisionConfig::vit_h14_with_embed_dim`, so it must be a positive multiple
  of 80 (ViT-H/14's head width) — `DiffusionConfig::validate` checks that.
- Output: per-patch tokens `DiffusionConfig::ip_adapter_context_dim()` wide
  (1024 by default), after IP-Adapter's projection. This is the *projection's*
  width, not the tower's, and is the width the U-Net's `attn_ip`
  cross-attention is built for — the two are deliberately one knob.
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
- Deterministic sampling for reproducibility — holds for the base
  256×256 pipeline (`pipeline.rs` seeds the initial latents through a
  dependency-free xorshift64 + Box-Muller stream specifically *because*
  `candle`'s `Tensor::randn` cannot be seeded on CPU) and for
  `upsampler_mode: Some(UpsamplerMode::BilinearVae)`. It does **not** hold
  for `Some(UpsamplerMode::SdX2)`: `LatentUpsampler::upsample_sdx2` draws its
  denoising noise with a plain, unseeded `Tensor::randn` call, so two
  `generate()` runs with the same `seed` at 512×512 diverge once the
  upsampler stage starts (see `upsampler.rs`)
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

These reductions only materialize once the un-tiled `(batch, heads, seq_q,
seq_k)` score matrix at a given attention level exceeds the 64 MiB budget
(`DEFAULT_SCORE_MATRIX_BUDGET` in `flash_attention.rs`); whether a
configuration crosses it depends on view count, head count, and sequence
length at *that* level — the deeper, lower-resolution U-Net stages stay well
below it even when the overall run targets 512×512 or 1024×1024 output.
Below the budget, `FlashAttention::forward` transparently runs the same
full-materialization kernel as standard attention, so those levels see no
memory reduction from turning the `flash_attention` feature on regardless of
output resolution.

## Statistics

- **Tests**: 3213 unit/integration tests (`cargo nextest run --all-features`,
  all passing: 3075 in the library target + 138 across 8 integration-test
  files) + 51 passing doc tests (`cargo test --doc --all-features`, 1
  ignored)
- **Source files**: 72 (~65,000 lines of code) — see [API Overview](#api-overview) above for the grouping
- **Benchmarks**: `diffusion_bench.rs` (requires the `flash_attention` feature), `flash_attention_bench.rs`

## Documentation

- [API Documentation](https://docs.rs/oxigaf-diffusion)
- [Repository](https://github.com/cool-japan/oxigaf)
- [Crate](https://crates.io/crates/oxigaf-diffusion)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE))
