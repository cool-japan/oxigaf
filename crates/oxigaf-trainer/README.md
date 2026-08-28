# oxigaf-trainer

GAF optimization pipeline — iterative denoising distillation.

**Status: Alpha (0.1.2, 2026-08-28).** Pre-1.0: this release breaks
struct-literal construction of `TrainingConfig`, `LossConfig`, and
`ViewConsistencyLoss` (each gained new fields) and changes several
function signatures and semantics (`detect_convergence_phase`,
`GaussianOptimizer::step`/`step_accumulated`, the `mr_*` functions, and
more). See "Migration Notes" below and `CHANGELOG.md`'s `[0.1.2]` entry for
the full list.

## Overview

This crate provides the training infrastructure for optimizing 3D Gaussian avatars:

- **Gaussian initialization** on FLAME mesh surfaces
- **Per-parameter Adam optimizer** with group-wise learning rates, plus
  opt-in **learning-rate schedules** (6 variants), **gradient clipping**
  (4 modes), **gradient accumulation**, and **EMA** shadow weights — all
  built by `Trainer::new` and consumed inside `train_step` when configured,
  no-op by default
- **Photometric + structural losses** (L1, SSIM, MS-SSIM, LPIPS) whose
  configured weights now actually shape the optimized gradient, not just
  the logged value (see "Migration Notes")
- **Position/scale/opacity regularization** with real gradients as of this
  release (see "Migration Notes" for the one exception)
- **Adaptive density control** (split, clone, prune), plus standalone
  **pruning utilities** (prune-to-sparsity, min-scale pruning, mask
  application)
- **Checkpoint management** (save/load with metadata)
- **Metric tracking** (PSNR, SSIM history)
- **Diffusion target generation** for iterative denoising distillation,
  including a real **score-distillation (SDS) gradient path** (see
  "Migration Notes")
- **Meta-learning avatar model** (`GaussianAvatarModel`) — a real
  `MetaModel` implementation trained through the actual rasterizer
  forward/backward pass
- **TensorBoard logging** for training visualization

## Migration Notes (0.1.1 → 0.1.2)

**If you trained with 0.1.1 using anything other than default `LossConfig`
weights, retrain — the objective actually being optimized has changed:**

- `Trainer::compute_gradients` optimized a **hardcoded L2 photometric
  loss** regardless of `LossConfig`. `w_l1`, `w_ssim`, and `w_ms_ssim` only
  ever changed the *logged* loss value, never the direction the optimizer
  descended. Fixed via the new `image_gradient::photometric_pixel_gradient`,
  built from the installed `LossComputer`'s actual config, SSIM window, and
  MS-SSIM scale weights.
- **Position/scale/opacity regularization** (`w_position_reg`,
  `w_scale_reg`, `w_opacity_reg`) were computed for logging only —
  `Gradients::offset` was never written by anything, so a Gaussian's
  `local_offsets` never moved during training no matter how
  `w_position_reg` was set. Fixed via the new (private)
  `add_regularization_gradients`. **`w_normal` and `w_gradient_penalty`
  remain logging-only** — no FLAME mesh is retained by the trainer and no
  external gradient buffer is threaded through for either term; this is a
  documented limitation, not a silent gap.
- **Score-distillation (SDS) training** (`use_sds`) had no real gradient
  path at all — it contributed nothing beyond the ordinary photometric
  term. Fixed via `DiffusionTargetGenerator::compute_sds_gradient` /
  `compute_sds_gradient_with_horizon`, and a validated
  `SdsLoss::new(weighting, max_timestep)` constructor (rejects
  `max_timestep < 2`, which previously divided by zero).

See `CHANGELOG.md`'s `[0.1.2]` entry for the complete list of signature and
semantics changes.

## Installation

```toml
[dependencies]
oxigaf-trainer = "0.1"
```

## Usage

The examples below use `oxigaf-render` (for `RasterConfig`), `oxigaf-flame`
(for FLAME mesh loading), `wgpu`, `rand`, and `pollster = "1"` directly —
add them to your own `[dependencies]` to run the snippets as-is (any async
executor works in place of `pollster`).

### Basic Training Loop

```rust
use oxigaf_render::RasterConfig;
use oxigaf_trainer::init::GaussianInitializer;
use oxigaf_trainer::{LossConfig, OptimizerConfig, Trainer, TrainingConfig};
use rand::SeedableRng;

fn main() -> Result<(), oxigaf_trainer::TrainerError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), oxigaf_trainer::TrainerError> {
    // Configure training. `TrainingConfig` embeds sub-configs for the
    // optimizer, loss, density control, and Gaussian initialization.
    let config = TrainingConfig {
        total_iterations: 1000,
        checkpoint_interval: 100,
        optimizer: OptimizerConfig::default(),
        loss: LossConfig::default(),
        ..Default::default()
    };
    config.validate()?;

    // Load a FLAME model and generate the neutral-expression mesh.
    let flame_model = oxigaf_flame::FlameModel::load("path/to/flame/model")?;
    let mesh = flame_model.forward(&oxigaf_flame::FlameParams::neutral());

    // Sample initial Gaussians on the mesh surface (rigid/flexible split,
    // counts and SH degree come from `config.init`). This returns a `Result`:
    // a face-less mesh, malformed topology, or a mesh too small to place the
    // requested number of Gaussians on is reported as an error rather than
    // silently yielding an empty model. To plug in a different sampling
    // strategy, use `initialize_with_sampler` with any `&dyn
    // oxigaf_flame::MeshSurfaceSampler`.
    let mut init_rng = rand::rngs::StdRng::seed_from_u64(42);
    let model = GaussianInitializer::initialize(&mesh, &config.init, &mut init_rng)?;

    // `Trainer::new` needs an already-created wgpu device/queue (device
    // creation is async); request one from the default adapter.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| oxigaf_trainer::TrainerError::Init(format!("No suitable GPU adapter: {e}")))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("oxigaf_trainer"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| oxigaf_trainer::TrainerError::Init(e.to_string()))?;

    let raster_config = RasterConfig::new()
        .with_sh_degree(config.init.sh_degree)
        .with_resolution(512, 512);

    // `config` is cloned so the local binding below (`config.total_iterations`
    // etc.) stays usable after `Trainer::new` takes ownership of its copy.
    let seed = 42u64;
    let mut trainer = Trainer::new(config.clone(), model, raster_config, device, queue, seed)?;

    // Training loop. Each `train_step()` samples camera views, renders the
    // current model, generates targets (diffusion once loaded, or a
    // self-supervised fallback during warmup), and runs the full
    // forward/backward/optimizer/density-control pipeline — there is no
    // separate "supply your own target images" step.
    for _ in 0..config.total_iterations {
        let output = trainer.train_step()?;

        if output.iteration % config.log_interval == 0 {
            println!(
                "Iteration {}: Loss={:.4}, {} Gaussians",
                output.iteration, output.loss.total, output.num_gaussians
            );
        }

        if output.iteration % config.checkpoint_interval == 0 {
            let path = format!("checkpoint_{}.json", output.iteration);
            trainer.save_checkpoint(std::path::Path::new(&path))?;
        }
    }

    // Save final model.
    trainer.save_checkpoint(std::path::Path::new("final_model.json"))?;

    Ok(())
}
```

### Custom Optimizer Configuration

```rust
use oxigaf_trainer::{OptimizerConfig, TrainingConfig};

fn main() -> Result<(), oxigaf_trainer::TrainerError> {
    // Configure per-parameter learning rates.
    let optimizer_config = OptimizerConfig {
        lr_position: 1.6e-4,       // Low for stability
        lr_position_final: 1.6e-6, // Target after exponential decay
        lr_rotation: 1e-3,         // Moderate for rotations
        lr_scale: 5e-3,            // Higher for scales
        lr_opacity: 5e-2,          // Highest for opacities
        lr_sh: 2.5e-3,             // Moderate for colors
        lr_offset: 1e-4,           // FLAME mesh-surface local offsets
        beta1: 0.9,                // Adam momentum
        beta2: 0.999,              // Adam RMS decay
        epsilon: 1e-15,            // Numerical stability
        position_lr_decay_steps: 30_000,
    };

    let config = TrainingConfig {
        optimizer: optimizer_config,
        ..Default::default()
    };
    config.validate()?;

    println!(
        "Trainer configured with custom learning rates: lr_position={}, lr_opacity={}",
        config.optimizer.lr_position, config.optimizer.lr_opacity
    );

    // Pass `config` (plus a model, raster config, GPU device/queue, and a
    // seed) to `Trainer::new` — see "Basic Training Loop" above.

    Ok(())
}
```

### Loss Function Configuration

```rust
use oxigaf_trainer::{LossConfig, TrainingConfig};

fn main() -> Result<(), oxigaf_trainer::TrainerError> {
    // Configure loss function weights.
    let loss_config = LossConfig {
        w_l1: 0.8,             // Photometric L1 loss
        w_ssim: 0.2,           // Structural similarity (used as 1 - SSIM)
        w_ms_ssim: 0.0,        // Multi-scale SSIM (optional)
        w_lpips: 0.05,         // Perceptual loss — see "LPIPS Perceptual Loss" below
        w_position_reg: 0.01,  // Penalize offset from the mesh surface
        w_scale_reg: 0.01,     // Penalize extreme scales
        w_opacity_reg: 0.001,  // Encourage binary opacity
        // Normal consistency. Logging-only: `Trainer::train_step` computes
        // this value but does not differentiate it into a gradient (no
        // FLAME mesh is retained by the trainer to compute it against).
        w_normal: 0.05,
        // Gradient penalty (training stability). Also logging-only, for the
        // same reason — no external gradient buffer is threaded through.
        w_gradient_penalty: 0.0,
        gradient_penalty_threshold: 100.0,
        // World-space scale (post-`exp()`) above which `w_scale_reg` starts
        // to charge; defaults to `oxigaf_trainer::loss::MAX_REASONABLE_WORLD_SCALE`.
        w_scale_reg_max_scale: oxigaf_trainer::loss::MAX_REASONABLE_WORLD_SCALE,
    };

    let config = TrainingConfig {
        loss: loss_config,
        ..Default::default()
    };
    config.validate()?;

    println!(
        "Using L1={}, SSIM={}, LPIPS={}",
        config.loss.w_l1, config.loss.w_ssim, config.loss.w_lpips
    );

    Ok(())
}
```

### Adaptive Density Control

```rust
use oxigaf_trainer::{DensityConfig, Trainer, TrainingConfig};

fn main() -> Result<(), oxigaf_trainer::TrainerError> {
    // Configure adaptive density control.
    let density_config = DensityConfig {
        grad_threshold: 0.0002,       // Split/clone if mean gradient exceeds this
        min_opacity: 0.005,           // Prune Gaussians below this opacity
        max_screen_size: 20.0,        // Prune Gaussians above this screen extent (px)
        split_scale_threshold: 0.01,  // Above -> split, below -> clone
        max_gaussians: 500_000,       // Hard cap on total Gaussian count
    };

    let config = TrainingConfig {
        density: density_config,
        density_control_start: 100,    // Start densification at iteration 100
        density_control_end: 500,      // Stop densification at iteration 500
        density_control_interval: 100, // Run every 100 iterations
        opacity_reset_interval: 300,   // Reset opacities every 300 iterations
        ..Default::default()
    };
    config.validate()?;

    let mut trainer = build_trainer(config.clone()); // see "Basic Training Loop"
    run_training(&mut trainer, &config)
}

/// Training loop with automatic densification — `train_step()` runs split /
/// clone / prune internally according to `config.density*`; no separate
/// call is needed.
fn run_training(
    trainer: &mut Trainer,
    config: &TrainingConfig,
) -> Result<(), oxigaf_trainer::TrainerError> {
    for _ in 0..config.total_iterations {
        trainer.train_step()?;

        if trainer.iteration % 100 == 0 {
            println!(
                "Iteration {}: {} Gaussians",
                trainer.iteration,
                trainer.model.len()
            );
        }
    }
    Ok(())
}

fn build_trainer(config: TrainingConfig) -> Trainer {
    unimplemented!("construct via Trainer::new(config, model, raster_config, device, queue, seed)")
}
```

### Checkpoint Management

```rust
use oxigaf_trainer::{Trainer, TrainingConfig};
use std::path::Path;

fn main() -> Result<(), oxigaf_trainer::TrainerError> {
    let config = TrainingConfig::default();
    let mut trainer = build_trainer(config.clone()); // see "Basic Training Loop"

    // Training loop.
    for _ in 0..config.total_iterations {
        let output = trainer.train_step()?;

        // Save a checkpoint every 100 iterations.
        if output.iteration % 100 == 0 {
            let checkpoint_path = format!("checkpoints/iter_{}.json", output.iteration);
            trainer.save_checkpoint(Path::new(&checkpoint_path))?;
            println!("Saved checkpoint: {checkpoint_path}");
        }
    }

    Ok(())
}

fn build_trainer(config: TrainingConfig) -> Trainer {
    unimplemented!("construct via Trainer::new(config, model, raster_config, device, queue, seed)")
}
```

Resuming loads a *new* trainer, which needs its own GPU device/queue (see
["Resuming from Checkpoint"](#resuming-from-checkpoint) below for the full
setup):

```rust
use oxigaf_render::RasterConfig;
use oxigaf_trainer::{Trainer, TrainingConfig};
use std::path::Path;

fn resume_example(
    config: TrainingConfig,
    checkpoint_path: &Path,
    raster_config: RasterConfig,
    device: wgpu::Device,
    queue: wgpu::Queue,
) -> Result<(), oxigaf_trainer::TrainerError> {
    let seed = 42u64;
    let trainer =
        Trainer::from_checkpoint(config, checkpoint_path, raster_config, device, queue, seed)?;
    println!("Resumed training from iteration {}", trainer.iteration);
    Ok(())
}
```

### Checkpoint Resume

Training can be interrupted and resumed seamlessly from checkpoints. The checkpoint system preserves all training state including model parameters, optimizer momentum, iteration counter, and metrics history.

#### Saving Checkpoints

Checkpoints should be saved periodically during training:

```rust
use oxigaf_trainer::Trainer;
use std::path::Path;

fn save_checkpoint_example(trainer: &Trainer, iteration: u32) -> Result<(), oxigaf_trainer::TrainerError> {
    // Save checkpoint with iteration number in filename
    let checkpoint_path = format!("checkpoints/training_{:06}.json", iteration);
    trainer.save_checkpoint(Path::new(&checkpoint_path))?;

    println!("Checkpoint saved at iteration {}", iteration);
    Ok(())
}
```

#### Resuming from Checkpoint

To resume training from a saved checkpoint:

```rust
use oxigaf_render::RasterConfig;
use oxigaf_trainer::{Trainer, TrainerError, TrainingConfig};
use std::path::Path;

async fn resume_training_example() -> Result<(), TrainerError> {
    let checkpoint_path = Path::new("checkpoints/training_001000.json");

    // Check if checkpoint exists
    if !checkpoint_path.exists() {
        return Err(TrainerError::CheckpointCorrupted(format!(
            "Checkpoint not found: {}",
            checkpoint_path.display()
        )));
    }

    // Load checkpoint first to inspect state
    let checkpoint = oxigaf_trainer::checkpoint::load_checkpoint(checkpoint_path)?;
    println!("Resuming from iteration {}", checkpoint.iteration);
    println!("Model has {} Gaussians", checkpoint.positions.len());

    // Set up the GPU (wgpu device creation is async).
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| TrainerError::Init(format!("No suitable GPU adapter: {e}")))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("oxigaf_trainer"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| TrainerError::Init(e.to_string()))?;

    // Create trainer from checkpoint
    let config = TrainingConfig::default();
    let raster_config = RasterConfig::default();
    let seed = 42u64;

    let mut trainer =
        Trainer::from_checkpoint(config, checkpoint_path, raster_config, device, queue, seed)?;

    // Continue training
    println!("Continuing from iteration {}", trainer.iteration);
    for _ in 0..1000 {
        trainer.train_step()?;
    }

    Ok(())
}
```

#### Best Practices

**Checkpoint Frequency**

- **Quick experiments**: Save every 50-100 iterations
- **Long training runs**: Save every 500-1000 iterations
- **Production training**: Save every 1000 iterations + on SIGTERM signal

**Storage Location**

- Use a dedicated `checkpoints/` directory
- Store on reliable storage (not /tmp for long-running jobs)
- Consider cloud backup for important training runs

**Naming Conventions**

```rust
// Iteration-based (recommended)
format!("checkpoint_{:06}.json", iteration)  // checkpoint_001000.json

// Timestamp-based (for parallel runs)
format!("checkpoint_{}_{}.json", run_id, timestamp)

// Latest + backup pattern
"checkpoint_latest.json"  // Always overwrite
"checkpoint_backup.json"  // Copy of previous latest
```

**When to Resume vs Restart**

- **Resume**: Normal interruption, want to continue exactly where you left off
- **Restart**: Poor convergence, want to try different hyperparameters
- **Partial resume**: Load model but reset optimizer (fine-tuning scenario)

#### Checkpoint Contents

A checkpoint file contains the complete training state:

**Model Parameters** (positions, rotations, scales, opacities, SH coefficients):
- Gaussian 3D positions and attributes
- Face indices and barycentric coordinates
- Local offsets and rigidity flags
- SH degree and coefficients

**Optimizer State** (Adam momentum):
- First moment estimates (m)
- Second moment estimates (v)
- Timestep counter (t)
- Per-parameter group states

**Training Progress**:
- Current iteration number
- Metrics history (PSNR, SSIM, loss values)
- Checkpoint format version

**File Format**:
- Format: JSON (human-readable, easy to debug)
- Alternative: Safetensors (planned for large models)
- Size: Typically 5-50 MB depending on number of Gaussians

**Size Estimates**:
- 1,000 Gaussians: ~2 MB
- 10,000 Gaussians: ~20 MB
- 100,000 Gaussians: ~200 MB

#### Troubleshooting

**Corrupted Checkpoint**

If a checkpoint is corrupted (power loss during save), use the fallback mechanism:

```rust
use oxigaf_trainer::checkpoint::try_load_checkpoint_with_fallback;
use std::path::Path;

fn load_with_fallback() -> Result<(), oxigaf_trainer::TrainerError> {
    let checkpoint_path = Path::new("checkpoints/training_001000.json");

    // Tries primary, then falls back to *.backup.json
    let checkpoint = try_load_checkpoint_with_fallback(checkpoint_path)?;

    println!("Successfully loaded checkpoint from iteration {}", checkpoint.iteration);
    Ok(())
}
```

**Version Mismatch**

If you see a version mismatch error:

```
TrainerError::CheckpointVersionMismatch { found: 2, expected: 1 }
```

This means the checkpoint was created with a newer version of oxigaf-trainer. Solutions:
- Update to the latest oxigaf-trainer version
- If downgrading is necessary, you'll need to convert the checkpoint manually
- Check the changelog for breaking changes between versions

**Missing Checkpoint File**

```rust
use std::path::Path;

fn check_checkpoint_exists(path: &Path) -> bool {
    if !path.exists() {
        eprintln!("Checkpoint not found: {}", path.display());
        eprintln!("Available checkpoints:");

        if let Some(dir) = path.parent() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                        println!("  - {}", entry.path().display());
                    }
                }
            }
        }
        return false;
    }
    true
}
```

**Partial Save Failures**

To implement atomic checkpoint saves:

```rust
use std::path::Path;
use std::fs;

fn atomic_save_checkpoint(
    trainer: &Trainer,
    checkpoint_path: &Path,
) -> Result<(), oxigaf_trainer::TrainerError> {
    // Save to temporary file first
    let temp_path = checkpoint_path.with_extension("json.tmp");
    trainer.save_checkpoint(&temp_path)?;

    // Create backup of existing checkpoint
    if checkpoint_path.exists() {
        let backup_path = checkpoint_path.with_extension("json.backup");
        fs::copy(checkpoint_path, &backup_path)?;
    }

    // Atomic rename
    fs::rename(&temp_path, checkpoint_path)?;

    Ok(())
}
```

#### Example Program

For a complete working example of checkpoint resume functionality, see:

```bash
cargo run --example checkpoint_resume -- --iterations-per-phase 50
```

This example demonstrates:
- Initial training with periodic checkpoint saving
- Automatic checkpoint detection
- Resume from checkpoint with state validation
- Metrics history preservation
- Iteration counter continuity

### Metric Tracking

`Trainer::train_step()` automatically records PSNR / SSIM / loss into
`trainer.metric_tracker` (a `metrics::MetricTracker`) after every step — no
manual bookkeeping is required:

```rust
use oxigaf_trainer::{Trainer, TrainingConfig};

fn main() -> Result<(), oxigaf_trainer::TrainerError> {
    let config = TrainingConfig::default();
    let mut trainer = build_trainer(config.clone()); // see "Basic Training Loop"

    for _ in 0..config.total_iterations {
        let output = trainer.train_step()?;

        if output.iteration % 10 == 0 {
            if let Some(latest) = trainer.metric_tracker.latest() {
                println!(
                    "Iteration {}: loss={:.4} PSNR={:.2} dB SSIM={:.4} ({} Gaussians)",
                    latest.iteration, latest.loss, latest.psnr, latest.ssim, output.num_gaussians,
                );
            }
        }
    }

    // Rolling statistics over the most recent window.
    println!(
        "\nFinal statistics (last 100 iterations):\n  Average PSNR: {:.2} dB\n  Average SSIM: {:.4}",
        trainer.metric_tracker.mean_psnr(100),
        trainer.metric_tracker.mean_ssim(100),
    );

    Ok(())
}

fn build_trainer(config: TrainingConfig) -> Trainer {
    unimplemented!("construct via Trainer::new(config, model, raster_config, device, queue, seed)")
}
```

### TensorBoard Logging

TensorBoard logging is configured through `TrainingConfig.tensorboard`
(`TensorBoardConfig`), not through a separate logger object. Once enabled,
`Trainer::train_step()` writes scalars every step and images/histograms at
the configured intervals automatically — no manual `log_scalar` calls are
needed:

```rust
use oxigaf_trainer::{TensorBoardConfig, Trainer, TrainingConfig};

fn main() -> Result<(), oxigaf_trainer::TrainerError> {
    let config = TrainingConfig {
        tensorboard: TensorBoardConfig::new("runs/experiment_1").with_run_name("run1"),
        ..Default::default()
    };
    config.validate()?;

    let mut trainer = build_trainer(config.clone()); // see "Basic Training Loop"

    for _ in 0..config.total_iterations {
        trainer.train_step()?;
    }

    println!("View training progress with: tensorboard --logdir runs/");

    Ok(())
}

fn build_trainer(config: TrainingConfig) -> Trainer {
    unimplemented!("construct via Trainer::new(config, model, raster_config, device, queue, seed)")
}
```

### LPIPS Perceptual Loss

**Caveat**: `LossConfig.w_lpips` alone does not enable LPIPS during
`Trainer::train_step()`. The trainer's built-in loss path
(`LossComputer::compute`) always evaluates the LPIPS term as `0.0`, because
LPIPS needs a loaded VGG network that the trainer does not manage
automatically. To use LPIPS you drive your own render/loss loop with
`LossComputer::compute_with_lpips` instead of calling `train_step()`:

```rust
use oxigaf_trainer::loss::LossComputer;
use oxigaf_trainer::{LossConfig, LpipsLossComputer, TrainingConfig};
use std::path::Path;

fn main() -> Result<(), oxigaf_trainer::TrainerError> {
    let loss_config = LossConfig {
        w_l1: 0.75,
        w_ssim: 0.2,
        w_lpips: 0.05, // Small weight for perceptual loss
        ..Default::default()
    };
    let config = TrainingConfig {
        loss: loss_config,
        ..Default::default()
    };
    config.validate()?;

    // Load pre-trained LPIPS VGG + linear weights (safetensors format).
    let mut lpips = LpipsLossComputer::new();
    lpips.init(
        Path::new("path/to/vgg_weights.safetensors"),
        Path::new("path/to/lpips_weights.safetensors"),
    )?;
    let loss_computer = LossComputer::new(config.loss.clone());

    println!(
        "LPIPS ready: {} (weight={})",
        lpips.is_initialized(),
        config.loss.w_lpips
    );

    // Inside your own render loop, per batch of rendered/target view pairs:
    //   let lpips_value = lpips.compute_multi(&rendered, &targets, width, height)?;
    //   let loss_output = loss_computer.compute_with_lpips(
    //       &rendered, &targets, width, height, &model, mesh, lpips_value,
    //   );
    //   println!("L1={:.4} SSIM={:.4} LPIPS={:.4}", loss_output.l1, loss_output.ssim, loss_output.lpips);

    Ok(())
}
```

## Training Configuration

### Default Hyperparameters

```rust
// The literal `Default::default()` values (see `src/config.rs`).
TrainingConfig {
    total_iterations: 15_000,
    views_per_step: 4,
    density_control_interval: 500,
    density_control_start: 1_000,
    density_control_end: 12_000,
    opacity_reset_interval: 3_000,
    checkpoint_interval: 1_000,
    log_interval: 50,
    guidance_scale_start: 7.5,
    guidance_scale_end: 3.0,
    guidance_anneal_steps: 10_000,

    // Optimizer
    optimizer: OptimizerConfig {
        lr_position: 1.6e-4,
        lr_position_final: 1.6e-6,
        lr_rotation: 1e-3,
        lr_scale: 5e-3,
        lr_opacity: 5e-2,
        lr_sh: 2.5e-3,
        lr_offset: 1e-4,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-15,
        position_lr_decay_steps: 30_000,
    },

    // Loss
    loss: LossConfig {
        w_l1: 0.8,
        w_ssim: 0.2,
        w_ms_ssim: 0.0,
        w_lpips: 0.0,
        w_position_reg: 0.01,
        w_scale_reg: 0.01,
        w_opacity_reg: 0.001,
        w_normal: 0.05,
        w_gradient_penalty: 0.0,
        gradient_penalty_threshold: 100.0,
        w_scale_reg_max_scale: oxigaf_trainer::loss::MAX_REASONABLE_WORLD_SCALE,
    },

    // Density control
    density: DensityConfig {
        grad_threshold: 0.0002,
        min_opacity: 0.005,
        max_screen_size: 20.0,
        split_scale_threshold: 0.01,
        max_gaussians: 500_000,
    },

    // Initialization
    init: InitConfig {
        num_rigid: 50_000,
        num_flexible: 50_000,
        initial_scale: -5.0,
        initial_opacity: -2.0,
        sh_degree: 3,
    },

    // TensorBoard disabled, Float32 precision, profiling off by default.
    tensorboard: TensorBoardConfig::default(),
    precision: TrainingPrecision::Float32,
    enable_profiling: false,

    // Learning-rate schedule, gradient clipping, gradient accumulation, and
    // EMA shadow weights are built by `Trainer::new` and consumed inside
    // `train_step` whenever set away from these defaults — no separate
    // wiring needed, and a no-op out of the box.
    lr_schedule: LrScheduleConfig::Fixed,
    gradient_clip: GradientClipConfig::Disabled,
    gradient_accumulation_steps: 1,
    ema_decay: None,
}
```

## Performance

Training performance on various hardware (512×512 resolution, 10K Gaussians):

| Hardware | Time/Iteration | Memory Usage |
|----------|---------------|--------------|
| NVIDIA RTX 4090 | ~50 ms | ~4 GB |
| NVIDIA RTX 3080 | ~80 ms | ~5 GB |
| Apple M2 Max | ~120 ms | ~6 GB |

These figures predate the 0.1.2 photometric-gradient fix (per-view
SSIM/MS-SSIM pixel-gradient computation replacing a fixed `2·(rendered −
target)` term) and have not been re-measured against it; treat them as
directional rather than current benchmarks.

Typical training times:

- **Quick preview**: 500 iterations (~1 minute on RTX 4090)
- **Good quality**: 2,000 iterations (~3 minutes)
- **High quality**: 5,000 iterations (~7 minutes)

## Statistics

- **Tests**: 3,271 `#[test]`-attributed tests in source (`src/` + `tests/`); 3,259 pass and 12 are `#[ignore]`d (GPU/slow — LPIPS VGG inference, GPU resume/end-to-end) with `cargo nextest run -p oxigaf-trainer --all-features` — the exact count that compiles and runs depends on enabled features and platform
- **Key modules**: `trainer.rs` (1,921 lines), `loss.rs` (1,879 lines), `image_gradient.rs` (833 lines — the corrected photometric-gradient path), `tensorboard.rs` (1,307 lines), `checkpoint.rs` (1,028 lines), `density.rs` (638 lines), `lpips.rs` (805 lines)
- **LPIPS**: Pure Rust VGG network (no Python/C dependencies)
- **TensorBoard**: Native Rust implementation — scalar, image, histogram, graph logging

## Documentation

- [API Documentation](https://docs.rs/oxigaf-trainer)
- [Repository](https://github.com/cool-japan/oxigaf)
- [Crate](https://crates.io/crates/oxigaf-trainer)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE))
