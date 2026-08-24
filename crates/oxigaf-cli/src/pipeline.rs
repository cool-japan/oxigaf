//! End-to-end reconstruction pipeline orchestration.
//!
//! Wires together FLAME model loading, Gaussian initialisation, trainer
//! creation, the training loop (with progress reporting), and final export.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use nalgebra as na;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use oxigaf::flame::{Camera, FlameModel, FlameParams};
use oxigaf::render::gaussian::GaussianModel;
use oxigaf::trainer::init::GaussianInitializer;
use oxigaf::trainer::Trainer;

use crate::config::ProjectConfig;
use crate::progress;
use crate::verbosity::Verbosity;

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// All inputs needed to run the reconstruction pipeline.
///
/// Deliberately carries no `#[allow(dead_code)]`: every field is read by
/// [`run_reconstruction`], and the blanket suppression is what previously hid
/// `input_path` and `device_index` being silently ignored.
pub struct PipelineConfig {
    /// Path to the converted FLAME model directory.
    pub flame_model_path: PathBuf,
    /// Optional pre-computed per-frame FLAME tracking parameters (JSON).
    pub flame_params_path: Option<PathBuf>,
    /// Frame directory (or single image) holding the subject's footage.
    ///
    /// Video containers are rejected — see [`collect_input_frames`].
    pub input_path: PathBuf,
    /// Output directory for checkpoints, exports, and logs.
    pub output_dir: PathBuf,
    /// Optional checkpoint to resume from.
    pub resume_checkpoint: Option<PathBuf>,
    /// GPU device index.
    pub device_index: usize,
    /// Parsed project configuration.
    pub project_config: ProjectConfig,
    /// Early stopping patience (iterations without improvement).
    pub patience: Option<u32>,
    /// Minimum delta for improvement (default: 1e-4).
    pub min_delta: Option<f32>,
    /// Optional metrics output file path.
    pub metrics_output: Option<PathBuf>,
    /// Metrics output format (CSV or JSON Lines).
    pub metrics_format: crate::cli::MetricsOutputFormat,
    /// Enable TensorBoard logging.
    pub tensorboard: bool,
    /// TensorBoard log directory.
    pub tensorboard_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Camera specification (for JSON trajectory files)
// ---------------------------------------------------------------------------

/// Lightweight camera description for JSON-based trajectory files.
///
/// The camera looks at the origin from a position specified by spherical
/// coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSpec {
    /// Azimuth angle in degrees (0 = front, 90 = right).
    pub azimuth: f32,
    /// Elevation angle in degrees (0 = horizon, 90 = top).
    pub elevation: f32,
    /// Distance from the origin.
    #[serde(default = "default_distance")]
    pub distance: f32,
}

fn default_distance() -> f32 {
    0.6
}

// ---------------------------------------------------------------------------
// Input footage discovery
// ---------------------------------------------------------------------------

/// Still-image extensions accepted inside a frame directory (lower-cased).
const FRAME_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "exr"];

/// Video container extensions recognised but not decodable in pure Rust.
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "avi", "mkv", "webm", "m4v", "mpg", "mpeg"];

/// The footage referenced by `--input`, resolved to a concrete list of frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputFrames {
    /// Frame image paths, sorted by file name so the sequence is deterministic.
    pub paths: Vec<PathBuf>,
    /// Pixel dimensions of the first frame (`width`, `height`).
    pub width: u32,
    pub height: u32,
}

fn has_extension(path: &Path, allowed: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| allowed.contains(&e.as_str()))
}

/// Enumerate the frame images referenced by `--input`.
///
/// `input_path` may be either a single image, or a directory containing a
/// frame sequence.  Video containers are detected and rejected with an
/// actionable message: OxiGAF is pure Rust (COOLJAPAN policy) and therefore
/// ships no video demuxer/decoder, so callers must extract frames first.
///
/// # Errors
///
/// Returns an error when the path does not exist, names a video container,
/// names an unsupported file type, or is a directory with no decodable frames.
pub fn collect_input_frames(input_path: &Path) -> Result<Vec<PathBuf>> {
    if !input_path.exists() {
        anyhow::bail!("Input path does not exist: {}", input_path.display());
    }

    if input_path.is_file() {
        if has_extension(input_path, VIDEO_EXTENSIONS) {
            anyhow::bail!(
                "Video input is not supported: {}\n\
                 OxiGAF is pure Rust and bundles no video decoder. Extract the frames first \
                 (e.g. `ffmpeg -i {} frames/%05d.png`) and pass the frame directory to --input.",
                input_path.display(),
                input_path.display(),
            );
        }
        if !has_extension(input_path, FRAME_EXTENSIONS) {
            anyhow::bail!(
                "Unsupported input file type: {} (expected one of: {})",
                input_path.display(),
                FRAME_EXTENSIONS.join(", "),
            );
        }
        return Ok(vec![input_path.to_path_buf()]);
    }

    let read_dir = std::fs::read_dir(input_path)
        .with_context(|| format!("Failed to read input directory: {}", input_path.display()))?;

    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in read_dir {
        let entry =
            entry.with_context(|| format!("Failed to read entry in {}", input_path.display()))?;
        let path = entry.path();
        if path.is_file() && has_extension(&path, FRAME_EXTENSIONS) {
            paths.push(path);
        }
    }

    if paths.is_empty() {
        anyhow::bail!(
            "No decodable frames found in {} (looked for: {})",
            input_path.display(),
            FRAME_EXTENSIONS.join(", "),
        );
    }

    // Sort by file name so `frame_00001.png`, `frame_00002.png`, … stay ordered
    // independently of the order the filesystem hands entries back in.
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(paths)
}

/// Resolve `--input` into a validated frame sequence.
///
/// Beyond enumerating the files this actually opens the first frame's header,
/// so a directory full of unreadable or truncated images fails here rather
/// than silently producing a model trained on nothing.
fn load_input_frames(input_path: &Path) -> Result<InputFrames> {
    let paths = collect_input_frames(input_path)?;
    let first = paths
        .first()
        .ok_or_else(|| anyhow::anyhow!("Input frame list is empty: {}", input_path.display()))?;
    let (width, height) = image::image_dimensions(first)
        .with_context(|| format!("Failed to read image header: {}", first.display()))?;
    Ok(InputFrames {
        paths,
        width,
        height,
    })
}

// ---------------------------------------------------------------------------
// Training result metadata
// ---------------------------------------------------------------------------

/// Metadata collected during training for summary reporting.
pub struct TrainingResult {
    pub model: GaussianModel,
    pub final_loss: f32,
    #[allow(dead_code)]
    pub best_loss: f32,
    pub total_iterations: u32,
    pub num_rigid: usize,
    pub num_flexible: usize,
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run the full reconstruction pipeline and return the trained Gaussian model with metadata.
pub fn run_reconstruction(
    config: PipelineConfig,
    verbosity: Verbosity,
    interactive_controller: Option<&crate::interactive::InteractiveController>,
) -> Result<TrainingResult> {
    let start = std::time::Instant::now();

    // 0. Resolve and validate the user's footage.  This fails fast on a missing
    //    path, a video container (no pure-Rust decoder), or an empty/unreadable
    //    frame directory instead of quietly training on nothing.
    let frames = load_input_frames(&config.input_path)
        .with_context(|| format!("Invalid --input {}", config.input_path.display()))?;
    tracing::info!(
        "Input footage: {} frame(s) at {}×{} from {}",
        frames.paths.len(),
        frames.width,
        frames.height,
        config.input_path.display(),
    );
    // The trainer currently supervises itself from rendered views plus diffusion
    // targets; it exposes no hook for injecting observed frames, and nothing in
    // the workspace produces the per-frame camera poses such supervision needs
    // (see the followups on `oxigaf_trainer::Trainer` and FLAME tracking).
    // Say so loudly rather than pretending the footage was used.
    tracing::warn!(
        "The {} input frame(s) are validated but NOT yet used as photometric \
         targets: per-frame camera poses and a trainer dataset hook are required. \
         This run is self-supervised (rendered views + diffusion targets) and its \
         result does not depend on {}.",
        frames.paths.len(),
        config.input_path.display(),
    );

    // 1. Load FLAME model
    tracing::info!(
        "Loading FLAME model from {}",
        config.flame_model_path.display()
    );
    let flame = FlameModel::load(&config.flame_model_path).with_context(|| {
        format!(
            "Failed to load FLAME model from {}",
            config.flame_model_path.display()
        )
    })?;
    tracing::info!(
        "FLAME model loaded: {} vertices, {} faces",
        flame.num_vertices(),
        flame.v_template.nrows(),
    );

    // 2. Optionally load per-frame FLAME parameters
    let flame_params = if let Some(ref params_path) = config.flame_params_path {
        tracing::info!("Loading FLAME params from {}", params_path.display());
        let json = std::fs::read_to_string(params_path)
            .with_context(|| format!("Failed to read FLAME params: {}", params_path.display()))?;
        let params: Vec<FlameParams> = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse FLAME params: {}", params_path.display()))?;
        tracing::info!("Loaded {} frames of FLAME parameters", params.len());
        Some(params)
    } else {
        tracing::info!("No FLAME params provided — using neutral pose for initialisation");
        None
    };

    // 3. Compute rest-pose mesh for Gaussian initialisation
    let init_params = flame_params
        .as_ref()
        .and_then(|p| p.first())
        .cloned()
        .unwrap_or_else(FlameParams::neutral);
    let mesh = flame.forward(&init_params);
    tracing::info!(
        "Rest-pose mesh computed: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces(),
    );

    // 4. Build internal configs
    let training_config = config.project_config.to_training_config();
    let raster_config = config.project_config.to_raster_config();

    // 4b. Request GPU device for the rasterizer
    let (device, queue) = request_gpu_device(config.device_index).with_context(|| {
        format!(
            "Failed to initialise GPU device {} for rasterizer",
            config.device_index
        )
    })?;
    tracing::info!("GPU device {} acquired for rasterizer", config.device_index);

    // 5. Initialise or resume trainer
    let mut trainer = if let Some(ref ckpt_path) = config.resume_checkpoint {
        tracing::info!("Resuming training from checkpoint: {}", ckpt_path.display());
        Trainer::from_checkpoint(training_config, ckpt_path, raster_config, device, queue, 42)
            .context("Failed to restore trainer from checkpoint")?
    } else {
        tracing::info!("Initialising Gaussians on mesh surface…");
        let mut rng = StdRng::seed_from_u64(42);
        let model = GaussianInitializer::initialize(&mesh, &training_config.init, &mut rng);
        tracing::info!(
            "Initialised {} Gaussians ({} rigid, {} flexible)",
            model.len(),
            model.is_rigid.iter().filter(|&&r| r).count(),
            model.is_rigid.iter().filter(|&&r| !r).count(),
        );
        Trainer::new(training_config, model, raster_config, device, queue, 42)
            .context("Failed to create trainer")?
    };

    // 6. Prepare output directories
    let checkpoint_dir = config.output_dir.join("checkpoints");
    std::fs::create_dir_all(&checkpoint_dir).context("Failed to create checkpoint directory")?;

    // 7. Training loop with progress bar and early stopping
    let total = trainer.config.total_iterations;
    let ckpt_interval = trainer.config.checkpoint_interval;
    let log_interval = trainer.config.log_interval;
    let start_iter = trainer.iteration;
    let remaining = total.saturating_sub(start_iter);

    tracing::info!(
        "Starting training: {} → {} ({} remaining iterations)",
        start_iter,
        total,
        remaining,
    );

    // Early stopping state
    let min_delta = config.min_delta.unwrap_or(1e-4);
    let mut best_loss = f32::INFINITY;
    let mut final_loss = f32::INFINITY; // Track the last loss value
    let mut patience_counter = 0u32;
    let mut loss_history: Vec<f32> = Vec::with_capacity(50);

    // Initialize metrics writer if requested
    let mut metrics_writer = if let Some(ref path) = config.metrics_output {
        let format = match config.metrics_format {
            crate::cli::MetricsOutputFormat::Csv => crate::metrics::MetricsFormat::Csv,
            crate::cli::MetricsOutputFormat::Json => crate::metrics::MetricsFormat::JsonLines,
        };
        Some(
            crate::metrics::MetricsWriter::new(path, format)
                .context("Failed to create metrics writer")?,
        )
    } else {
        None
    };

    // Initialize TensorBoard logger if requested
    let mut tensorboard_writer = if config.tensorboard {
        use oxigaf_trainer::tensorboard::{TensorBoardConfig, TensorBoardWriter};

        let tb_config = TensorBoardConfig::new(&config.tensorboard_dir)
            .with_run_name(format!(
                "run_{}",
                chrono::Local::now().format("%Y%m%d_%H%M%S")
            ))
            .with_flush_interval(10);

        let writer =
            TensorBoardWriter::new(tb_config).context("Failed to create TensorBoard writer")?;
        tracing::info!("TensorBoard logging enabled: {:?}", writer.file_path());
        Some(writer)
    } else {
        None
    };

    let pb = progress::training_progress(remaining as u64, verbosity);

    // Print interactive controls if enabled
    if let Some(controller) = interactive_controller {
        controller.print_controls();
    }

    while trainer.iteration < total {
        // Check interactive controller state
        if let Some(controller) = interactive_controller {
            // Handle pause state
            while controller.paused.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
                // Check for quit during pause
                if controller
                    .quit_requested
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    break;
                }
            }

            // Check quit request
            if controller
                .quit_requested
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                pb.finish_with_message("interrupted by user");
                tracing::info!(
                    "Training interrupted by user at iteration {}",
                    trainer.iteration
                );
                break;
            }

            // Check save request
            if controller
                .save_requested
                .swap(false, std::sync::atomic::Ordering::Relaxed)
            {
                let path = checkpoint_dir.join(format!("manual_{:06}.json", trainer.iteration));
                match trainer.save_checkpoint(&path) {
                    Ok(()) => {
                        tracing::info!("Manual checkpoint saved to {}", path.display());
                    }
                    Err(e) => {
                        tracing::error!("Manual checkpoint save failed: {}", e);
                    }
                }
            }
        }

        let step = trainer.train_step().context("Training step failed")?;
        let current_loss = step.loss.total;
        final_loss = current_loss; // Track for summary

        // Update loss history for sparkline (keep last 50)
        loss_history.push(current_loss);
        if loss_history.len() > 50 {
            loss_history.remove(0);
        }

        // Generate ASCII sparkline
        let sparkline = generate_sparkline(&loss_history);

        // Check for improvement
        let improved = current_loss < (best_loss - min_delta);
        if improved {
            best_loss = current_loss;
            patience_counter = 0;
        } else if let Some(patience) = config.patience {
            patience_counter += 1;
            if patience_counter >= patience {
                pb.finish_with_message("early stopping triggered");
                tracing::info!(
                    "Early stopping at iteration {} (best loss: {:.6}, current: {:.6})",
                    step.iteration,
                    best_loss,
                    current_loss
                );
                break;
            }
        }

        pb.set_position((step.iteration - start_iter) as u64);
        pb.set_message(format!(
            "curr: {:.6} | best: {:.6} {} | patience: {}/{}",
            current_loss,
            best_loss,
            sparkline,
            patience_counter,
            config.patience.unwrap_or(0)
        ));

        // Periodic checkpointing
        if ckpt_interval > 0 && step.iteration % ckpt_interval == 0 {
            let path = checkpoint_dir.join(format!("ckpt_{:06}.json", step.iteration));
            trainer
                .save_checkpoint(&path)
                .with_context(|| format!("Checkpoint save failed at iter {}", step.iteration))?;
        }

        // Periodic logging
        if log_interval > 0 && step.iteration % log_interval == 0 {
            tracing::info!(
                iter = step.iteration,
                loss = %format!("{:.6}", current_loss),
                best_loss = %format!("{:.6}", best_loss),
                gaussians = step.num_gaussians,
                patience = format!("{}/{}", patience_counter, config.patience.unwrap_or(0)),
                "training",
            );
        }

        // Write metrics to file if enabled
        if let Some(ref mut writer) = metrics_writer {
            let elapsed = start.elapsed();
            let metrics = crate::metrics::TrainingMetrics {
                iteration: step.iteration,
                loss_total: current_loss,
                loss_l1: step.loss.l1,
                loss_ssim: step.loss.ssim,
                loss_lpips: Some(step.loss.lpips),
                loss_reg: step.loss.position_reg + step.loss.scale_reg + step.loss.opacity_reg,
                num_gaussians: step.num_gaussians as u32,
                lr_position: trainer.config.optimizer.lr_position,
                lr_scaling: trainer.config.optimizer.lr_scale,
                lr_rotation: trainer.config.optimizer.lr_rotation,
                memory_mb: None, // Could add GPU memory tracking here
                elapsed_seconds: elapsed.as_secs_f32(),
            };

            if let Err(e) = writer.write_metrics(&metrics) {
                tracing::warn!("Failed to write metrics: {}", e);
            }
        }

        // Log to TensorBoard if enabled
        if let Some(ref mut tb_writer) = tensorboard_writer {
            let step_i64 = step.iteration as i64;

            // Log loss components
            if let Err(e) = tb_writer.log_scalars(
                &[
                    ("loss/total", current_loss),
                    ("loss/l1", step.loss.l1),
                    ("loss/ssim", step.loss.ssim),
                    ("loss/lpips", step.loss.lpips),
                    ("loss/sds", step.sds_loss),
                    (
                        "loss/regularization",
                        step.loss.position_reg + step.loss.scale_reg + step.loss.opacity_reg,
                    ),
                ],
                step_i64,
            ) {
                tracing::warn!("Failed to log losses to TensorBoard: {}", e);
            }

            // Log model metrics
            if let Err(e) = tb_writer.log_scalars(
                &[
                    ("model/num_gaussians", step.num_gaussians as f32),
                    ("model/best_loss", best_loss),
                ],
                step_i64,
            ) {
                tracing::warn!("Failed to log model metrics to TensorBoard: {}", e);
            }

            // Log learning rates
            if let Err(e) = tb_writer.log_scalars(
                &[
                    ("lr/position", trainer.config.optimizer.lr_position),
                    ("lr/rotation", trainer.config.optimizer.lr_rotation),
                    ("lr/scale", trainer.config.optimizer.lr_scale),
                    ("lr/opacity", trainer.config.optimizer.lr_opacity),
                    ("lr/sh", trainer.config.optimizer.lr_sh),
                ],
                step_i64,
            ) {
                tracing::warn!("Failed to log learning rates to TensorBoard: {}", e);
            }
        }
    }

    if trainer.iteration >= total {
        pb.finish_with_message("training complete");
    }

    // 8. Save final checkpoint
    let final_ckpt = checkpoint_dir.join("final.json");
    trainer
        .save_checkpoint(&final_ckpt)
        .context("Failed to save final checkpoint")?;
    tracing::info!("Saved final checkpoint to {}", final_ckpt.display());

    let elapsed = start.elapsed();
    tracing::info!(
        "Training finished in {:.1}s — {} Gaussians",
        elapsed.as_secs_f64(),
        trainer.model.len(),
    );

    // Count rigid and flexible Gaussians
    let num_rigid = trainer.model.is_rigid.iter().filter(|&&r| r).count();
    let num_flexible = trainer.model.is_rigid.iter().filter(|&&r| !r).count();

    Ok(TrainingResult {
        model: trainer.model.clone(),
        final_loss,
        best_loss,
        total_iterations: trainer.iteration,
        num_rigid,
        num_flexible,
    })
}

// ---------------------------------------------------------------------------
// GPU device initialisation
// ---------------------------------------------------------------------------

/// Validate a `--device` index against the number of enumerated adapters.
///
/// Pure helper so the out-of-range branch is unit-testable without a GPU.
///
/// # Errors
///
/// Returns an error when no adapter is available at all, or when `requested`
/// is not a valid index into the adapter list.
fn select_adapter_index(num_adapters: usize, requested: usize) -> Result<usize> {
    if num_adapters == 0 {
        anyhow::bail!("No GPU adapters found (--device {requested} requested)");
    }
    if requested >= num_adapters {
        anyhow::bail!(
            "--device {requested} is out of range: {num_adapters} GPU adapter(s) available \
             (valid indices 0..={})",
            num_adapters - 1,
        );
    }
    Ok(requested)
}

/// Request a wgpu device and queue for GPU-accelerated rasterization.
///
/// `device_index` is the `--device` GPU index:
///
/// * `0` means "let wgpu pick the best adapter" — a
///   [`wgpu::PowerPreference::HighPerformance`] request.  Enumerating
///   unconditionally and taking `adapters[0]` would regress this, because on a
///   laptop the first *enumerated* adapter is usually the integrated GPU.
/// * any other index selects `enumerate_adapters()[index]` explicitly and
///   errors (listing the adapters) when the index is out of range.
///
/// A consequence of that asymmetry: `--device 0` and `--device N` can resolve
/// to the same physical GPU when the high-performance pick happens to be
/// adapter `N`.  The selected adapter name, backend, and `device_index` are
/// logged on both paths so the mapping is visible.
///
/// Uses `pollster` style blocking — safe to call from a tokio context because
/// `wgpu::Instance::request_adapter` / `Adapter::request_device` only block on
/// the GPU driver, not on tokio I/O.
fn request_gpu_device(device_index: usize) -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = if device_index == 0 {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| anyhow::anyhow!("No suitable GPU adapter found: {e}"))?
    } else {
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let listing: Vec<String> = adapters
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let info = a.get_info();
                format!("[{i}] {} ({:?})", info.name, info.backend)
            })
            .collect();
        let index = select_adapter_index(adapters.len(), device_index)
            .with_context(|| format!("Enumerated GPU adapters: {}", listing.join(", ")))?;
        adapters
            .into_iter()
            .nth(index)
            .ok_or_else(|| anyhow::anyhow!("GPU adapter {index} vanished during selection"))?
    };

    tracing::info!(
        adapter = adapter.get_info().name,
        backend = ?adapter.get_info().backend,
        device_index,
        "Selected GPU adapter"
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("oxigaf_pipeline"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::default(),
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| anyhow::anyhow!("GPU device creation failed: {e}"))?;

    Ok((device, queue))
}

// ---------------------------------------------------------------------------
// Camera utilities
// ---------------------------------------------------------------------------

/// Create a pinhole camera looking at the origin from spherical coordinates.
///
/// * `azimuth_deg`  — horizontal angle in degrees (0 = +Z, 90 = +X).
/// * `elevation_deg` — vertical angle in degrees (0 = horizon, +90 = top).
/// * `distance`      — distance from the origin.
pub fn orbit_camera(
    azimuth_deg: f32,
    elevation_deg: f32,
    distance: f32,
    width: u32,
    height: u32,
) -> Camera {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();

    let x = distance * el.cos() * az.sin();
    let y = distance * el.sin();
    let z = distance * el.cos() * az.cos();

    let eye = na::Vector3::new(x, y, z);
    let forward = (-eye).normalize();
    let world_up = na::Vector3::new(0.0, 1.0, 0.0);

    // Robust right vector (handle looking straight up/down).
    let right = if forward.cross(&world_up).norm() < 1e-6 {
        na::Vector3::new(1.0, 0.0, 0.0)
    } else {
        forward.cross(&world_up).normalize()
    };
    let up = right.cross(&forward);

    let rotation = na::Matrix3::from_columns(&[right, up, -forward]).transpose();
    let translation = -(rotation * eye);

    let focal = width as f32 * 1.5;

    Camera {
        rotation,
        translation,
        focal_x: focal,
        focal_y: focal,
        cx: width as f32 / 2.0,
        cy: height as f32 / 2.0,
        width,
        height,
        near: 0.01,
        far: 10.0,
    }
}

/// Generate a default set of 8 orbit cameras evenly spaced around the head.
pub fn default_orbit_cameras(width: u32, height: u32) -> Vec<Camera> {
    let azimuths = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];
    azimuths
        .iter()
        .map(|&az| orbit_camera(az, 10.0, 0.6, width, height))
        .collect()
}

// ---------------------------------------------------------------------------
// Sparkline visualization
// ---------------------------------------------------------------------------

/// Number of loss samples rendered in the progress-bar sparkline.
const SPARKLINE_WIDTH: usize = 20;

/// Generate an ASCII sparkline from a sequence of loss values.
///
/// Uses Unicode block characters to show the trend over the **most recent**
/// [`SPARKLINE_WIDTH`] samples of `values`.  The min/max normalisation is
/// computed over that same window, so the rendered glyphs always span the full
/// block range for the segment being shown.
fn generate_sparkline(values: &[f32]) -> String {
    if values.is_empty() {
        return String::new();
    }

    // Tail window: `values` is an append-only history, so the newest samples
    // live at the end.  (`take(N)` would render the *oldest* N and freeze the
    // graphic once the history exceeded the window.)
    let window = &values[values.len().saturating_sub(SPARKLINE_WIDTH)..];

    let min = window.iter().copied().fold(f32::INFINITY, f32::min);
    let max = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    if range < 1e-9 {
        // All values in the window are essentially the same
        return "▄".repeat(window.len());
    }

    let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let num_chars = chars.len();

    window
        .iter()
        .map(|&v| {
            let normalized = (v - min) / range;
            let idx = (normalized * (num_chars - 1) as f32).round() as usize;
            chars[idx.min(num_chars - 1)]
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_sparkline_renders_newest_samples() {
        // Monotonically decreasing history of 50 samples (50.0 → 1.0), i.e.
        // longer than the 20-wide window.  The rendered window must be the tail
        // (20.0 → 1.0) normalised over itself: full block first, lowest block
        // last.  Rendering the *oldest* 20 (50.0 → 31.0) normalised over the
        // whole history would end on a mid-height block instead.
        let values: Vec<f32> = (0..50).map(|i| 50.0 - i as f32).collect();
        let spark = generate_sparkline(&values);
        let glyphs: Vec<char> = spark.chars().collect();
        assert_eq!(
            glyphs.len(),
            SPARKLINE_WIDTH,
            "sparkline must render exactly the window width"
        );
        assert_eq!(
            glyphs[0], '█',
            "oldest sample of the tail window is the window maximum"
        );
        assert_eq!(
            glyphs[SPARKLINE_WIDTH - 1],
            '▁',
            "newest sample is the window minimum — a mid-height block here means \
             the oldest samples are being rendered"
        );
    }

    #[test]
    fn test_sparkline_window_tracks_latest_value() {
        // With the old `take(20)` behaviour the glyphs stopped changing once
        // the history exceeded 20 samples.  Appending a new extreme value must
        // change the rendering.
        let mut values: Vec<f32> = vec![1.0; 30];
        values[29] = 0.5;
        let before = generate_sparkline(&values);
        values.push(0.0);
        let after = generate_sparkline(&values);
        assert_ne!(
            before, after,
            "appending a new sample must change the sparkline"
        );
        assert_eq!(after.chars().count(), SPARKLINE_WIDTH);
    }

    #[test]
    fn test_sparkline_short_and_flat_histories() {
        assert_eq!(generate_sparkline(&[]), "");
        // Flat history → single mid-block per sample, window-limited.
        assert_eq!(generate_sparkline(&[1.0; 3]).chars().count(), 3);
        assert_eq!(
            generate_sparkline(&[1.0; 40]).chars().count(),
            SPARKLINE_WIDTH
        );
    }

    #[test]
    fn test_select_adapter_index() {
        assert_eq!(
            select_adapter_index(2, 1).expect("index 1 of 2 is valid"),
            1
        );
        assert_eq!(
            select_adapter_index(1, 0).expect("index 0 of 1 is valid"),
            0
        );
        let err = select_adapter_index(2, 5).expect_err("index 5 of 2 must fail");
        let msg = err.to_string();
        assert!(msg.contains("out of range"), "unexpected message: {msg}");
        assert!(msg.contains("0..=1"), "must name the valid range: {msg}");
        let none = select_adapter_index(0, 0).expect_err("no adapters must fail");
        assert!(none.to_string().contains("No GPU adapters"));
    }

    #[test]
    fn test_collect_input_frames_sorted_and_filtered() {
        let dir = env::temp_dir().join("oxigaf_pipeline_input_frames");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        // Deliberately create out of lexical order plus a non-image file.
        for name in ["frame_002.png", "frame_001.png", "notes.txt"] {
            std::fs::write(dir.join(name), b"x").expect("write fixture");
        }
        let frames = collect_input_frames(&dir).expect("directory with frames must resolve");
        assert_eq!(frames.len(), 2, "non-image files must be skipped");
        assert!(frames[0].ends_with("frame_001.png"));
        assert!(frames[1].ends_with("frame_002.png"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_collect_input_frames_rejects_empty_dir_and_video() {
        let dir = env::temp_dir().join("oxigaf_pipeline_input_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let err = collect_input_frames(&dir).expect_err("empty dir must fail");
        assert!(err.to_string().contains("No decodable frames"));

        let video = dir.join("clip.MP4");
        std::fs::write(&video, b"not really a video").expect("write fixture");
        let err = collect_input_frames(&video).expect_err("video input must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("Video input is not supported"),
            "unexpected message: {msg}"
        );
        assert!(
            msg.contains("ffmpeg"),
            "must suggest frame extraction: {msg}"
        );

        let missing = dir.join("nope").join("frames");
        assert!(collect_input_frames(&missing).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
