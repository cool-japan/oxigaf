//! End-to-end reconstruction pipeline orchestration.
//!
//! Wires together FLAME model loading, Gaussian initialisation, trainer
//! creation, the training loop (with progress reporting), and final export.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use nalgebra as na;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use oxigaf::flame::{Camera, FlameModel, FlameParams};
use oxigaf::render::gaussian::GaussianModel;
use oxigaf::trainer::diffusion_target::DiffusionTargetConfig;
use oxigaf::trainer::init::GaussianInitializer;
use oxigaf::trainer::Trainer;

use crate::config::{DeviceSection, ProjectConfig, TrainingSection};
use crate::progress;
use crate::verbosity::Verbosity;

// ---------------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------------

/// Seed used when `[training] seed` (and therefore `--seed`) is unset.
///
/// A fixed value rather than an entropy draw: an OxiGAF run is reproducible by
/// default, and `--seed` changes *which* run you get rather than switching
/// reproducibility on. This is the literal that used to be hardcoded at both
/// RNG construction sites.
///
/// # What the seed does and does not reach
///
/// It reaches Gaussian initialisation (`STREAM_GAUSSIAN_INIT`) and the
/// trainer's own `StdRng` (`STREAM_TRAINER`) — which drives view sampling
/// and densification — so those become a pure function of it.
///
/// It does **not** reach the diffusion denoiser's noise. `MultiViewDiffusionPipeline`
/// is started with `u64::from(iteration)` as its seed
/// (`oxigaf_trainer::diffusion_target`), a value derived from the step counter
/// alone. Diffusion noise is therefore still perfectly *reproducible* — the
/// same iteration always draws the same noise — but it is not *varied* by
/// `--seed`: two runs that differ only in their seed share their diffusion
/// noise. Changing that needs a seed parameter on the pipeline's session
/// entry point in `oxigaf-trainer`, which is filed as a followup.
pub const DEFAULT_SEED: u64 = 42;

/// Label for the RNG stream that places the initial Gaussians on the mesh.
const STREAM_GAUSSIAN_INIT: &str = "gaussian-init";

/// Label for the trainer's own RNG stream (view sampling, densification,
/// diffusion noise).
const STREAM_TRAINER: &str = "trainer";

/// Derive an independent, reproducible sub-seed for one named RNG stream.
///
/// Every RNG in the pipeline is seeded from the single master seed, but they
/// must not all be seeded with the *same* value: two `StdRng`s built from one
/// seed emit the identical sequence, so the Gaussian initialiser and the
/// trainer's view sampler would draw correlated numbers. Mixing the stream
/// label in gives each consumer its own stream while keeping the whole run a
/// pure function of the master seed.
///
/// The mixing function is SplitMix64 over `base` combined with an FNV-1a hash
/// of `stream` — chosen because it is deterministic across platforms and
/// releases (unlike [`std::hash::DefaultHasher`], whose output is explicitly
/// not guaranteed stable), which is exactly what reproducibility requires.
fn derive_seed(base: u64, stream: &str) -> u64 {
    // FNV-1a over the stream label.
    let mut label_hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in stream.as_bytes() {
        label_hash ^= u64::from(*byte);
        label_hash = label_hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    // SplitMix64 finaliser over the combined value.
    let mut z = base
        .wrapping_add(label_hash)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Cooperative interruption
// ---------------------------------------------------------------------------

/// Set when the process has been asked to stop at the next clean boundary.
///
/// Process-global because the only writer is a signal handler, which has no
/// way to reach the running [`run_reconstruction`] call. The training loop
/// polls it once per iteration and, when it is set, breaks out — falling
/// through to the same final-checkpoint write a completed run performs, so an
/// interrupted run keeps its work instead of losing everything since the last
/// periodic checkpoint.
///
/// This is inert until something calls [`request_interrupt`]. `main.rs`
/// currently arms its own `INTERRUPT_FLAG` only under `train --interactive`
/// (it hands the `InteractiveController`'s `quit_requested` to the SIGINT
/// handler), so non-interactive training is still killed outright by Ctrl+C
/// until that handler also calls [`request_interrupt`] — a one-line change
/// filed as a followup, because `main.rs` is outside this module's ownership.
static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Ask any in-flight [`run_reconstruction`] to stop at its next iteration
/// boundary and write a final checkpoint.
///
/// Safe to call from a signal handler thread, and idempotent.
pub fn request_interrupt() {
    INTERRUPT_REQUESTED.store(true, Ordering::SeqCst);
}

/// Whether [`request_interrupt`] has been called.
#[must_use]
pub fn interrupt_requested() -> bool {
    INTERRUPT_REQUESTED.load(Ordering::SeqCst)
}

/// Clear the cooperative-interrupt flag.
///
/// Exposed so a long-lived process that runs more than one reconstruction can
/// reset the flag between runs; `run_reconstruction` clears it on entry for
/// the same reason.
pub fn clear_interrupt() {
    INTERRUPT_REQUESTED.store(false, Ordering::SeqCst);
}

/// The `stop_note` value for a run that was never asked to stop early.
const STOP_COMPLETED: &str = "completed";

/// Whether the run should report itself as having completed normally.
///
/// Checking `iteration >= total` alone is not enough: an early stop can fire
/// on the very last iteration — patience running out, or `--early-stop-loss`
/// being met at iteration `total` — leaving the counter at `total` even
/// though the loop exited through a `break`. The progress bar would then be
/// finished twice, the second call overwriting the message that explained why
/// the run stopped.
fn reached_full_iteration_count(stop_note: &str, iteration: u32, total: u32) -> bool {
    stop_note == STOP_COMPLETED && iteration >= total
}

/// Why the training loop stopped, when it stopped early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopReason {
    /// The interactive controller's `q` key.
    UserQuit,
    /// A cooperative process interrupt (SIGINT).
    Interrupted,
}

/// Decide whether the training loop should stop before the next step.
///
/// Pure so the precedence between the two shutdown channels is testable
/// without a terminal, a signal, or the process-global flag: `interrupted` is
/// passed in rather than read from [`INTERRUPT_REQUESTED`], which keeps the
/// tests free of shared mutable state (and therefore safe under the parallel
/// test harness).
fn stop_reason(
    controller: Option<&crate::interactive::InteractiveController>,
    interrupted: bool,
) -> Option<StopReason> {
    if let Some(controller) = controller {
        if controller.quit_requested.load(Ordering::Relaxed) {
            return Some(StopReason::UserQuit);
        }
    }
    if interrupted {
        return Some(StopReason::Interrupted);
    }
    None
}

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// All inputs needed to run the reconstruction pipeline.
///
/// Deliberately carries no `#[allow(dead_code)]`: every field is read by
/// [`run_reconstruction`], and the blanket suppression is what previously hid
/// `input_path` and `device_index` being silently ignored.
///
/// # Where the newer knobs live
///
/// The RNG seed, the evaluation interval, and the loss-threshold early stop
/// are **not** fields here — they are carried by
/// [`ProjectConfig::training`](crate::config::TrainingSection) instead, which
/// this struct already holds. That keeps them settable from `oxigaf.toml` and
/// from the environment (not only from the command line), and keeps this
/// struct's field list — of which `main.rs` is the sole constructor —
/// stable.
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
    pub best_loss: f32,
    pub total_iterations: u32,
    pub num_rigid: usize,
    pub num_flexible: usize,
    /// Master seed the whole run was derived from.
    ///
    /// Reported so a run can be reproduced verbatim by passing it back as
    /// `--seed`, including runs that never named one (they used
    /// [`DEFAULT_SEED`]).
    pub seed: u64,
    /// Why the loop stopped, in a form suitable for a summary line.
    pub stop_reason: &'static str,
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

    // Clear any interrupt left over from an earlier reconstruction in this
    // process, so a stale flag cannot stop this run before it starts. Done
    // here, at entry, rather than just before the loop: everything below —
    // frame validation, FLAME loading, GPU acquisition, restoring a large
    // checkpoint — can take a long time, and a Ctrl+C sent during it means
    // "do not start", which the loop's first check then honours.
    clear_interrupt();

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

    // 4a. Resolve the master seed. Every RNG below is derived from it, so the
    //     whole run is a pure function of this one number.
    let seed = config.project_config.training.seed.unwrap_or(DEFAULT_SEED);
    tracing::info!(
        seed,
        explicit = config.project_config.training.seed.is_some(),
        "Reproducibility: all RNG streams derived from this master seed"
    );

    // 4b. Request GPU device for the rasterizer.
    let device_section = &config.project_config.device;
    let device_index = resolve_device_index(config.device_index, device_section.gpu_index);
    let (device, queue) = request_gpu_device(device_index, device_section).with_context(|| {
        format!("Failed to initialise GPU device {device_index} for rasterizer")
    })?;
    tracing::info!("GPU device {device_index} acquired for rasterizer");

    // 5. Initialise or resume trainer.
    //
    // Built through the `*_with_diffusion` constructors rather than the
    // convenience ones, which hardcode `DiffusionTargetConfig::default()` and
    // so discarded `[training] num_inference_steps` entirely.
    let trainer_seed = derive_seed(seed, STREAM_TRAINER);
    let diffusion_target_config = diffusion_target_config(&config.project_config.training);
    let mut trainer = if let Some(ref ckpt_path) = config.resume_checkpoint {
        tracing::info!("Resuming training from checkpoint: {}", ckpt_path.display());
        Trainer::from_checkpoint_with_diffusion(
            training_config,
            ckpt_path,
            raster_config,
            device,
            queue,
            trainer_seed,
            diffusion_target_config,
        )
        .context("Failed to restore trainer from checkpoint")?
    } else {
        tracing::info!("Initialising Gaussians on mesh surface…");
        let mut rng = StdRng::seed_from_u64(derive_seed(seed, STREAM_GAUSSIAN_INIT));
        // Initialisation is fallible: an empty/malformed/collapsed mesh used to
        // yield a silently empty model while the log below still reported a
        // full Gaussian count. Surface that as a run-ending error instead.
        let model = GaussianInitializer::initialize(&mesh, &training_config.init, &mut rng)
            .context("Failed to initialise Gaussians on the FLAME mesh surface")?;
        tracing::info!(
            "Initialised {} Gaussians ({} rigid, {} flexible)",
            model.len(),
            model.is_rigid.iter().filter(|&&r| r).count(),
            model.is_rigid.iter().filter(|&&r| !r).count(),
        );
        Trainer::with_diffusion_config(
            training_config,
            model,
            raster_config,
            device,
            queue,
            trainer_seed,
            diffusion_target_config,
        )
        .context("Failed to create trainer")?
    };

    // 5b. Load the multi-view diffusion weights, if the configured directory
    //     actually holds them. Nothing used to call this at all, so every
    //     `oxigaf train` run silently trained without Score Distillation
    //     Sampling and `[model] diffusion_weights_dir` was inert.
    load_diffusion_weights(
        &mut trainer,
        &config.project_config.model.diffusion_weights_dir,
    );

    // 6. Prepare output directories
    let checkpoint_dir = config.output_dir.join("checkpoints");
    std::fs::create_dir_all(&checkpoint_dir).context("Failed to create checkpoint directory")?;

    // 7. Training loop with progress bar and early stopping.
    let total = trainer.config.total_iterations;
    let ckpt_interval = trainer.config.checkpoint_interval;
    let base_log_interval = trainer.config.log_interval;
    let start_iter = trainer.iteration;
    let remaining = total.saturating_sub(start_iter);
    let eval_interval = config.project_config.training.eval_interval;
    let early_stop_loss = config.project_config.training.early_stop_loss;
    let mut last_eval_iteration = start_iter;
    let mut stop_note = STOP_COMPLETED;

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
            // Handle pause state. A cooperative interrupt must also break the
            // pause gate, otherwise Ctrl+C on a paused run hangs forever.
            while controller.paused.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if stop_reason(Some(controller), interrupt_requested()).is_some() {
                    break;
                }
            }

            // Check save request
            if controller.save_requested.swap(false, Ordering::Relaxed) {
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

        // Cooperative shutdown: the interactive `q` key, or a SIGINT that
        // reached `request_interrupt`. Breaking here falls through to the
        // unconditional final-checkpoint write below, so an interrupted run
        // keeps everything it has learned rather than only what the last
        // periodic checkpoint captured.
        if let Some(reason) = stop_reason(interactive_controller, interrupt_requested()) {
            let (message, note) = match reason {
                StopReason::UserQuit => ("interrupted by user", "user-quit"),
                StopReason::Interrupted => ("interrupted (SIGINT)", "interrupted"),
            };
            stop_note = note;
            pb.finish_with_message(message);
            tracing::info!(
                iteration = trainer.iteration,
                reason = note,
                "Training stopping early — writing a final checkpoint before returning"
            );
            break;
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
                stop_note = "early-stop-patience";
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
        if ckpt_interval > 0 && step.iteration.is_multiple_of(ckpt_interval) {
            let path = checkpoint_dir.join(format!("ckpt_{:06}.json", step.iteration));
            trainer
                .save_checkpoint(&path)
                .with_context(|| format!("Checkpoint save failed at iter {}", step.iteration))?;
        }

        // Periodic logging. The interactive `[v]` key raises the rate to
        // every iteration for as long as it is toggled on, so a user watching
        // a run go wrong can see each step without restarting it.
        let verbose_now = interactive_controller
            .is_some_and(|controller| controller.verbose_toggle.load(Ordering::Relaxed));
        let log_interval = effective_log_interval(base_log_interval, verbose_now);
        if log_interval > 0 && step.iteration.is_multiple_of(log_interval) {
            tracing::info!(
                iter = step.iteration,
                loss = %format!("{:.6}", current_loss),
                best_loss = %format!("{:.6}", best_loss),
                gaussians = step.num_gaussians,
                patience = format!("{}/{}", patience_counter, config.patience.unwrap_or(0)),
                "training",
            );
            if verbose_now {
                // Detail that is only worth printing when the user has asked
                // for it: the individual loss terms behind `loss`.
                tracing::info!(
                    iter = step.iteration,
                    l1 = %format!("{:.6}", step.loss.l1),
                    ssim = %format!("{:.6}", step.loss.ssim),
                    lpips = %format!("{:.6}", step.loss.lpips),
                    sds = %format!("{:.6}", step.sds_loss),
                    reg = %format!("{:.6}",
                        step.loss.position_reg + step.loss.scale_reg + step.loss.opacity_reg),
                    used_diffusion = step.used_diffusion,
                    "training/verbose",
                );
            }
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

        // Periodic evaluation pass. Runs last in the iteration so it reports
        // on a fully-recorded step.
        if should_evaluate(step.iteration, eval_interval) {
            let window = eval_window(step.iteration, last_eval_iteration);
            last_eval_iteration = step.iteration;
            let evaluation = Evaluation::collect(&trainer, window);
            evaluation.log(step.iteration);
            if let Some(ref mut tb_writer) = tensorboard_writer {
                if let Err(e) =
                    tb_writer.log_scalars(&evaluation.scalars(), i64::from(step.iteration))
                {
                    tracing::warn!("Failed to log evaluation to TensorBoard: {}", e);
                }
            }
        }

        // Loss-threshold early stop. Checked at the very end of the iteration
        // so the step that crossed the threshold is still fully reported —
        // progress bar, log line, metrics file, TensorBoard, and evaluation.
        if let Some(threshold) = early_stop_loss {
            if current_loss <= threshold {
                stop_note = "early-stop-loss";
                pb.finish_with_message("loss threshold reached");
                tracing::info!(
                    iteration = step.iteration,
                    loss = %format!("{:.6}", current_loss),
                    threshold = %format!("{:.6}", threshold),
                    "Early stopping: total loss reached the --early-stop-loss threshold"
                );
                break;
            }
        }
    }

    // Only the run that fell out of the loop naturally reports "complete".
    // Testing `trainer.iteration >= total` alone would relabel an early stop
    // that happened to trigger on the very last iteration — patience running
    // out, or `--early-stop-loss` being met at iteration `total` — as a
    // completed run, overwriting the message that said why it stopped.
    if reached_full_iteration_count(stop_note, trainer.iteration, total) {
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

    tracing::info!(
        seed,
        stop_reason = stop_note,
        "Run finished — reproduce it exactly with `--seed {seed}`"
    );

    Ok(TrainingResult {
        model: trainer.model.clone(),
        final_loss,
        best_loss,
        total_iterations: trainer.iteration,
        num_rigid,
        num_flexible,
        seed,
        stop_reason: stop_note,
    })
}

// ---------------------------------------------------------------------------
// Diffusion wiring
// ---------------------------------------------------------------------------

/// Build the diffusion target configuration from the project's `[training]`
/// section.
///
/// Only `num_inference_steps` is carried across: the guidance schedule
/// (`guidance_scale_start`/`_end`/`guidance_anneal_steps`) reaches the
/// generator through [`oxigaf::trainer::TrainingConfig`], which the trainer
/// merges in itself, and the remaining knobs (warmup, timestep annealing, SDS
/// weight) have no `oxigaf.toml` representation yet and keep their defaults.
///
/// Without this, both trainer constructors fell back to
/// `DiffusionTargetConfig::default()`, so a user who set
/// `num_inference_steps = 20` to halve their iteration cost silently got 50.
fn diffusion_target_config(training: &TrainingSection) -> DiffusionTargetConfig {
    DiffusionTargetConfig {
        num_inference_steps: training.num_inference_steps,
        ..DiffusionTargetConfig::default()
    }
}

/// Load multi-view diffusion weights into `trainer`, if they are present.
///
/// The weights are a large external asset that `oxigaf setup` downloads into
/// the cache; a fresh install legitimately does not have them. Their absence
/// is therefore not an error — training falls back to self-supervision — but
/// it is not silent either, because the difference in what the run optimises
/// is enormous. Both branches say plainly which mode the run is in.
fn load_diffusion_weights(trainer: &mut Trainer, weights_dir: &Path) {
    let resolved = crate::config::expand_tilde(weights_dir);
    if !resolved.is_dir() {
        tracing::warn!(
            "Diffusion weights not found at {} — training self-supervised (no Score \
             Distillation Sampling). Run `oxigaf setup` to download them, or point \
             `[model] diffusion_weights_dir` at an existing directory.",
            resolved.display(),
        );
        return;
    }
    match trainer.load_diffusion_pipeline(&resolved) {
        Ok(()) => tracing::info!(
            "Multi-view diffusion pipeline loaded from {} — SDS enabled after warmup",
            resolved.display()
        ),
        Err(e) => tracing::warn!(
            "Failed to load diffusion weights from {}: {e} — training self-supervised \
             (no Score Distillation Sampling)",
            resolved.display(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Periodic evaluation
// ---------------------------------------------------------------------------

/// Whether iteration `iteration` should trigger an evaluation pass.
///
/// `None` (the `--eval-interval` default) disables evaluation entirely; an
/// interval of `0` would make `is_multiple_of` degenerate to "only iteration
/// 0", so it is rejected here as well as by
/// [`ProjectConfig::validate`](crate::config::ProjectConfig::validate).
fn should_evaluate(iteration: u32, interval: Option<u32>) -> bool {
    match interval {
        Some(interval) if interval > 0 => iteration.is_multiple_of(interval),
        _ => false,
    }
}

/// Number of recorded steps an evaluation should aggregate over.
///
/// This is the span since the previous evaluation, so consecutive evaluations
/// partition the run rather than overlapping (a fixed window would keep
/// re-reporting the same steps, which makes a metric look flatter than it is).
/// Always at least one sample, so the very first evaluation reports something.
fn eval_window(iteration: u32, last_eval_iteration: u32) -> usize {
    iteration.saturating_sub(last_eval_iteration).max(1) as usize
}

/// One periodic evaluation of the run in progress.
///
/// # What these numbers measure
///
/// PSNR and SSIM here are the trainer's own recorded agreement between its
/// **rendered views** and its **diffusion targets** — the signal it is
/// actually optimising. They are *not* measured against the `--input`
/// footage: this pipeline is self-supervised and does not consume the input
/// frames as photometric targets (see the warning `run_reconstruction` emits
/// on entry), so there is no observed ground truth to score against. Reporting
/// one would mean inventing it.
///
/// What the numbers are genuinely good for is convergence and stability: a
/// PSNR that stops climbing, an SSIM that falls, or a Gaussian count running
/// away all show up here long before the final export does.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Evaluation {
    /// Steps aggregated.
    window: usize,
    /// Mean PSNR (dB) of rendered views against diffusion targets.
    mean_psnr: f32,
    /// Mean SSIM of rendered views against diffusion targets.
    mean_ssim: f32,
    /// Mean total loss over the window.
    mean_loss: f32,
    /// Live Gaussian count at evaluation time.
    num_gaussians: usize,
}

impl Evaluation {
    /// Aggregate the trainer's recorded metrics over the last `window` steps.
    fn collect(trainer: &Trainer, window: usize) -> Self {
        Self {
            window,
            mean_psnr: trainer.metric_tracker.mean_psnr(window),
            mean_ssim: trainer.metric_tracker.mean_ssim(window),
            mean_loss: trainer.metric_tracker.mean_loss(window),
            num_gaussians: trainer.model.len(),
        }
    }

    /// TensorBoard scalar series for this evaluation.
    fn scalars(&self) -> [(&'static str, f32); 4] {
        [
            ("eval/psnr", self.mean_psnr),
            ("eval/ssim", self.mean_ssim),
            ("eval/loss", self.mean_loss),
            ("eval/num_gaussians", self.num_gaussians as f32),
        ]
    }

    /// Emit the evaluation on the log sink.
    fn log(&self, iteration: u32) {
        tracing::info!(
            iter = iteration,
            window = self.window,
            psnr = %format!("{:.3}", self.mean_psnr),
            ssim = %format!("{:.4}", self.mean_ssim),
            loss = %format!("{:.6}", self.mean_loss),
            gaussians = self.num_gaussians,
            "evaluation (rendered views vs diffusion targets — not vs --input footage)",
        );
    }
}

// ---------------------------------------------------------------------------
// Interactive verbosity
// ---------------------------------------------------------------------------

/// Logging interval to use for this step.
///
/// While the interactive `[v]` toggle is on, every iteration is logged
/// regardless of the configured `log_interval` — including when the
/// configured interval is `0` ("never log"), which is precisely the
/// configuration a user reaches for the toggle from.
fn effective_log_interval(base: u32, verbose: bool) -> u32 {
    if verbose {
        1
    } else {
        base
    }
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
pub(crate) fn select_adapter_index(num_adapters: usize, requested: usize) -> Result<usize> {
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

/// Reconcile the `--device` index with `[device] gpu_index` from the config.
///
/// `--device` must win when the user passed it, but `TrainArgs::device` is a
/// plain `usize` with clap's `default_value = "0"`, so by the time the value
/// reaches this module "the user typed `--device 0`" and "clap filled in its
/// default" are the same bit pattern. The only rule available is therefore:
///
/// * a non-zero `--device` is necessarily explicit, so it wins outright;
/// * a zero `--device` may or may not be explicit, so `[device] gpu_index`
///   is honoured instead — which matters, because `gpu_index` has no other
///   way to take effect and a config file the user wrote is itself an
///   explicit statement.
///
/// The ambiguity is not fixable here; it needs `TrainArgs::device` to become
/// `Option<usize>` in `cli.rs` (filed as a followup, alongside the same
/// change for `TrainArgs::config`). The precedence is logged by
/// [`request_gpu_device`] so the resolved index is never a mystery.
pub(crate) fn resolve_device_index(cli_device: usize, config_gpu_index: usize) -> usize {
    if cli_device != 0 {
        cli_device
    } else {
        config_gpu_index
    }
}

/// Request a wgpu device and queue for GPU-accelerated rasterization.
///
/// `device_index` is the GPU index [`resolve_device_index`] settled on from
/// `--device` and `[device] gpu_index`:
///
/// * `0` means "let wgpu pick the best adapter" — a
///   [`wgpu::PowerPreference::HighPerformance`] request.  Enumerating
///   unconditionally and taking `adapters[0]` would regress this, because on a
///   laptop the first *enumerated* adapter is usually the integrated GPU.
/// * any other index selects `enumerate_adapters()[index]` explicitly and
///   errors (listing the adapters) when the index is out of range.
///
/// Both paths are constrained to the configured backend, so index `N` counts
/// only the adapters of that backend.
///
/// A consequence of that asymmetry: `--device 0` and `--device N` can resolve
/// to the same physical GPU when the high-performance pick happens to be
/// adapter `N`.  The selected adapter name, backend, and `device_index` are
/// logged on both paths so the mapping is visible.
///
/// `device_section` selects the wgpu backend: an unset (empty) `backend` key
/// means "let wgpu choose" and yields [`wgpu::Backends::all`], while an
/// explicit `vulkan`/`metal`/`dx12`/`gl` restricts both adapter enumeration
/// and the instance to that backend.
///
/// Uses `pollster` style blocking — safe to call from a tokio context because
/// `wgpu::Instance::request_adapter` / `Adapter::request_device` only block on
/// the GPU driver, not on tokio I/O.
fn request_gpu_device(
    device_index: usize,
    device_section: &DeviceSection,
) -> Result<(wgpu::Device, wgpu::Queue)> {
    let configured = device_section
        .resolve_backends()
        .context("Invalid [device] backend")?;
    let backends = configured.unwrap_or_else(wgpu::Backends::all);
    tracing::debug!(
        ?backends,
        explicit = configured.is_some(),
        device_index,
        "GPU backend selection"
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
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
        let adapters = pollster::block_on(instance.enumerate_adapters(backends));
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
        required_limits: RASTERIZER_LIMITS.with(wgpu::Limits::default()),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::default(),
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| anyhow::anyhow!("GPU device creation failed: {e}"))?;

    Ok((device, queue))
}

/// Device limits the Gaussian rasterizer needs beyond `wgpu`'s defaults.
///
/// The backward pass binds 13+ storage buffers in a single shader stage,
/// against a default `max_storage_buffers_per_shader_stage` of 8 — so a
/// device requested with plain defaults fails on the *first backward pass*,
/// long after the run looks like it started successfully. `Rasterizer::new`
/// (oxigaf-render) and `benchmark::request_benchmark_gpu_device` already ask
/// for 16; a device built here is handed straight to the same rasterizer and
/// must match.
struct RasterizerLimits {
    max_storage_buffers_per_shader_stage: u32,
}

impl RasterizerLimits {
    /// Apply these requirements on top of a baseline limit set.
    fn with(&self, base: wgpu::Limits) -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: self.max_storage_buffers_per_shader_stage,
            ..base
        }
    }
}

/// The concrete limits [`request_gpu_device`] requests.
const RASTERIZER_LIMITS: RasterizerLimits = RasterizerLimits {
    max_storage_buffers_per_shader_stage: 16,
};

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

    // -----------------------------------------------------------------------
    // Reproducibility (--seed)
    //
    // Regression coverage for: both RNG construction sites hardcoded
    // `seed_from_u64(42)`, so `--seed` could not change the run at all.
    // -----------------------------------------------------------------------

    #[test]
    fn derive_seed_is_deterministic() {
        // Same inputs, same output — across calls and across processes (the
        // mixing function is fixed, not `DefaultHasher`).
        assert_eq!(
            derive_seed(7, STREAM_TRAINER),
            derive_seed(7, STREAM_TRAINER)
        );
        assert_eq!(derive_seed(0, "x"), derive_seed(0, "x"));
    }

    #[test]
    fn derive_seed_separates_streams() {
        // The whole point: two RNGs seeded from one master seed must not
        // emit the same sequence.
        assert_ne!(
            derive_seed(DEFAULT_SEED, STREAM_GAUSSIAN_INIT),
            derive_seed(DEFAULT_SEED, STREAM_TRAINER),
            "each RNG stream must get its own sub-seed"
        );
    }

    #[test]
    fn derive_seed_responds_to_the_master_seed() {
        // A different `--seed` must produce a different run.
        assert_ne!(
            derive_seed(1, STREAM_TRAINER),
            derive_seed(2, STREAM_TRAINER)
        );
        assert_ne!(
            derive_seed(DEFAULT_SEED, STREAM_GAUSSIAN_INIT),
            derive_seed(DEFAULT_SEED + 1, STREAM_GAUSSIAN_INIT),
        );
        // Adjacent seeds must not merely shift the stream, which a plain
        // `base + label` mix would do.
        assert_ne!(
            derive_seed(1, STREAM_TRAINER),
            derive_seed(0, STREAM_GAUSSIAN_INIT)
        );
    }

    #[test]
    fn derive_seed_drives_distinct_rng_sequences() {
        let mut a = StdRng::seed_from_u64(derive_seed(DEFAULT_SEED, STREAM_GAUSSIAN_INIT));
        let mut b = StdRng::seed_from_u64(derive_seed(DEFAULT_SEED, STREAM_TRAINER));
        let mut c = StdRng::seed_from_u64(derive_seed(DEFAULT_SEED, STREAM_GAUSSIAN_INIT));
        use rand::Rng;
        let first: [u64; 4] = std::array::from_fn(|_| a.next_u64());
        let second: [u64; 4] = std::array::from_fn(|_| b.next_u64());
        let repeat: [u64; 4] = std::array::from_fn(|_| c.next_u64());
        assert_ne!(first, second, "streams must be independent");
        assert_eq!(first, repeat, "the same stream must be reproducible");
    }

    // -----------------------------------------------------------------------
    // Diffusion configuration
    //
    // Regression coverage for: both trainer constructors used by the pipeline
    // hardcoded `DiffusionTargetConfig::default()`, so `[training]
    // num_inference_steps` never reached the denoiser.
    // -----------------------------------------------------------------------

    #[test]
    fn diffusion_target_config_carries_num_inference_steps() {
        let mut training = TrainingSection::default();
        training.num_inference_steps = 20;
        let cfg = diffusion_target_config(&training);
        assert_eq!(
            cfg.num_inference_steps, 20,
            "a configured step count must reach the denoiser, not fall back to the default"
        );
        // Everything else stays at the trainer's own defaults.
        let default = DiffusionTargetConfig::default();
        assert_eq!(cfg.warmup_iterations, default.warmup_iterations);
        assert_eq!(cfg.timestep_start, default.timestep_start);
        assert!((cfg.sds_weight - default.sds_weight).abs() < f32::EPSILON);
    }

    #[test]
    fn diffusion_target_config_is_valid_for_the_default_section() {
        // `Trainer::with_diffusion_config` validates the merged config, so a
        // default `oxigaf.toml` must not produce a rejected one.
        let cfg = diffusion_target_config(&TrainingSection::default());
        assert!(
            cfg.validate().is_ok(),
            "the default configuration must build a valid diffusion config"
        );
    }

    // -----------------------------------------------------------------------
    // Cooperative interruption
    //
    // `stop_reason` is pure precisely so these assertions need no signal, no
    // terminal, and no access to the process-global INTERRUPT_REQUESTED flag
    // (which would make them race under the parallel test harness).
    // -----------------------------------------------------------------------

    #[test]
    fn stop_reason_none_when_nothing_asked_to_stop() {
        assert_eq!(stop_reason(None, false), None);
        let ctrl = crate::interactive::InteractiveController::new();
        assert_eq!(stop_reason(Some(&ctrl), false), None);
    }

    #[test]
    fn stop_reason_reports_a_process_interrupt_without_a_controller() {
        // The gap this closes: non-interactive training had no cooperative
        // shutdown path at all, so Ctrl+C discarded every iteration since the
        // last periodic checkpoint.
        assert_eq!(stop_reason(None, true), Some(StopReason::Interrupted));
    }

    #[test]
    fn stop_reason_prefers_the_user_quit_key() {
        let ctrl = crate::interactive::InteractiveController::new();
        ctrl.quit_requested.store(true, Ordering::Relaxed);
        assert_eq!(stop_reason(Some(&ctrl), false), Some(StopReason::UserQuit));
        assert_eq!(
            stop_reason(Some(&ctrl), true),
            Some(StopReason::UserQuit),
            "the interactive quit key is the more specific reason"
        );
    }

    #[test]
    fn reached_full_iteration_count_distinguishes_a_last_iteration_early_stop() {
        // A genuine completion.
        assert!(reached_full_iteration_count(STOP_COMPLETED, 15_000, 15_000));
        assert!(!reached_full_iteration_count(
            STOP_COMPLETED,
            14_999,
            15_000
        ));

        // The case an `iteration >= total` test alone gets wrong: the early
        // stop fired on the final iteration, so the counter reads `total`
        // even though the loop exited through a `break`. Reporting
        // "training complete" here would finish the progress bar a second
        // time and overwrite the reason the run stopped.
        for note in [
            "early-stop-loss",
            "early-stop-patience",
            "interrupted",
            "user-quit",
        ] {
            assert!(
                !reached_full_iteration_count(note, 15_000, 15_000),
                "{note} must not be relabelled as a completed run"
            );
        }
    }

    #[test]
    fn interrupt_flag_round_trips() {
        // The global is exercised only here, and restored, so no other test
        // observes it.
        let previously = interrupt_requested();
        clear_interrupt();
        assert!(!interrupt_requested());
        request_interrupt();
        assert!(interrupt_requested());
        clear_interrupt();
        assert!(!interrupt_requested());
        if previously {
            request_interrupt();
        }
    }

    // -----------------------------------------------------------------------
    // Periodic evaluation (--eval-interval)
    // -----------------------------------------------------------------------

    #[test]
    fn should_evaluate_only_on_interval_boundaries() {
        assert!(should_evaluate(100, Some(50)));
        assert!(should_evaluate(150, Some(50)));
        assert!(!should_evaluate(149, Some(50)));
    }

    #[test]
    fn should_evaluate_disabled_when_unset_or_zero() {
        // Regression: `--eval-interval` was accepted and then ignored. It
        // must stay off by default, and a zero interval must not degenerate
        // into "evaluate only at iteration 0" via `is_multiple_of(0)`.
        for iteration in [0u32, 1, 7, 1000] {
            assert!(!should_evaluate(iteration, None));
            assert!(!should_evaluate(iteration, Some(0)));
        }
    }

    #[test]
    fn eval_window_partitions_the_run() {
        assert_eq!(eval_window(100, 0), 100);
        assert_eq!(eval_window(150, 100), 50);
        // Never zero — the first evaluation must still report something.
        assert_eq!(eval_window(100, 100), 1);
        assert_eq!(eval_window(50, 100), 1);
    }

    // -----------------------------------------------------------------------
    // Interactive verbosity ([v])
    // -----------------------------------------------------------------------

    #[test]
    fn effective_log_interval_honours_the_verbose_toggle() {
        assert_eq!(effective_log_interval(50, false), 50);
        assert_eq!(effective_log_interval(50, true), 1);
        // Even when logging was configured off entirely, which is exactly
        // when a user reaches for the toggle.
        assert_eq!(effective_log_interval(0, true), 1);
        assert_eq!(effective_log_interval(0, false), 0);
    }

    // -----------------------------------------------------------------------
    // GPU device selection
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_device_index_prefers_an_explicit_cli_device() {
        // A non-zero --device is necessarily explicit.
        assert_eq!(resolve_device_index(2, 1), 2);
        // A zero --device is indistinguishable from clap's default, so the
        // config's gpu_index — which has no other way to take effect — wins.
        assert_eq!(resolve_device_index(0, 1), 1);
        assert_eq!(resolve_device_index(0, 0), 0);
    }

    #[test]
    fn rasterizer_limits_raise_the_storage_buffer_ceiling() {
        // Regression: the pipeline requested `wgpu::Limits::default()` (8
        // storage buffers per shader stage) and handed the device to a
        // rasterizer whose backward pass binds 13+, so training started
        // through `oxigaf train` failed on its first backward pass.
        let limits = RASTERIZER_LIMITS.with(wgpu::Limits::default());
        assert!(
            limits.max_storage_buffers_per_shader_stage >= 13,
            "the backward pass binds 13+ storage buffers, got {}",
            limits.max_storage_buffers_per_shader_stage
        );
        assert_eq!(
            limits.max_storage_buffers_per_shader_stage,
            wgpu::Limits {
                max_storage_buffers_per_shader_stage: 16,
                ..wgpu::Limits::default()
            }
            .max_storage_buffers_per_shader_stage,
            "must match what Rasterizer::new requests"
        );
        // Nothing else may be perturbed.
        assert_eq!(
            limits.max_buffer_size,
            wgpu::Limits::default().max_buffer_size
        );
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
