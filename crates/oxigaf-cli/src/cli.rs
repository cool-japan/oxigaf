//! CLI command definitions (clap derive API).
//!
//! Provides comprehensive command-line interface for OxiGAF with:
//! - Training with progress, early stopping, and checkpointing
//! - Multi-format rendering with quality settings
//! - Export to PLY, safetensors, and glTF 2.0
//! - FLAME model conversion utilities
//! - Performance benchmarking
//! - System diagnostics
//!
//! # Where argument structs live
//!
//! The pipeline commands (`train`, `render`, `export`, …) keep their `Args`
//! structs in this file. The twenty *tool families* — `anim`, `analyze`,
//! `batch`, `camera`, `dataset`, `inspect`, `monitor`, `perf`, `pipeline`,
//! `preset`, `preview`, `profile`, `quality`, `report`, `runs`, `scene`,
//! `sweep`, `training`, `video`, `workspace` — keep theirs in
//! [`crate::commands`], one module per family, next to the handler that
//! consumes them. There are forty-one library modules to expose;
//! concentrating every argument struct here would push this file far past
//! the 2000-line ceiling, so [`Command`] refers to the per-family types by
//! path and the families own their own surface.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::Shell;

use crate::verbosity::Verbosity;

/// OxiGAF — Pure Rust Gaussian Avatar Reconstruction.
#[derive(Parser, Debug)]
#[command(name = "oxigaf", version, about, long_about = None)]
pub struct Cli {
    /// Increase verbosity (-v, -vv, -vvv).
    ///
    /// Multiple -v flags increase detail:
    /// - `-v`: Debug info and timing
    /// - `-vv`: Trace-level logging
    /// - `-vvv`: Maximum verbosity
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Quiet mode (only errors).
    ///
    /// Suppresses all output except errors. Conflicts with --verbose.
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,

    /// Dry run - validate without executing.
    ///
    /// Validates inputs, checks permissions, verifies GPU availability,
    /// estimates resources, and reports what would be done without
    /// executing any modifications.
    #[arg(long, global = true)]
    pub dry_run: bool,

    /// Output results as JSON (for scripting).
    ///
    /// Suppresses all normal output (progress bars, info messages) and
    /// outputs only valid JSON on stdout. Enables programmatic parsing.
    #[arg(long, global = true, conflicts_with = "verbose")]
    pub json: bool,

    /// Write logs to file.
    ///
    /// Enables structured logging to the specified file with automatic
    /// rotation and cleanup. Logs are written in the format specified
    /// by --log-format (default: JSON Lines).
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    pub log_file: Option<PathBuf>,

    /// Log rotation strategy.
    ///
    /// Controls when log files are rotated:
    /// - never: Single file, no rotation
    /// - hourly: New file every hour
    /// - daily: New file every day (default)
    #[arg(long, global = true, value_enum, default_value = "daily")]
    pub log_rotation: LogRotationStrategy,

    /// Maximum number of log files to keep.
    ///
    /// When the number of log files exceeds this limit, the oldest
    /// files are automatically deleted.
    #[arg(long, global = true, default_value = "5")]
    pub log_max_files: usize,

    /// Log file format.
    ///
    /// Format for log file output:
    /// - json: JSON Lines format (recommended for parsing)
    /// - pretty: Pretty-printed format (human-readable)
    /// - compact: Compact format (minimal whitespace)
    #[arg(long, global = true, value_enum, default_value = "json")]
    pub log_format: LogFormatType,

    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    /// Get the verbosity level from command-line flags.
    #[must_use]
    pub fn verbosity(&self) -> Verbosity {
        Verbosity::from_flags(self.verbose, self.quiet)
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Train (reconstruct) a 3D Gaussian avatar from a directory of frames.
    ///
    /// OxiGAF is pure Rust and bundles no video demuxer, so a `.mp4`/`.mov`
    /// container is refused with an actionable message: extract the frames
    /// first (`ffmpeg -i clip.mp4 frames/%05d.png`) and pass the directory.
    #[command(alias = "reconstruct")]
    Train(TrainArgs),

    /// Render an existing avatar from novel viewpoints.
    Render(RenderArgs),

    /// Export an avatar to standard formats (PLY, glTF, safetensors), or
    /// all of them at once with `--format all`.
    Export(ExportArgs),

    /// Convert FLAME model files (.pkl to .npy format).
    Convert(ConvertArgs),

    /// Run performance benchmarks.
    Benchmark(BenchmarkArgs),

    /// Check system configuration and dependencies.
    Doctor(DoctorArgs),

    /// Download and cache required model weights.
    Setup(SetupArgs),

    /// Manage cached assets.
    Cache {
        #[command(subcommand)]
        command: CacheCommands,
    },

    /// Inspect a model or data file and display its metadata.
    ///
    /// Supported file types:
    /// - `.ply` — PLY header parse: Gaussian count, SH degree, properties, file size, bounding box, opacity/scale stats
    /// - `.safetensors` — tensor names/shapes/dtypes, metadata dict, file size
    /// - `.json` — training config or checkpoint metadata, key fields
    Info(InfoArgs),

    /// Compare two model files and report structural and statistical differences.
    ///
    /// Supported file types: `.ply`, `.safetensors`
    ///
    /// # Example
    ///
    /// ```bash
    /// oxigaf compare model_a.ply model_b.ply
    /// oxigaf compare model_a.ply model_b.ply --format json
    /// oxigaf compare model_a.ply model_b.ply --threshold 0.85
    /// ```
    Compare(CompareArgs),

    /// Manage training configuration files.
    ///
    /// Subcommands:
    /// - `init [--output <path>]` — write default config TOML to stdout or a file
    /// - `validate <path>` — parse the config file and report errors or "OK"
    /// - `show <path>` — parse and pretty-print the config
    ///
    /// # Why the name is spelled twice
    ///
    /// The Rust variant has to be called `ConfigCmd` because [`crate::config`]
    /// already occupies the module path a `Config` variant would document, but
    /// the *command* users type — and the one [`crate::error::CliError`]
    /// suggestions and the CHANGELOG name — is `oxigaf config`. The derived
    /// kebab-case name (`config-cmd`) is kept as an alias so scripts written
    /// against the earlier spelling keep working.
    #[command(name = "config", alias = "config-cmd")]
    ConfigCmd {
        #[command(subcommand)]
        command: ConfigCmdSubcommand,
    },

    /// Generate shell completion scripts.
    ///
    /// # Installation
    ///
    /// ## Bash
    /// ```bash
    /// oxigaf completions bash > /etc/bash_completion.d/oxigaf
    /// # Or for user-level installation:
    /// oxigaf completions bash > ~/.local/share/bash-completion/completions/oxigaf
    /// ```
    ///
    /// ## Zsh
    /// ```bash
    /// oxigaf completions zsh > /usr/local/share/zsh/site-functions/_oxigaf
    /// # Or for user-level installation:
    /// mkdir -p ~/.zsh/completion
    /// oxigaf completions zsh > ~/.zsh/completion/_oxigaf
    /// # Add to ~/.zshrc: fpath=(~/.zsh/completion $fpath)
    /// ```
    ///
    /// ## Fish
    /// ```bash
    /// oxigaf completions fish > ~/.config/fish/completions/oxigaf.fish
    /// ```
    ///
    /// ## PowerShell
    /// ```powershell
    /// oxigaf completions powershell | Out-String | Invoke-Expression
    /// # Or add to profile:
    /// oxigaf completions powershell >> $PROFILE
    /// ```
    ///
    /// ## Elvish
    /// ```elvish
    /// oxigaf completions elvish > ~/.config/elvish/lib/oxigaf.elv
    /// # Add to ~/.config/elvish/rc.elv: use oxigaf
    /// ```
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },

    // -----------------------------------------------------------------
    // Tool families (arguments and handlers live in `crate::commands`)
    // -----------------------------------------------------------------
    /// Inspect and transform per-frame Gaussian animation sequences.
    Anim(crate::commands::anim::AnimArgs),

    /// Read-only inspection: colour calibration, model diffs, image metrics.
    Analyze(crate::commands::analyze::AnalyzeArgs),

    /// Run many model conversions as one dependency-ordered batch.
    Batch(crate::commands::batch::BatchArgs),

    /// Author camera paths and evaluate arcball navigation.
    Camera(crate::commands::camera::CameraArgs),

    /// Scan, validate and split training datasets.
    Dataset(crate::commands::dataset::DatasetArgs),

    /// Read-only interrogation of models, PLY files and memory budgets.
    Inspect(crate::commands::inspect::InspectArgs),

    /// Render a training run's metrics stream as a live dashboard.
    Monitor(crate::commands::monitor::MonitorArgs),

    /// Micro-benchmark the CPU-side numeric kernels.
    Perf(crate::commands::perf::PerfArgs),

    /// Inspect and apply named training hyper-parameter presets.
    Preset(crate::commands::preset::PresetArgs),

    /// Drive a model's camera and re-render it to a live image file.
    Preview(crate::commands::preview::PreviewArgs),

    /// Run the reconstruction workflow one composable stage at a time.
    ///
    /// `plan` describes the sequence; `track`, `diffuse` and `export` run the
    /// stages that stand alone. The training stage needs a GPU device, queue
    /// and initialised Gaussians, so `oxigaf train` remains the end-to-end
    /// driver for it.
    Pipeline(crate::commands::pipeline_cmd::PipelineArgs),

    /// Turn a phase-timing log into a bottleneck report.
    Profile(crate::commands::profile::ProfileArgs),

    /// Quality-gate rendered images against references and hunt for artefacts.
    ///
    /// Unlike `analyze eval`, which scores a render set for reporting, these
    /// commands apply pass/fail thresholds and exit non-zero when a render
    /// misses them — so they can stand in a CI pipeline.
    Quality(crate::commands::quality::QualityArgs),

    /// Build comparison reports across training runs.
    Report(crate::commands::report::ReportArgs),

    /// Create, list, prune and retire training run workspaces.
    ///
    /// Manages the *collection* of run directories; `oxigaf workspace` looks
    /// inside a single one and ranks its checkpoints.
    Runs(crate::commands::runs::RunsArgs),

    /// Whole-scene operations: alignment, analysis, merging, filtering,
    /// optimisation, LOD, compression and streaming plans.
    Scene(crate::commands::scene_ops::SceneArgs),

    /// Plan and score hyper-parameter sweeps.
    Sweep(crate::commands::sweep::SweepArgs),

    /// Analyse a finished training run: summary, smoothing, reports,
    /// resume recommendations and timing traces.
    Training(crate::commands::training::TrainingArgs),

    /// Turn a directory of rendered frames into a GIF, a frame sequence,
    /// a manifest, or a self-contained HTML viewer.
    Video(crate::commands::video::VideoArgs),

    /// Browse and compare the checkpoints of a run directory.
    Workspace(crate::commands::workspace::WorkspaceArgs),
}

// ---------------------------------------------------------------------------
// Train Command (enhanced)
// ---------------------------------------------------------------------------

/// Train a Gaussian avatar from input video
///
/// # Configuration Priority
///
/// Configuration is loaded with the following priority (highest to lowest):
/// 1. CLI arguments (this command's flags)
/// 2. Environment variables (OXIGAF_*)
/// 3. Project config file (--config or ./oxigaf.toml)
/// 4. User config file (~/.config/oxigaf/config.toml)
/// 5. Default values
///
/// # Environment Variables
///
/// ## Training Parameters
///   OXIGAF_TOTAL_ITERATIONS       Number of training iterations (default: 15000)
///   OXIGAF_IMAGE_SIZE             Training image resolution (default: 512)
///   OXIGAF_VIEWS_PER_STEP         Number of views per training step (default: 4)
///   OXIGAF_GUIDANCE_SCALE_START   Starting guidance scale for diffusion (default: 7.5)
///   OXIGAF_GUIDANCE_SCALE_END     Ending guidance scale for diffusion (default: 3.0)
///
/// ## Optimizer Learning Rates
///   OXIGAF_POSITION_LR            Position learning rate (default: 0.00016)
///   OXIGAF_SCALING_LR             Scaling/size learning rate (default: 0.005)
///   OXIGAF_ROTATION_LR            Rotation learning rate (default: 0.001)
///   OXIGAF_OPACITY_LR             Opacity learning rate (default: 0.05)
///   OXIGAF_SH_LR                  Spherical harmonics learning rate (default: 0.0025)
///
/// ## Initialization Parameters
///   OXIGAF_SH_DEGREE              Spherical harmonics degree (0-3, default: 3)
///   OXIGAF_NUM_RIGID_GAUSSIANS    Number of rigid Gaussians (default: 50000)
///   OXIGAF_NUM_FLEXIBLE_GAUSSIANS Number of flexible Gaussians (default: 10000)
///
/// ## Device Configuration
///   OXIGAF_DEVICE_BACKEND         GPU backend (vulkan, metal, dx12, gl, default: vulkan)
///   OXIGAF_DEVICE_GPU_INDEX       GPU device index (default: 0)
///
/// ## Output Configuration
///   OXIGAF_OUTPUT_CHECKPOINT_INTERVAL  Checkpoint save frequency in iterations (default: 1000)
///   OXIGAF_OUTPUT_LOG_INTERVAL         Log interval in iterations (default: 50)
///   OXIGAF_OUTPUT_EXPORT_FORMAT        Export format (ply, safetensors, gltf, default: ply)
///
/// # Example
///
/// ```bash
/// # Frames first — OxiGAF has no video decoder:
/// ffmpeg -i clip.mp4 frames/%05d.png
///
/// # Use environment variables to override config
/// export OXIGAF_TOTAL_ITERATIONS=10000
/// export OXIGAF_POSITION_LR=0.0002
/// oxigaf train -i frames/ -o output/ --flame-model ~/.cache/oxigaf/flame2023
///
/// # CLI args have highest priority
/// oxigaf train -i frames/ -o output/ --flame-model model/ --max-iterations 5000
/// ```
#[derive(Debug, clap::Args)]
pub struct TrainArgs {
    /// Directory of extracted frames (or a single frame image).
    ///
    /// Video containers (`.mp4`, `.mov`, …) are rejected: OxiGAF ships no
    /// video decoder, so extract the frames first with an external tool.
    #[arg(short, long, value_hint = ValueHint::AnyPath)]
    pub input: PathBuf,

    /// Output directory for the reconstructed avatar.
    #[arg(short, long, value_hint = ValueHint::DirPath)]
    pub output: PathBuf,

    /// Path to the converted FLAME model directory (`.npy` files).
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub flame_model: PathBuf,

    /// Path to pre-computed per-frame FLAME tracking parameters (JSON).
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub flame_params: Option<PathBuf>,

    /// Training configuration TOML file.
    #[arg(short, long, default_value = "oxigaf.toml", value_hint = ValueHint::FilePath)]
    pub config: PathBuf,

    /// GPU device index.
    #[arg(long, default_value = "0")]
    pub device: usize,

    /// Resume from a checkpoint file.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub resume: Option<PathBuf>,

    /// Random seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Maximum number of training iterations (overrides config).
    #[arg(long)]
    pub max_iterations: Option<u32>,

    /// Early stopping loss threshold (stop when total loss falls below).
    #[arg(long)]
    pub early_stop_loss: Option<f32>,

    /// Early stopping patience: iterations without improvement before stopping.
    #[arg(long)]
    pub patience: Option<u32>,

    /// Minimum improvement delta for early stopping (default: 1e-4).
    #[arg(long)]
    pub min_delta: Option<f32>,

    /// Checkpoint save interval in iterations (overrides config).
    #[arg(long)]
    pub checkpoint_interval: Option<u32>,

    /// Validation/evaluation interval in iterations.
    #[arg(long)]
    pub eval_interval: Option<u32>,

    /// Disable preview image generation during training.
    #[arg(long)]
    pub no_preview: bool,

    /// Enable interactive mode with keyboard controls.
    #[arg(long)]
    pub interactive: bool,

    /// Export metrics to file (CSV or JSON Lines).
    #[arg(long, value_name = "PATH", value_hint = ValueHint::FilePath)]
    pub metrics_output: Option<PathBuf>,

    /// Metrics output format.
    #[arg(long, value_enum, default_value = "csv")]
    pub metrics_format: MetricsOutputFormat,

    /// Enable TensorBoard logging.
    #[arg(long)]
    pub tensorboard: bool,

    /// TensorBoard log directory.
    #[arg(long, default_value = "runs", value_hint = ValueHint::DirPath)]
    pub tensorboard_dir: PathBuf,

    /// Named hyper-parameter preset, applied before the individual CLI
    /// overrides on this command.
    ///
    /// One of `quick`, `balanced`, `quality`, `research`, `production`,
    /// `portrait` or `video`. The aliases `fast`, `default`, `high`,
    /// `debug`, `prod`, `headshot` and `animation` are accepted as well but
    /// are not offered by shell completion. There is no "custom" profile:
    /// an unknown name is refused by the parser, which names the
    /// alternatives, instead of failing after the configuration has loaded.
    #[arg(long, value_parser = training_profile_parser(), ignore_case = true)]
    pub profile: Option<String>,
}

/// Accepted values for `train --profile`.
///
/// [`crate::config_presets::TrainingPresetName::from_str`] remains the
/// authority on which names resolve; enumerating them for clap is what gives
/// `oxigaf completions <shell>` something to offer for this otherwise
/// stringly-typed flag, and turns a typo into a parse error listing the
/// alternatives. Every alias `from_str` accepts is attached to its canonical
/// value, so `--profile prod` keeps working while completion offers only the
/// canonical spelling. `cli::tests` asserts the two lists stay in step.
fn training_profile_parser() -> clap::builder::PossibleValuesParser {
    use clap::builder::{PossibleValue, PossibleValuesParser};

    PossibleValuesParser::new([
        PossibleValue::new("quick").alias("fast"),
        PossibleValue::new("balanced").alias("default"),
        PossibleValue::new("quality").alias("high"),
        PossibleValue::new("research").alias("debug"),
        PossibleValue::new("production").alias("prod"),
        PossibleValue::new("portrait").alias("headshot"),
        PossibleValue::new("video").alias("animation"),
    ])
}

/// Metrics output format for training metrics export.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MetricsOutputFormat {
    /// CSV format (comma-separated values).
    Csv,
    /// JSON Lines format (one JSON object per line).
    Json,
}

// ---------------------------------------------------------------------------
// Render Command (enhanced)
// ---------------------------------------------------------------------------

#[derive(Debug, clap::Args)]
pub struct RenderArgs {
    /// Path to the avatar model (`.safetensors`, `.ply`, or `.json` checkpoint).
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub model: PathBuf,

    /// Output directory for rendered images.
    #[arg(short, long, value_hint = ValueHint::DirPath)]
    pub output: PathBuf,

    /// Render width in pixels.
    ///
    /// Defaults to the width implied by `--quality`. Passing this flag
    /// always wins over the preset — including when the requested value
    /// happens to equal the old 512 default.
    #[arg(long)]
    pub width: Option<u32>,

    /// Render height in pixels.
    ///
    /// Defaults to the height implied by `--quality`; an explicit value
    /// always wins over the preset.
    #[arg(long)]
    pub height: Option<u32>,

    /// Camera trajectory JSON file with azimuth/elevation/distance.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub cameras: Option<PathBuf>,

    /// FLAME parameters for animation (per-frame JSON).
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub flame_params: Option<PathBuf>,

    /// Rendering mode.
    #[arg(long, default_value = "frames")]
    pub mode: RenderMode,

    /// Number of frames to render (for turntable/orbit modes).
    #[arg(long, default_value = "8")]
    pub num_frames: u32,

    /// Output image format.
    #[arg(long, default_value = "png")]
    pub format: ImageFormat,

    /// Background color (hex color like "ffffff" or "transparent").
    #[arg(long, default_value = "1e1e1e")]
    pub background: String,

    /// Splat radius for software renderer (1-5).
    #[arg(long, default_value = "2")]
    pub splat_radius: u32,

    /// Quality preset affecting render fidelity.
    #[arg(long, default_value = "medium")]
    pub quality: RenderQuality,

    /// Number of parallel threads for rendering. 0 = auto (all cores).
    ///
    /// When set to 0 (default), rayon's global thread pool is used which
    /// selects the number of threads automatically based on available CPU cores.
    /// Set to 1 for sequential (single-threaded) rendering.
    #[arg(long, default_value = "0")]
    pub parallel: usize,
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum RenderMode {
    /// Render individual frames from camera trajectory.
    #[default]
    Frames,
    /// 360-degree turntable rotation around subject.
    Turntable,
    /// Orbit around subject with elevation variation.
    Orbit,
    /// Dolly zoom in/out.
    Dolly,
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum ImageFormat {
    /// PNG (lossless, recommended).
    #[default]
    Png,
    /// JPEG (lossy, smaller file size).
    Jpeg,
    /// EXR (high dynamic range, 16-bit or 32-bit float).
    Exr,
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum RenderQuality {
    /// Fast rendering, lower quality (preview).
    Low,
    /// Balanced quality and speed.
    #[default]
    Medium,
    /// High quality, slower rendering.
    High,
    /// Ultra quality (4096x4096, SH degree 3).
    Ultra,
}

impl RenderQuality {
    /// Get the resolution for this quality preset.
    #[must_use]
    pub fn resolution(&self) -> (u32, u32) {
        match self {
            Self::Low => (512, 512),
            Self::Medium => (1024, 1024),
            Self::High => (2048, 2048),
            Self::Ultra => (4096, 4096),
        }
    }

    /// Get the recommended SH degree for this quality preset.
    ///
    /// Applied by `oxigaf render`: a model whose SH degree exceeds the
    /// preset's is downsampled in memory before rasterisation, so the
    /// preset controls view-dependent fidelity as documented.
    #[must_use]
    pub fn sh_degree(&self) -> u32 {
        match self {
            Self::Low => 0,
            Self::Medium => 1,
            Self::High => 2,
            Self::Ultra => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Export Command (enhanced)
// ---------------------------------------------------------------------------

#[derive(Debug, clap::Args)]
pub struct ExportArgs {
    /// Path to the avatar model (`.safetensors`, `.ply`, or `.json` checkpoint).
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub model: PathBuf,

    /// Output path.
    ///
    /// A file for every format except `all`, which treats this as a
    /// directory (created if it does not exist) and writes `model.ply`,
    /// `model.safetensors`, `model.glb`, and `model.json` into it.
    #[arg(short, long, value_hint = ValueHint::AnyPath)]
    pub output: PathBuf,

    /// Export format.
    ///
    /// `all` writes every format at once -- see `--output` -- instead of
    /// picking one.
    #[arg(long, default_value = "ply")]
    pub format: ExportFormat,

    /// Include training metadata in export.
    #[arg(long)]
    pub include_metadata: bool,

    /// Source checkpoint for metadata (optional).
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub checkpoint: Option<PathBuf>,

    /// PLY format variant (only for `--format ply`).
    ///
    /// `ascii` is human-readable and round-trips through every viewer;
    /// `binary-le` is several times smaller and is what `oxigaf compare`
    /// can analyse in full. `binary-be` is rejected: no 3DGS tool writes
    /// big-endian PLY and this crate has no writer for it.
    #[arg(long, default_value = "ascii")]
    pub ply_format: PlyFormat,

    /// SH degree to export (downsample if less than model's degree).
    #[arg(long)]
    pub sh_degree: Option<u32>,

    /// Overwrite existing output file without prompting.
    #[arg(long)]
    pub force: bool,

    /// Color mode for point cloud export (only for `--format pointcloud`).
    ///
    /// Controls how each point's RGB color is derived:
    /// - `sh-dc`: View-independent color from SH DC coefficient (default)
    /// - `white`: All points rendered as white (255, 255, 255)
    /// - `opacity`: Grayscale proportional to sigmoid(opacity)
    /// - `scale`: Rainbow-ish hue from average log-scale magnitude
    #[arg(long, value_enum, default_value = "sh-dc")]
    pub point_color_mode: PointColorMode,

    /// Voxel grid resolution along the longest axis for mesh export (default 128, max 256).
    ///
    /// Only used when `--format mesh`. Higher values produce finer geometry but
    /// increase memory and compute time proportionally to the cube of this value.
    #[arg(long, default_value = "128")]
    pub mesh_resolution: u32,

    /// Density isosurface threshold for mesh export (default 0.5).
    ///
    /// Only used when `--format mesh`. Lower values capture thinner shells of
    /// Gaussian density; higher values extract only dense core regions.
    #[arg(long, default_value = "0.5")]
    pub mesh_iso: f32,

    /// Fractional bounding-box padding for mesh export (default 0.1).
    ///
    /// Only used when `--format mesh`. Adds padding around the Gaussian extent
    /// so the isosurface is never clipped at the grid boundary.
    #[arg(long, default_value = "0.1")]
    pub mesh_padding: f32,
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum ExportFormat {
    /// Standard 3DGS PLY format (compatible with viewers).
    #[default]
    Ply,
    /// Safetensors format (efficient tensor storage).
    Safetensors,
    /// glTF 2.0 with custom Gaussian extension.
    Gltf,
    /// JSON checkpoint format.
    Json,
    /// Colored PLY point cloud (xyzirgb) from SH DC coefficients.
    PointCloud,
    /// Surface Nets triangle mesh exported as binary little-endian PLY.
    Mesh,
    /// Every format at once: PLY, safetensors, glTF, and JSON checkpoint,
    /// written concurrently.
    ///
    /// `--output` is treated as a directory rather than a single file (see
    /// its help text) and receives `model.ply`, `model.safetensors`,
    /// `model.glb`, and `model.json`. The PLY component is always ASCII
    /// (`--ply-format` has no effect here, same as for every non-PLY
    /// format) and the glTF component is a self-contained `.glb`, not the
    /// `.gltf` + `.bin` pair `--format gltf` writes on its own.
    All,
}

/// Color mode for point cloud export.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum PointColorMode {
    /// Use SH DC coefficient for view-independent color (default).
    #[default]
    ShDc,
    /// Use white (255, 255, 255) for all points.
    White,
    /// Grayscale proportional to sigmoid(opacity).
    Opacity,
    /// Rainbow color by average scale magnitude.
    Scale,
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum PlyFormat {
    /// ASCII format (human-readable, larger).
    #[default]
    Ascii,
    /// Binary little-endian format (compact) — the 3DGS ecosystem default.
    BinaryLe,
    /// Binary big-endian format — accepted by the parser, rejected by the
    /// writer: nothing in the 3DGS ecosystem emits or reads it.
    BinaryBe,
}

// ---------------------------------------------------------------------------
// Convert Command (new)
// ---------------------------------------------------------------------------

#[derive(Debug, clap::Args)]
pub struct ConvertArgs {
    /// Input FLAME pickle file (.pkl) or NPZ file (.npz).
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub input: PathBuf,

    /// Output directory for converted .npy files.
    #[arg(short, long, value_hint = ValueHint::DirPath)]
    pub output: PathBuf,

    /// FLAME model version (2020 or 2023).
    ///
    /// Restricted to the two releases the converter can validate: anything
    /// else used to be accepted and then silently skipped the
    /// version-specific output checks.
    #[arg(long, default_value = "2023", value_parser = ["2020", "2023"])]
    pub version: String,

    /// Include UV coordinates in conversion.
    #[arg(long)]
    pub include_uv: bool,

    /// Verify output integrity after conversion.
    #[arg(long)]
    pub verify: bool,

    /// Force overwrite existing output files.
    #[arg(long)]
    pub force: bool,
}

// ---------------------------------------------------------------------------
// Benchmark Command (new)
// ---------------------------------------------------------------------------

#[derive(Debug, clap::Args)]
pub struct BenchmarkArgs {
    /// Benchmark target component.
    #[arg(short, long, default_value = "full")]
    pub target: BenchTarget,

    /// Number of warmup iterations before timing.
    #[arg(long, default_value = "3")]
    pub warmup: u32,

    /// Number of timed iterations (at least 1).
    ///
    /// Zero is refused by the parser rather than several seconds into the
    /// run: with no timed sample every statistic the report prints (mean,
    /// std-dev, min/max, throughput) is undefined.
    #[arg(long, default_value = "10", value_parser = clap::value_parser!(u32).range(1..))]
    pub iterations: u32,

    /// Output format for benchmark results.
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Save benchmark report to file.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub output: Option<PathBuf>,

    /// Model size for synthetic benchmarks.
    #[arg(long, default_value = "medium")]
    pub size: BenchSize,

    /// Path to a converted FLAME model directory (`.npy` files).
    ///
    /// **Required for `--target flame`**: the FLAME benchmark times a real
    /// forward pass and there is no way to synthesise a meaningful model, so
    /// without this flag the target cannot be measured and the command exits
    /// non-zero. Under the default `--target full` the FLAME leg is instead
    /// recorded in the report's `skipped` list, with the same reason, and the
    /// remaining targets still produce numbers.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub flame_model: Option<PathBuf>,

    /// Compare results against baseline file.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub baseline: Option<PathBuf>,
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum BenchTarget {
    /// FLAME forward pass only.
    Flame,
    /// Rasterizer forward/backward.
    Raster,
    /// Single training iteration.
    Train,
    /// Export performance.
    Export,
    /// Full pipeline (all components).
    #[default]
    Full,
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum OutputFormat {
    /// Human-readable summary.
    #[default]
    Human,
    /// JSON format for programmatic use.
    Json,
    /// CSV format for spreadsheet analysis.
    Csv,
    /// Markdown table for documentation.
    Markdown,
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum BenchSize {
    /// Tiny model (1K Gaussians).
    Tiny,
    /// Small model (10K Gaussians).
    Small,
    /// Medium model (50K Gaussians).
    #[default]
    Medium,
    /// Large model (100K Gaussians).
    Large,
    /// Extra-large model (500K Gaussians).
    Xlarge,
}

impl BenchSize {
    /// Get the number of Gaussians for this benchmark size.
    #[must_use]
    pub fn num_gaussians(&self) -> usize {
        match self {
            Self::Tiny => 1_000,
            Self::Small => 10_000,
            Self::Medium => 50_000,
            Self::Large => 100_000,
            Self::Xlarge => 500_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Doctor Command (new)
// ---------------------------------------------------------------------------

#[derive(Debug, clap::Args)]
pub struct DoctorArgs {
    /// Check specific component only.
    #[arg(long)]
    pub check: Option<DoctorCheck>,

    /// GPU device index to inspect, matching `train --device`.
    ///
    /// The GPU check used to ask wgpu for whichever adapter it considered
    /// highest-performance, which on a multi-GPU machine is not necessarily
    /// the one a run will use: `oxigaf train --device 1` runs on adapter 1.
    /// Passing the same index here inspects that adapter, so `doctor` can
    /// actually pre-flight the device the job will run on.
    #[arg(long, default_value = "0")]
    pub device: usize,

    /// Path to FLAME model directory to verify.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub flame_model: Option<PathBuf>,

    /// Path to cache directory to check.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum DoctorCheck {
    /// Check GPU availability and capabilities.
    Gpu,
    /// Check FLAME model files.
    Flame,
    /// Check asset cache status.
    Cache,
    /// Check Rust/crate versions.
    Version,
    /// Check all components.
    All,
}

// ---------------------------------------------------------------------------
// Setup Command
// ---------------------------------------------------------------------------

#[derive(Debug, clap::Args)]
pub struct SetupArgs {
    /// Directory to cache downloaded model weights.
    ///
    /// Defaults to the platform cache directory (`$XDG_CACHE_HOME/oxigaf`,
    /// `~/Library/Caches/oxigaf`, …), overridable with `OXIGAF_CACHE_DIR`.
    /// `doctor`, `setup` and `cache` all resolve it the same way, so
    /// `oxigaf cache path` always names the directory `setup` populates.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub cache_dir: Option<PathBuf>,

    /// Skip checksum verification.
    #[arg(long)]
    pub skip_checksum: bool,

    /// Download only specified assets (comma-separated).
    #[arg(long)]
    pub only: Option<String>,

    /// Run in offline mode (check cache only).
    #[arg(long)]
    pub offline: bool,

    /// Download from HuggingFace Hub.
    ///
    /// Specify a model identifier like:
    /// - "cool-japan/oxigaf-flame-2023" (default revision)
    /// - "cool-japan/oxigaf-flame-2023:main" (specific branch/tag)
    /// - "cool-japan/oxigaf-flame-2023@v1.0" (specific commit/version)
    #[arg(long)]
    pub from_hub: Option<String>,

    /// HuggingFace authentication token for private models.
    ///
    /// Can also be set via HF_TOKEN environment variable or
    /// in ~/.huggingface/token file.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Model revision (branch, tag, or commit SHA).
    ///
    /// Overrides revision specified in --from-hub if both are provided.
    #[arg(long)]
    pub revision: Option<String>,

    /// Filename to download from the HuggingFace repository.
    ///
    /// Default: "model.safetensors"
    #[arg(long)]
    pub filename: Option<String>,
}

// ---------------------------------------------------------------------------
// Cache Commands
// ---------------------------------------------------------------------------

#[derive(Debug, Subcommand)]
pub enum CacheCommands {
    /// List all cached assets with details.
    List,

    /// Clean old cached assets.
    Clean {
        /// Maximum age in days (assets older than this will be removed).
        #[arg(long, default_value = "30")]
        max_age_days: u64,

        /// Dry run (show what would be deleted without deleting).
        #[arg(long)]
        dry_run: bool,
    },

    /// Verify cache integrity (check file existence, size, and checksums).
    Verify,

    /// Print cache directory path.
    Path,
}

// ---------------------------------------------------------------------------
// Info Command
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf info <path>`.
#[derive(Debug, clap::Args)]
pub struct InfoArgs {
    /// File to inspect (.ply, .safetensors, or .json).
    #[arg(value_hint = ValueHint::FilePath)]
    pub path: PathBuf,
}

// ---------------------------------------------------------------------------
// Compare Command
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf compare <model1> <model2>`.
#[derive(Debug, clap::Args)]
pub struct CompareArgs {
    /// First model file to compare (.ply or .safetensors).
    #[arg(value_hint = ValueHint::FilePath)]
    pub model1: PathBuf,

    /// Second model file to compare (.ply or .safetensors).
    #[arg(value_hint = ValueHint::FilePath)]
    pub model2: PathBuf,

    /// Output format: "text" (default) or "json".
    ///
    /// The explicit value list also gives shell completions something to
    /// offer for this otherwise stringly-typed flag.
    #[arg(long, default_value = "text", value_parser = ["text", "json"])]
    pub format: String,

    /// Similarity threshold for recommendation (0.0–1.0).
    ///
    /// Models above this threshold are considered similar. Below it, they are
    /// flagged as significantly different.
    #[arg(long, default_value = "0.8")]
    pub threshold: f64,
}

// ---------------------------------------------------------------------------
// ConfigCmd Subcommands
// ---------------------------------------------------------------------------

/// Subcommands for `oxigaf config` (also reachable as `config-cmd`).
#[derive(Debug, Subcommand)]
pub enum ConfigCmdSubcommand {
    /// Write a default OxiGAF configuration TOML to stdout or to a file.
    Init {
        /// Write to this file instead of stdout.
        #[arg(long, short, value_hint = ValueHint::FilePath)]
        output: Option<PathBuf>,

        /// Generate config via hardware-detection wizard (non-interactive).
        ///
        /// Queries a real `wgpu` adapter and picks VRAM-bound defaults for
        /// `sh_degree`, `views_per_step`, `image_size` and `max_gaussians` —
        /// the four settings a 3DGS trainer is actually limited by. (It is
        /// deliberately *not* driven by CPU-core count, which says nothing
        /// about how large a scene fits on the GPU.) With no adapter
        /// reachable it falls back to a conservative CPU-class profile.
        /// Prints the reasoning for each configuration choice.
        #[arg(long)]
        interactive: bool,
    },

    /// Parse a configuration file and report any errors, or print "OK".
    Validate {
        /// Path to the configuration TOML file.
        #[arg(value_hint = ValueHint::FilePath)]
        path: PathBuf,
    },

    /// Parse a configuration file and pretty-print all fields.
    Show {
        /// Path to the configuration TOML file.
        #[arg(value_hint = ValueHint::FilePath)]
        path: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Log Rotation Enums
// ---------------------------------------------------------------------------

/// Log rotation strategy for file logging.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LogRotationStrategy {
    /// Never rotate (single file).
    Never,
    /// Rotate hourly.
    Hourly,
    /// Rotate daily.
    Daily,
}

/// Log file format.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LogFormatType {
    /// JSON Lines format (recommended for parsing).
    Json,
    /// Pretty-printed format (human-readable).
    Pretty,
    /// Compact format (minimal whitespace).
    Compact,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_presets::TrainingPresetName;
    use clap::CommandFactory;

    /// Regression: [`crate::error::CliError::suggestion`] and the CHANGELOG
    /// both tell users to run `oxigaf config …`, but the derived kebab-case
    /// name was `config-cmd`, so every suggestion the CLI printed named a
    /// subcommand that did not exist. The canonical name is now `config`; the
    /// old spelling survives as an alias so existing scripts keep working.
    #[test]
    fn config_family_is_named_config_and_keeps_the_legacy_alias() {
        let cli = Cli::command();
        assert!(
            cli.find_subcommand("config").is_some(),
            "`oxigaf config` is not registered"
        );
        assert!(
            cli.find_subcommand("config-cmd").is_some(),
            "the legacy `oxigaf config-cmd` alias was dropped"
        );
        for spelling in ["config", "config-cmd"] {
            let parsed = Cli::try_parse_from(["oxigaf", spelling, "validate", "oxigaf.toml"]);
            assert!(
                parsed.is_ok(),
                "`oxigaf {spelling} validate` did not parse: {:?}",
                parsed.err().map(|e| e.to_string())
            );
        }
    }

    /// Every suggestion string the error taxonomy prints must name a
    /// subcommand the parser actually accepts — the defect that made this
    /// pair drift apart in the first place.
    #[test]
    fn error_suggestions_name_real_subcommands() {
        let cli = Cli::command();
        let suggestions = [
            crate::error::CliError::ConfigValidationError {
                reason: "test".to_string(),
            },
            crate::error::CliError::FlameModelInvalid {
                path: PathBuf::from("flame"),
                reason: "test".to_string(),
            },
            crate::error::CliError::GpuNotAvailable {
                backend: "any".to_string(),
                fallback: None,
            },
        ];
        let mut checked = 0usize;
        for error in suggestions {
            let Some(text) = error.suggestion() else {
                continue;
            };
            // The three messages above name `config`, `setup` and `doctor`.
            for name in ["config", "setup", "doctor"] {
                if text.contains(&format!("oxigaf {name}")) {
                    assert!(
                        cli.find_subcommand(name).is_some(),
                        "suggestion names `oxigaf {name}`, which is not a subcommand"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(
            checked, 3,
            "expected one command reference in each of the three suggestions"
        );
    }

    /// `--iterations 0` leaves the timing vector empty, which makes every
    /// reported statistic undefined. It is refused at parse time rather than
    /// after warmup has already run.
    #[test]
    fn benchmark_iterations_must_be_at_least_one() {
        assert!(Cli::try_parse_from(["oxigaf", "benchmark", "--iterations", "0"]).is_err());
        assert!(Cli::try_parse_from(["oxigaf", "benchmark", "--iterations", "1"]).is_ok());
    }

    /// Parse `train --profile <name>` and return the accepted value.
    fn parse_profile(name: &str) -> Result<Option<String>, clap::Error> {
        Cli::try_parse_from([
            "oxigaf",
            "train",
            "-i",
            "in.mp4",
            "-o",
            "out",
            "--flame-model",
            "flame",
            "--profile",
            name,
        ])
        .map(|cli| match cli.command {
            Command::Train(args) => args.profile,
            _ => None,
        })
    }

    /// Every preset [`crate::commands::preset::resolve`] can resolve has to
    /// survive the parser, or `--profile` would reject a name the crate
    /// documents and ships a preset for.
    #[test]
    fn every_preset_name_is_accepted_by_the_profile_parser() {
        for preset in TrainingPresetName::all() {
            let name = preset.as_str();
            assert_eq!(
                parse_profile(name).ok().flatten().as_deref(),
                Some(name),
                "--profile {name} was rejected by the parser"
            );
        }
    }

    /// The aliases `TrainingPresetName::from_str` accepts must keep working
    /// even though shell completion only offers the canonical spellings.
    #[test]
    fn profile_aliases_and_case_variants_are_accepted() {
        for alias in [
            "fast",
            "default",
            "high",
            "debug",
            "prod",
            "headshot",
            "animation",
            "Quick",
            "BALANCED",
        ] {
            assert!(
                parse_profile(alias).is_ok(),
                "--profile {alias} was rejected by the parser"
            );
        }
    }

    /// Regression: the help text used to advertise "dev, prod, or custom",
    /// but only `prod` ever resolved — the other two failed several seconds
    /// into the run, after the configuration had loaded. An unknown name is
    /// now refused at parse time.
    #[test]
    fn unknown_profile_is_refused_at_parse_time() {
        for name in ["dev", "custom", "ultra"] {
            assert!(
                parse_profile(name).is_err(),
                "--profile {name} should not parse"
            );
        }
    }

    /// Wiring regression: `ExportFormat::All` (`--format all`) must parse
    /// and reach `cmd_export`'s dispatch, not merely exist as a documented
    /// variant while `export::export_all_formats_parallel` stayed
    /// unreachable from the binary the way it used to.
    #[test]
    fn export_format_all_parses() {
        let format = Cli::try_parse_from([
            "oxigaf",
            "export",
            "-m",
            "model.ply",
            "-o",
            "out_dir",
            "--format",
            "all",
        ])
        .map(|cli| match cli.command {
            Command::Export(args) => Some(args.format),
            _ => None,
        });
        assert!(
            matches!(format, Ok(Some(ExportFormat::All))),
            "`oxigaf export --format all` did not parse to ExportFormat::All"
        );
    }
}
