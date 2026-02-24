//! CLI command definitions (clap derive API).
//!
//! Provides comprehensive command-line interface for OxiGAF with:
//! - Training with progress, early stopping, and checkpointing
//! - Multi-format rendering with quality settings
//! - Export to PLY, safetensors, and glTF 2.0
//! - FLAME model conversion utilities
//! - Performance benchmarking
//! - System diagnostics

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
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
    #[arg(long, global = true)]
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
    /// Train (reconstruct) a 3D Gaussian avatar from monocular video.
    #[command(alias = "reconstruct")]
    Train(TrainArgs),

    /// Render an existing avatar from novel viewpoints.
    Render(RenderArgs),

    /// Export an avatar to standard formats (PLY, glTF, safetensors).
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
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
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
/// # Use environment variables to override config
/// export OXIGAF_TOTAL_ITERATIONS=10000
/// export OXIGAF_POSITION_LR=0.0002
/// oxigaf train -i video.mp4 -o output/ --flame-model ~/.cache/oxigaf/flame2023
///
/// # CLI args have highest priority
/// oxigaf train -i video.mp4 -o output/ --flame-model model/ --max-iterations 5000
/// ```
#[derive(Debug, clap::Args)]
pub struct TrainArgs {
    /// Input video file or directory of extracted frames.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory for the reconstructed avatar.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Path to the converted FLAME model directory (`.npy` files).
    #[arg(long)]
    pub flame_model: PathBuf,

    /// Path to pre-computed per-frame FLAME tracking parameters (JSON).
    #[arg(long)]
    pub flame_params: Option<PathBuf>,

    /// Training configuration TOML file.
    #[arg(short, long, default_value = "oxigaf.toml")]
    pub config: PathBuf,

    /// GPU device index.
    #[arg(long, default_value = "0")]
    pub device: usize,

    /// Resume from a checkpoint file.
    #[arg(long)]
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
    #[arg(long, value_name = "PATH")]
    pub metrics_output: Option<PathBuf>,

    /// Metrics output format.
    #[arg(long, value_enum, default_value = "csv")]
    pub metrics_format: MetricsOutputFormat,

    /// Enable TensorBoard logging.
    #[arg(long)]
    pub tensorboard: bool,

    /// TensorBoard log directory.
    #[arg(long, default_value = "runs")]
    pub tensorboard_dir: PathBuf,

    /// Training profile (dev, prod, or custom).
    #[arg(long)]
    pub profile: Option<String>,
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
    #[arg(short, long)]
    pub model: PathBuf,

    /// Output directory for rendered images.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Render width in pixels.
    #[arg(long, default_value = "512")]
    pub width: u32,

    /// Render height in pixels.
    #[arg(long, default_value = "512")]
    pub height: u32,

    /// Camera trajectory JSON file with azimuth/elevation/distance.
    #[arg(long)]
    pub cameras: Option<PathBuf>,

    /// FLAME parameters for animation (per-frame JSON).
    #[arg(long)]
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
    #[must_use]
    #[allow(dead_code)]
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
    #[arg(short, long)]
    pub model: PathBuf,

    /// Output file path.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Export format.
    #[arg(long, default_value = "ply")]
    pub format: ExportFormat,

    /// Include training metadata in export.
    #[arg(long)]
    pub include_metadata: bool,

    /// Source checkpoint for metadata (optional).
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,

    /// PLY format variant (only for PLY export).
    #[arg(long, default_value = "ascii")]
    pub ply_format: PlyFormat,

    /// SH degree to export (downsample if less than model's degree).
    #[arg(long)]
    pub sh_degree: Option<u32>,

    /// Overwrite existing output file without prompting.
    #[arg(long)]
    pub force: bool,
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
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum PlyFormat {
    /// ASCII format (human-readable, larger).
    #[default]
    Ascii,
    /// Binary little-endian format (compact).
    BinaryLe,
    /// Binary big-endian format.
    BinaryBe,
}

// ---------------------------------------------------------------------------
// Convert Command (new)
// ---------------------------------------------------------------------------

#[derive(Debug, clap::Args)]
pub struct ConvertArgs {
    /// Input FLAME pickle file (.pkl) or NPZ file (.npz).
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory for converted .npy files.
    #[arg(short, long)]
    pub output: PathBuf,

    /// FLAME model version (2020 or 2023).
    #[arg(long, default_value = "2023")]
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

    /// Number of timed iterations.
    #[arg(long, default_value = "10")]
    pub iterations: u32,

    /// Output format for benchmark results.
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Save benchmark report to file.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Model size for synthetic benchmarks.
    #[arg(long, default_value = "medium")]
    pub size: BenchSize,

    /// Path to FLAME model for FLAME benchmarks.
    #[arg(long)]
    pub flame_model: Option<PathBuf>,

    /// Compare results against baseline file.
    #[arg(long)]
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

    /// Path to FLAME model directory to verify.
    #[arg(long)]
    pub flame_model: Option<PathBuf>,

    /// Path to cache directory to check.
    #[arg(long)]
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
    #[arg(long, default_value = "~/.cache/oxigaf")]
    pub cache_dir: PathBuf,

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
