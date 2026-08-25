//! Process-level support shared by the pipeline subcommands.
//!
//! These helpers used to live in `main.rs`. They are here so that
//! (a) `main.rs` stays a thin dispatcher well under the 2000-line ceiling
//! while three wiring stages add subcommands to it, and (b) they are
//! reachable from `tests/`, which cannot see anything declared in a binary
//! target.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cli::{DoctorCheck, LogFormatType, LogRotationStrategy};
use crate::error::CliError;
use crate::verbosity::Verbosity;
use oxigaf::render::gaussian::GaussianModel;

// ---------------------------------------------------------------------------
// Memory tracking
// ---------------------------------------------------------------------------

/// Return the current process peak RSS in megabytes, if the platform exposes
/// it without an FFI dependency.
///
/// On Linux this parses `/proc/self/status` field `VmHWM`, the resident-set
/// high-water mark. (`VmPeak` is peak *virtual* address space — much larger,
/// and not what a user means by "peak memory".) On every other platform this
/// returns `None`.
#[must_use]
pub fn peak_rss_mb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                // Format: "VmHWM:  1234 kB"
                let kb_str = rest.trim().split_whitespace().next()?;
                let kb: u64 = kb_str.parse().ok()?;
                return Some(kb / 1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// Terminal state
// ---------------------------------------------------------------------------

/// Restores the terminal out of raw mode when dropped.
///
/// `--interactive` puts the terminal into raw mode for the keyboard
/// listener. Restoring it only at the end of the happy path is not enough:
/// any `?` in between would leave the user's shell in raw mode. A guard
/// makes the restore unconditional.
pub struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

// ---------------------------------------------------------------------------
// Spherical harmonics
// ---------------------------------------------------------------------------

/// Number of SH float channels stored per Gaussian at `degree`.
#[must_use]
pub fn sh_channels(degree: u32) -> usize {
    let bands = (degree + 1) as usize;
    bands * bands * 3
}

/// Reduce a model's SH degree in place, keeping the low-order coefficients.
///
/// `sh_coeffs` is a flat `[N, C]` array with `C = (degree + 1)² × 3` and the
/// DC term in the first three channels of each row (the layout
/// [`crate::export::export_ply`] writes as `f_dc_*` followed by `f_rest_*`),
/// so the retained coefficients are a prefix of every row.
///
/// Returns `true` when the model was actually changed. Requesting a degree
/// at or above the model's own is a no-op: this only ever downsamples.
pub fn downsample_sh(model: &mut GaussianModel, target_degree: u32) -> bool {
    let target = target_degree.min(3);
    if target >= model.sh_degree {
        return false;
    }
    let count = model.gaussians.len();
    let old_channels = sh_channels(model.sh_degree);
    let new_channels = sh_channels(target);
    if count == 0 || model.sh_coeffs.len() < count * old_channels {
        // Malformed or empty coefficient block: leave it alone rather than
        // slicing out of range.
        return false;
    }

    let mut trimmed = Vec::with_capacity(count * new_channels);
    for index in 0..count {
        let base = index * old_channels;
        trimmed.extend_from_slice(&model.sh_coeffs[base..base + new_channels]);
    }
    model.sh_coeffs = trimmed;
    model.sh_degree = target;
    true
}

// ---------------------------------------------------------------------------
// Cache location
// ---------------------------------------------------------------------------

/// The one cache directory every subcommand agrees on.
///
/// `setup`, `doctor` and `cache` used to compute this three different ways
/// (`~/.cache/oxigaf`, `$HOME/.cache/oxigaf`, `dirs::cache_dir()`), so on
/// macOS `oxigaf setup` populated `~/.cache/oxigaf` while `oxigaf cache list`
/// looked in `~/Library/Caches/oxigaf` and reported an empty cache.
///
/// `OXIGAF_CACHE_DIR` overrides the platform location for all of them.
///
/// # Errors
///
/// Returns an error when the platform cache directory cannot be determined
/// and `OXIGAF_CACHE_DIR` is unset.
pub fn default_cache_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("OXIGAF_CACHE_DIR") {
        if !dir.trim().is_empty() {
            return Ok(crate::config::expand_tilde(Path::new(&dir)));
        }
    }
    dirs::cache_dir()
        .map(|p| p.join("oxigaf"))
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))
}

/// Resolve an explicit `--cache-dir` against the shared default.
///
/// # Errors
///
/// Propagates [`default_cache_dir`] when no explicit path is given.
pub fn resolve_cache_dir(explicit: Option<&PathBuf>) -> Result<PathBuf> {
    match explicit {
        Some(path) => Ok(crate::config::expand_tilde(path)),
        None => default_cache_dir(),
    }
}

/// Filter expected asset paths by a comma-separated `--only` list matched
/// against file names.
#[must_use]
pub fn select_assets<'a>(paths: &'a [PathBuf], only: Option<&str>) -> Vec<&'a PathBuf> {
    let Some(filter) = only else {
        return paths.iter().collect();
    };
    let wanted: Vec<&str> = filter
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if wanted.is_empty() {
        return paths.iter().collect();
    }
    paths
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|name| wanted.iter().any(|w| name.contains(w)))
                .unwrap_or(false)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Logging initialization
// ---------------------------------------------------------------------------

/// Initialize tracing with the specified verbosity and optional file logging.
///
/// # Errors
///
/// Propagates subscriber-installation and log-directory failures.
pub fn init_logging(
    verbosity: Verbosity,
    log_file: Option<PathBuf>,
    rotation: LogRotationStrategy,
    max_files: usize,
    format: LogFormatType,
) -> Result<()> {
    use crate::log_rotation::{LogConfig, LogFormat, LogRotation};

    let log_config = LogConfig {
        file_path: log_file.clone(),
        rotation: match rotation {
            LogRotationStrategy::Never => LogRotation::Never,
            LogRotationStrategy::Hourly => LogRotation::Hourly,
            LogRotationStrategy::Daily => LogRotation::Daily,
        },
        max_files,
        format: match format {
            LogFormatType::Json => LogFormat::Json,
            LogFormatType::Pretty => LogFormat::Pretty,
            LogFormatType::Compact => LogFormat::Compact,
        },
    };

    crate::log_rotation::init_logging_with_file(log_config, verbosity)?;
    cleanup_logs(log_file.as_deref(), max_files);
    Ok(())
}

/// Initialize tracing that writes **only** to the log file.
///
/// `--json` promises that stdout carries nothing but one JSON document, and
/// the shared initializer always attaches a stdout layer. Rather than
/// dropping file logging entirely (which silently voided `--log-file`,
/// `--log-rotation` and `--log-format` whenever `--json` was passed), the
/// file layer is built on its own here.
///
/// # Errors
///
/// Returns an error when the log directory or appender cannot be created, or
/// when a subscriber is already installed.
pub fn init_file_only_logging(
    verbosity: Verbosity,
    log_file: &Path,
    rotation_strategy: LogRotationStrategy,
    max_files: usize,
    format: LogFormatType,
) -> Result<()> {
    use tracing_appender::rolling::{RollingFileAppender, Rotation};
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::{
        fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
    };

    let dir = match log_file.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create log directory: {}", dir.display()))?;

    let filename = log_file
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid log filename: {}", log_file.display()))?;

    let rotation = match rotation_strategy {
        LogRotationStrategy::Never => Rotation::NEVER,
        LogRotationStrategy::Hourly => Rotation::HOURLY,
        LogRotationStrategy::Daily => Rotation::DAILY,
    };
    let appender = RollingFileAppender::builder()
        .rotation(rotation)
        .filename_prefix(filename)
        .build(dir)
        .with_context(|| format!("Failed to create log appender in {}", dir.display()))?;

    let level = LevelFilter::from_level(verbosity.tracing_level());
    let env_filter = EnvFilter::from_default_env().add_directive(level.into());

    let layer = match format {
        LogFormatType::Json => fmt::layer()
            .json()
            .with_writer(appender)
            .with_ansi(false)
            .with_filter(env_filter)
            .boxed(),
        LogFormatType::Pretty => fmt::layer()
            .pretty()
            .with_writer(appender)
            .with_ansi(false)
            .with_filter(env_filter)
            .boxed(),
        LogFormatType::Compact => fmt::layer()
            .compact()
            .with_writer(appender)
            .with_ansi(false)
            .with_filter(env_filter)
            .boxed(),
    };

    tracing_subscriber::registry()
        .with(layer)
        .try_init()
        .map_err(|e| anyhow::anyhow!("Failed to initialize tracing subscriber: {e}"))?;

    cleanup_logs(Some(log_file), max_files);
    Ok(())
}

/// Delete rotated log files beyond `max_files` (best-effort).
pub fn cleanup_logs(log_file: Option<&Path>, max_files: usize) {
    let Some(path) = log_file else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if let Some(prefix) = path.file_stem().and_then(|s| s.to_str()) {
        let _ = crate::log_rotation::cleanup_old_logs(parent, prefix, max_files);
    }
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Recover the typed [`CliError`] behind an `anyhow::Error`.
///
/// Handlers return `anyhow::Result`, so a `CliError` raised deep inside gets
/// boxed on the way out. `err.into()` would blindly re-wrap it as
/// [`CliError::Other`], collapsing the whole exit-code taxonomy onto `1`;
/// downcasting first preserves the intended code.
#[must_use]
pub fn to_cli_error(err: anyhow::Error) -> CliError {
    match err.downcast::<CliError>() {
        Ok(cli_err) => cli_err,
        Err(other) => match other.downcast::<std::io::Error>() {
            Ok(io_err) => {
                let context = io_err.to_string();
                CliError::IoError {
                    context,
                    source: io_err,
                }
            }
            Err(rest) => CliError::Other(rest),
        },
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Whether `doctor` should run `check`, given the `--check` selector.
#[must_use]
pub fn doctor_runs(selector: Option<&DoctorCheck>, check: DoctorCheck) -> bool {
    match selector {
        None | Some(DoctorCheck::All) => true,
        Some(selected) => std::mem::discriminant(selected) == std::mem::discriminant(&check),
    }
}

/// Check GPU availability via wgpu.
///
/// # Errors
///
/// Returns an error when no adapter can be acquired.
pub fn check_gpu() -> Result<String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .map_err(|e| anyhow::anyhow!("No GPU adapter found: {e}"))?;

    let info = adapter.get_info();
    Ok(format!("{} ({:?})", info.name, info.backend))
}

/// Check a FLAME model directory for the required `.npy` files.
///
/// # Errors
///
/// Returns an error when the directory is missing or incomplete.
pub fn check_flame_model(path: &Path) -> Result<String> {
    let expanded = crate::config::expand_tilde(path);
    if !expanded.exists() {
        anyhow::bail!("Directory does not exist");
    }

    let required_files = ["v_template.npy", "shapedirs.npy", "faces.npy"];
    let mut found = 0;
    for file in &required_files {
        if expanded.join(file).exists() {
            found += 1;
        }
    }

    if found == required_files.len() {
        Ok(format!(
            "{} (all {} required files present)",
            expanded.display(),
            found
        ))
    } else {
        anyhow::bail!("{} of {} required files found", found, required_files.len())
    }
}

/// Summarise how much of the expected asset set is cached.
///
/// # Errors
///
/// Returns an error when the cache directory does not exist.
pub fn check_cache(path: &Path) -> Result<String> {
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "Cache directory does not exist. Run `oxigaf setup` to create it."
        ));
    }

    let assets = crate::assets::expected_asset_paths(path);
    let cached = assets.iter().filter(|p| p.exists()).count();

    Ok(format!(
        "{} ({}/{} assets cached)",
        path.display(),
        cached,
        assets.len()
    ))
}

/// Version information reported by `oxigaf doctor`.
pub struct VersionInfo {
    /// This crate's version.
    pub oxigaf: String,
    /// The toolchain version, or an explicit "unknown" marker.
    pub rust: String,
    /// `os arch`.
    pub platform: String,
}

/// Collect version information.
///
/// The Rust version is read from the toolchain on `PATH` at run time. The
/// crate has no build script, so the `RUSTC_VERSION` compile-time variable
/// this used to consult was never set and the field always printed
/// "unknown"; it is still honoured first for anyone who does set it.
#[must_use]
pub fn get_version_info() -> VersionInfo {
    let rust = option_env!("RUSTC_VERSION")
        .map(str::to_string)
        .or_else(rustc_version_from_path)
        .unwrap_or_else(|| "unknown (rustc not found on PATH)".to_string());

    VersionInfo {
        oxigaf: env!("CARGO_PKG_VERSION").to_string(),
        rust,
        platform: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
    }
}

/// Ask the `rustc` on `PATH` for its version string.
fn rustc_version_from_path() -> Option<String> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let probe = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()?;
    if !probe.status.success() {
        return None;
    }
    let text = String::from_utf8(probe.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Report the free space on the filesystem holding `path`, in megabytes.
///
/// Implemented by asking the platform's `df` for the POSIX-format listing,
/// which keeps the crate free of a `libc`/`-sys` dependency (COOLJAPAN pure
/// Rust policy) while reporting a real number instead of the previous
/// "space check not implemented" placeholder.
///
/// # Errors
///
/// Returns an error when the path has no existing ancestor, when `df` is
/// unavailable or fails, or on a platform without it.
pub fn available_disk_mb(path: &Path) -> Result<u64> {
    // `df` needs an existing path; fall back to the nearest existing ancestor.
    let mut probe = path;
    while !probe.exists() {
        probe = probe
            .parent()
            .ok_or_else(|| anyhow::anyhow!("No existing ancestor for {}", path.display()))?;
    }

    #[cfg(unix)]
    {
        let df = std::process::Command::new("df")
            .arg("-Pk")
            .arg(probe)
            .output()
            .context("Failed to run `df`")?;
        if !df.status.success() {
            anyhow::bail!("`df` exited with {}", df.status);
        }
        let text = String::from_utf8_lossy(&df.stdout);
        parse_df_available_kb(&text)
            .map(|kb| kb / 1024)
            .ok_or_else(|| anyhow::anyhow!("Could not parse `df -Pk` output"))
    }
    #[cfg(not(unix))]
    {
        let _ = probe;
        anyhow::bail!("Free-space reporting is not implemented on this platform")
    }
}

/// Extract the "Available" column (1K blocks) from POSIX `df -Pk` output.
///
/// POSIX format guarantees one record per filesystem on a single line:
/// `Filesystem 1024-blocks Used Available Capacity Mounted-on`.
#[must_use]
pub fn parse_df_available_kb(text: &str) -> Option<u64> {
    let line = text.lines().nth(1)?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    fields.get(3)?.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{EXIT_GPU_ERROR, EXIT_IO_ERROR};
    use oxigaf::render::gaussian::GaussianAttributes;

    fn model_with_degree(degree: u32, count: usize) -> GaussianModel {
        let channels = sh_channels(degree);
        GaussianModel {
            gaussians: (0..count)
                .map(|i| GaussianAttributes {
                    position: [i as f32, 0.0, 0.0],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-2.0, -2.0, -2.0],
                    opacity: 0.0,
                })
                .collect(),
            sh_coeffs: (0..count * channels).map(|i| i as f32).collect(),
            sh_degree: degree,
            face_indices: Vec::new(),
            barycentric: Vec::new(),
            local_offsets: Vec::new(),
            is_rigid: Vec::new(),
        }
    }

    #[test]
    fn sh_channels_matches_the_ply_layout() {
        assert_eq!(sh_channels(0), 3);
        assert_eq!(sh_channels(1), 12);
        assert_eq!(sh_channels(2), 27);
        assert_eq!(sh_channels(3), 48);
    }

    #[test]
    fn downsample_sh_keeps_the_low_order_prefix_of_every_row() {
        let mut model = model_with_degree(3, 2);
        assert!(downsample_sh(&mut model, 1));
        assert_eq!(model.sh_degree, 1);
        assert_eq!(model.sh_coeffs.len(), 2 * sh_channels(1));
        // Row 0 keeps coefficients 0..12 of the original 48-wide row.
        assert_eq!(model.sh_coeffs[0], 0.0);
        assert_eq!(model.sh_coeffs[11], 11.0);
        // Row 1 starts at original offset 48 (the second row's own prefix).
        assert_eq!(model.sh_coeffs[12], 48.0);
    }

    #[test]
    fn downsample_sh_never_upsamples() {
        let mut model = model_with_degree(1, 3);
        assert!(!downsample_sh(&mut model, 3));
        assert_eq!(model.sh_degree, 1);
        assert_eq!(model.sh_coeffs.len(), 3 * sh_channels(1));
    }

    #[test]
    fn doctor_check_selector_filters_individual_checks() {
        assert!(doctor_runs(None, DoctorCheck::Gpu));
        assert!(doctor_runs(Some(&DoctorCheck::All), DoctorCheck::Cache));
        assert!(doctor_runs(Some(&DoctorCheck::Gpu), DoctorCheck::Gpu));
        assert!(!doctor_runs(Some(&DoctorCheck::Gpu), DoctorCheck::Cache));
    }

    #[test]
    fn df_parser_reads_the_available_column() {
        let sample = "Filesystem 1024-blocks      Used Available Capacity Mounted on\n\
                      /dev/disk1s1  976490576 512000000 400000000      57% /\n";
        assert_eq!(parse_df_available_kb(sample), Some(400_000_000));
        assert_eq!(parse_df_available_kb("only a header\n"), None);
    }

    #[test]
    fn cli_error_survives_the_anyhow_round_trip() {
        let err: anyhow::Error = CliError::GpuNotAvailable {
            backend: "vulkan".to_string(),
            fallback: None,
        }
        .into();
        assert_eq!(
            to_cli_error(err).exit_code(),
            EXIT_GPU_ERROR,
            "the typed exit code must survive being boxed in anyhow"
        );
    }

    #[test]
    fn io_errors_map_onto_the_io_exit_code() {
        let err: anyhow::Error =
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied").into();
        assert_eq!(to_cli_error(err).exit_code(), EXIT_IO_ERROR);
    }

    #[test]
    fn only_filter_selects_matching_asset_names() {
        let paths = vec![
            PathBuf::from("/cache/flame2023.npz"),
            PathBuf::from("/cache/diffusion.safetensors"),
        ];
        assert_eq!(select_assets(&paths, None).len(), 2);
        assert_eq!(select_assets(&paths, Some("flame")).len(), 1);
        assert_eq!(select_assets(&paths, Some("  ")).len(), 2);
        assert_eq!(select_assets(&paths, Some("flame,diffusion")).len(), 2);
    }

    #[test]
    fn explicit_cache_dir_wins_over_the_platform_default() {
        let explicit = std::env::temp_dir().join("oxigaf_cache_explicit");
        assert_eq!(
            resolve_cache_dir(Some(&explicit)).ok(),
            Some(explicit),
            "--cache-dir must be honoured verbatim (after tilde expansion)"
        );
    }

    #[test]
    fn setup_doctor_and_cache_agree_on_the_default_directory() {
        // All three used to compute this differently; the shared helper is
        // the single source of truth now.
        let a = default_cache_dir();
        let b = resolve_cache_dir(None);
        assert_eq!(a.is_ok(), b.is_ok());
        if let (Ok(a), Ok(b)) = (a, b) {
            assert_eq!(a, b);
        }
    }
}
