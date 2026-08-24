//! Dry-run validation utilities.
//!
//! Provides validation and reporting for dry-run mode, which validates
//! inputs, checks permissions, verifies GPU availability, and estimates
//! resources without executing any modifications.

use std::path::Path;

use anyhow::{Context, Result};

use crate::output;

/// Dry-run validation result.
#[derive(Debug, Default)]
pub struct DryRunReport {
    pub would_create: Vec<String>,
    pub would_modify: Vec<String>,
    pub would_delete: Vec<String>,
    pub resource_estimates: ResourceEstimates,
}

/// Resource estimates for dry-run reporting.
#[derive(Debug, Default)]
pub struct ResourceEstimates {
    pub estimated_duration_sec: Option<u64>,
    pub estimated_vram_mb: Option<u64>,
    pub estimated_disk_mb: Option<u64>,
}

/// Estimate training resource requirements from real configuration values.
///
/// Every `--dry-run` call site used to hardcode the same fixed literals
/// (`estimated_duration_sec = Some(3600)`, `estimated_vram_mb = Some(4096)`)
/// regardless of scene size or iteration count, so a 100-iteration toy run
/// and a 30,000-iteration 2M-Gaussian run both reported "~60 minutes,
/// ~4096 MB". This estimates from the actual configuration instead.
///
/// `estimated_vram_mb` accounts for Gaussian parameters (sized by
/// `sh_degree`), gradients, two Adam optimizer moments, and RGBA+depth
/// render buffers at `image_size`. This mirrors the per-Gaussian formula
/// used by `memory_estimator::estimate_training_memory`, duplicated here in
/// self-contained form: `dry_run` compiles only into the `oxigaf` binary
/// crate's own module tree (see `main.rs`'s `mod` list), which does not
/// include `memory_estimator` (a `lib.rs`-only module), so it cannot be
/// called from here directly.
///
/// `estimated_duration_sec` is only populated when `iters_per_sec` is
/// supplied (e.g. a measured rate from a resumed checkpoint or a prior run
/// of the same scene); with `None`, no throughput is fabricated and it
/// stays `None`, which `DryRunReport::print_report` renders honestly as
/// "(not estimated)" instead of a made-up number.
///
/// Not yet called by any command: main.rs's `cmd_train` dry-run path
/// (which owns `total_iterations`/`sh_degree`/`image_size` from the loaded
/// [`crate::config::ProjectConfig`]) needs to be updated to call this
/// instead of assigning fixed literals -- tracked as a followup, since
/// `main.rs` is outside this fix's file ownership.
#[allow(dead_code)]
pub fn estimate_training_resources(
    num_gaussians: usize,
    sh_degree: u32,
    image_size: u32,
    total_iterations: u32,
    iters_per_sec: Option<f32>,
) -> ResourceEstimates {
    let degree = u64::from(sh_degree.min(3));
    let sh_coeffs = (degree + 1) * (degree + 1);
    // positions(3) + rotations(4) + scales(3) + opacity(1) + SH coeffs,
    // each an f32 (4 bytes); x4 for parameters + gradients + two Adam
    // optimizer moments.
    let bytes_per_gaussian = (3 + 4 + 3 + 1 + sh_coeffs) * 4 * 4;
    let param_bytes = (num_gaussians as u64).saturating_mul(bytes_per_gaussian);

    // RGBA f32 framebuffer + f32 depth buffer at the training resolution.
    let pixels = u64::from(image_size).saturating_mul(u64::from(image_size));
    let render_bytes = pixels
        .saturating_mul(4 * 4)
        .saturating_add(pixels.saturating_mul(4));

    let total_vram_bytes = param_bytes.saturating_add(render_bytes);
    let estimated_vram_mb = Some(total_vram_bytes / (1024 * 1024));

    let estimated_duration_sec = iters_per_sec
        .filter(|rate| *rate > 0.0)
        .map(|rate| (f64::from(total_iterations) / f64::from(rate)).ceil() as u64);

    ResourceEstimates {
        estimated_duration_sec,
        estimated_vram_mb,
        estimated_disk_mb: estimate_export_disk_mb(num_gaussians, sh_degree),
    }
}

/// Estimate on-disk size in MB for exporting `num_gaussians` Gaussians at
/// `sh_degree` (PLY/safetensors/glTF all store roughly the same
/// per-Gaussian float payload; format-specific framing overhead is
/// negligible at any real Gaussian count this runs on).
///
/// Not yet called by any command -- see [`estimate_training_resources`]'s
/// doc for why, and the corresponding followup for `cmd_export`'s dry-run
/// path in main.rs (which currently hardcodes `estimated_disk_mb = Some(100)`
/// regardless of how many Gaussians are being exported).
#[allow(dead_code)]
pub fn estimate_export_disk_mb(num_gaussians: usize, sh_degree: u32) -> Option<u64> {
    let degree = u64::from(sh_degree.min(3));
    let sh_coeffs = (degree + 1) * (degree + 1);
    let bytes_per_gaussian = (3 + 4 + 3 + 1 + sh_coeffs) * 4;
    Some((num_gaussians as u64).saturating_mul(bytes_per_gaussian) / (1024 * 1024))
}

impl DryRunReport {
    /// Create a new empty dry-run report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file or directory that would be created.
    pub fn add_create(&mut self, path: impl Into<String>) {
        self.would_create.push(path.into());
    }

    /// Add a file or directory that would be modified.
    pub fn add_modify(&mut self, path: impl Into<String>) {
        self.would_modify.push(path.into());
    }

    /// Add a file or directory that would be deleted.
    #[allow(dead_code)]
    pub fn add_delete(&mut self, path: impl Into<String>) {
        self.would_delete.push(path.into());
    }

    /// Print the dry-run report to stdout.
    pub fn print_report(&self) {
        println!("\n{}", "=".repeat(60));
        println!("DRY RUN - No changes will be made");
        println!("{}", "=".repeat(60));

        if !self.would_create.is_empty() {
            println!("\nWould create:");
            for path in &self.would_create {
                println!("  + {}", path);
            }
        }

        if !self.would_modify.is_empty() {
            println!("\nWould modify:");
            for path in &self.would_modify {
                println!("  ~ {}", path);
            }
        }

        if !self.would_delete.is_empty() {
            println!("\nWould delete:");
            for path in &self.would_delete {
                println!("  - {}", path);
            }
        }

        println!("\nResource estimates:");
        if let Some(duration) = self.resource_estimates.estimated_duration_sec {
            println!("  Duration: ~{} minutes", duration / 60);
        } else {
            println!("  Duration: (not estimated)");
        }
        if let Some(vram) = self.resource_estimates.estimated_vram_mb {
            println!("  VRAM: ~{} MB", vram);
        } else {
            println!("  VRAM: (not estimated)");
        }
        if let Some(disk) = self.resource_estimates.estimated_disk_mb {
            println!("  Disk: ~{} MB", disk);
        } else {
            println!("  Disk: (not estimated)");
        }

        println!("\n{}", "=".repeat(60));
        println!("To execute, run without --dry-run");
        println!("{}", "=".repeat(60));
    }
}

/// Check if a path is genuinely writable by the current process.
///
/// For an existing file, opens it in append mode (never truncating or
/// otherwise modifying its content) -- this is a real permission test,
/// unlike checking the readonly bit, which on Unix only reports whether
/// *any* write bit is set on the mode (a root-owned `0755` directory or
/// file reports "not readonly" even though the current, non-root user
/// cannot write to it).
///
/// For an existing directory, or for a non-existing path (checking its
/// parent directory), actually creates and immediately removes a
/// uniquely-named probe file, rather than merely checking that the
/// directory's metadata is readable -- metadata being readable says
/// nothing about whether this process can create files there.
pub fn check_writable(path: &Path) -> Result<()> {
    if path.exists() {
        let metadata = path
            .metadata()
            .with_context(|| format!("Cannot read metadata for {}", path.display()))?;

        if metadata.is_dir() {
            probe_write_in_dir(path)
        } else {
            std::fs::OpenOptions::new()
                .append(true)
                .open(path)
                .with_context(|| format!("Path is not writable: {}", path.display()))?;
            Ok(())
        }
    } else {
        match path.parent() {
            // `Path::parent()` returns `Some("")` (an empty path, not
            // `None`) for a bare relative filename like "out.ply" with no
            // directory component -- treat that the same as "no parent
            // component at all" and probe the current directory.
            Some(parent) if !parent.as_os_str().is_empty() => {
                if !parent.exists() {
                    anyhow::bail!("Parent directory does not exist: {}", parent.display());
                }
                probe_write_in_dir(parent)
            }
            _ => probe_write_in_dir(Path::new(".")),
        }
    }
}

/// Probe real write permission in `dir` by creating and immediately
/// removing a uniquely-named file.
fn probe_write_in_dir(dir: &Path) -> Result<()> {
    let unique = {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    };
    let probe_path = dir.join(format!(".oxigaf-writetest-{}-{unique}", std::process::id()));

    std::fs::File::create(&probe_path)
        .with_context(|| format!("Directory is not writable: {}", dir.display()))?;
    // Best-effort cleanup: failing to remove the probe file is not itself
    // evidence the directory is unwritable -- creating it already
    // succeeded -- so this is not propagated as an error.
    std::fs::remove_file(&probe_path).ok();
    Ok(())
}

/// Check GPU availability via wgpu.
///
/// This function verifies that a GPU adapter can be requested,
/// which is necessary for training and rendering operations.
pub fn check_gpu() -> Result<()> {
    output::info("Would verify GPU availability");

    // In dry-run mode, we perform a lightweight check
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter_result =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }));

    match adapter_result {
        Ok(adapter) => {
            let info = adapter.get_info();
            output::success(&format!(
                "GPU available: {} ({:?})",
                info.name, info.backend
            ));
            Ok(())
        }
        Err(e) => {
            anyhow::bail!(
                "No GPU adapter found: {}. A GPU is required for training and rendering.",
                e
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn test_dry_run_report_empty() {
        let report = DryRunReport::new();
        assert!(report.would_create.is_empty());
        assert!(report.would_modify.is_empty());
        assert!(report.would_delete.is_empty());
    }

    #[test]
    fn test_dry_run_report_add_operations() {
        let mut report = DryRunReport::new();
        report.add_create("file1.txt");
        report.add_modify("file2.txt");
        report.add_delete("file3.txt");

        assert_eq!(report.would_create.len(), 1);
        assert_eq!(report.would_modify.len(), 1);
        assert_eq!(report.would_delete.len(), 1);
    }

    // -----------------------------------------------------------------------
    // estimate_training_resources / estimate_export_disk_mb
    //
    // Regression coverage for: every dry-run call site hardcoded the same
    // fixed literals regardless of scene size or iteration count.
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_training_resources_scales_with_gaussian_count() {
        let small = estimate_training_resources(1_000, 3, 512, 15_000, None);
        let large = estimate_training_resources(2_000_000, 3, 512, 15_000, None);
        assert!(
            large.estimated_vram_mb.unwrap() > small.estimated_vram_mb.unwrap() * 100,
            "a 2000x larger scene must not report roughly the same VRAM"
        );
    }

    #[test]
    fn test_estimate_training_resources_scales_with_sh_degree() {
        let low = estimate_training_resources(500_000, 0, 512, 15_000, None);
        let high = estimate_training_resources(500_000, 3, 512, 15_000, None);
        assert!(high.estimated_vram_mb.unwrap() > low.estimated_vram_mb.unwrap());
    }

    #[test]
    fn test_estimate_training_resources_duration_none_without_rate() {
        let est = estimate_training_resources(500_000, 3, 512, 30_000, None);
        assert!(
            est.estimated_duration_sec.is_none(),
            "no throughput was supplied, so no duration should be fabricated"
        );
    }

    #[test]
    fn test_estimate_training_resources_duration_scales_with_iterations() {
        let short = estimate_training_resources(500_000, 3, 512, 100, Some(10.0));
        let long = estimate_training_resources(500_000, 3, 512, 30_000, Some(10.0));
        assert_eq!(short.estimated_duration_sec, Some(10)); // 100 / 10.0
        assert_eq!(long.estimated_duration_sec, Some(3_000)); // 30000 / 10.0
        assert!(long.estimated_duration_sec > short.estimated_duration_sec);
    }

    #[test]
    fn test_estimate_training_resources_never_reports_the_old_fixed_literals() {
        // The bug this fixes: every call site reported the exact same
        // numbers (3600s / 4096MB) no matter the configuration. A tiny toy
        // run must not coincidentally land on those.
        let toy = estimate_training_resources(100, 1, 256, 100, Some(5.0));
        assert_ne!(toy.estimated_vram_mb, Some(4096));
        assert_ne!(toy.estimated_duration_sec, Some(3600));
    }

    #[test]
    fn test_estimate_export_disk_mb_scales_with_gaussian_count() {
        let small = estimate_export_disk_mb(50_000, 3).unwrap();
        let large = estimate_export_disk_mb(1_000_000, 3).unwrap(); // 20x more Gaussians
        assert!(
            large > small * 15,
            "20x more Gaussians should give a proportionally larger estimate \
             (small={small}, large={large})"
        );
    }

    #[test]
    fn test_estimate_export_disk_mb_scales_with_sh_degree() {
        let low = estimate_export_disk_mb(1_000_000, 0).unwrap();
        let high = estimate_export_disk_mb(1_000_000, 3).unwrap();
        assert!(
            high > low,
            "higher SH degree must report more disk usage (low={low}, high={high})"
        );
    }

    #[test]
    fn test_check_writable_existing_dir() {
        let temp_dir = env::temp_dir();
        assert!(check_writable(&temp_dir).is_ok());
    }

    #[test]
    fn test_check_writable_non_existing_path() {
        let temp_dir = env::temp_dir();
        let non_existing = temp_dir.join("oxigaf_test_nonexisting_file.tmp");
        assert!(check_writable(&non_existing).is_ok());
    }

    #[test]
    fn test_check_writable_invalid_parent() {
        let invalid_path = Path::new("/nonexistent_parent_dir_oxigaf/file.tmp");
        assert!(check_writable(invalid_path).is_err());
    }

    #[test]
    fn test_check_writable_probe_leaves_no_trace() {
        // Regression coverage for: the old check only ever read metadata,
        // never actually tested write access. This confirms the new probe
        // both succeeds against a genuinely writable directory *and*
        // cleans up after itself (no leftover `.oxigaf-writetest-*` file).
        let dir = env::temp_dir().join("oxigaf_test_writable_probe_dir");
        fs::create_dir_all(&dir).expect("create test dir");

        assert!(check_writable(&dir).is_ok());

        let leftover_count = fs::read_dir(&dir)
            .map(|entries| entries.count())
            .unwrap_or(0);
        assert_eq!(leftover_count, 0, "write probe file must be cleaned up");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_check_writable_existing_writable_file() {
        let path = env::temp_dir().join("oxigaf_test_writable_existing_file.tmp");
        fs::write(&path, b"content").expect("write test file");
        assert!(check_writable(&path).is_ok());
        // The append-mode probe must not have altered the file's content.
        assert_eq!(fs::read(&path).expect("read back"), b"content");
        fs::remove_file(&path).ok();
    }

    // Regression coverage for: `metadata.permissions().readonly()` on Unix
    // only reports whether *any* write bit is set, which is a real (if
    // narrower) proxy for "not writable" in the common case of a fully
    // read-only file -- confirm the rewritten check, which opens the file
    // for real rather than inspecting the mode bits, still catches this.
    #[cfg(unix)]
    #[test]
    fn test_check_writable_detects_readonly_file_via_real_open() {
        use std::os::unix::fs::PermissionsExt;

        let path = env::temp_dir().join("oxigaf_test_readonly_probe.tmp");
        fs::write(&path, b"content").expect("write test file");
        let mut perms = fs::metadata(&path).expect("metadata").permissions();
        perms.set_mode(0o444); // read-only for everyone, including the owner
        fs::set_permissions(&path, perms).expect("set permissions");

        // Root (common in containerised CI) bypasses Unix permission bits
        // entirely, so a 0o444 file may genuinely still be writable there.
        // Only assert the negative outcome when a raw, independent open
        // attempt confirms this process is not bypassing permissions --
        // otherwise the assertion below would be a false failure under root.
        let raw_probe_confirms_non_root = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .is_err();

        let result = check_writable(&path);

        // Restore write permission before cleanup regardless of outcome
        // (some platforms refuse to remove a read-only file).
        if let Ok(meta) = fs::metadata(&path) {
            let mut perms = meta.permissions();
            perms.set_mode(0o644);
            fs::set_permissions(&path, perms).ok();
        }
        fs::remove_file(&path).ok();

        if raw_probe_confirms_non_root {
            assert!(
                result.is_err(),
                "a 0o444 file must be reported as not writable"
            );
        }
    }

    #[test]
    fn test_check_writable_bare_relative_filename_uses_cwd() {
        // `Path::parent()` returns `Some("")` (not `None`) for a bare
        // filename with no directory component; this must not be
        // misreported as "parent directory does not exist".
        let bare = Path::new("oxigaf_bare_filename_probe.tmp");
        assert!(
            !bare.exists(),
            "test assumes this file does not exist in cwd"
        );
        assert!(check_writable(bare).is_ok());
    }
}
