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

/// Check if a path is writable.
///
/// For existing paths, checks if the file/directory is read-only.
/// For non-existing paths, checks if the parent directory exists and is accessible.
pub fn check_writable(path: &Path) -> Result<()> {
    if path.exists() {
        // Check if we can modify existing file/dir
        let metadata = path
            .metadata()
            .with_context(|| format!("Cannot read metadata for {}", path.display()))?;

        if metadata.permissions().readonly() {
            anyhow::bail!("Path is read-only: {}", path.display());
        }
    } else {
        // Check if parent directory exists and is writable
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                anyhow::bail!("Parent directory does not exist: {}", parent.display());
            }

            // Try to check if we can create files in parent
            std::fs::metadata(parent)
                .with_context(|| format!("Cannot access parent directory: {}", parent.display()))?;
        }
    }
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
}
