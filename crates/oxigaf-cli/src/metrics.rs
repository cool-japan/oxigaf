//! Training metrics export to CSV and JSON Lines formats.
//!
//! Provides real-time metrics logging for training sessions with support
//! for CSV (human-readable) and JSON Lines (machine-readable) formats.
//!
//! ## Usage
//!
//! ```ignore
//! use oxigaf_cli::metrics::{MetricsWriter, MetricsFormat, TrainingMetrics};
//!
//! let mut writer = MetricsWriter::new(&path, MetricsFormat::Csv)?;
//!
//! let metrics = TrainingMetrics {
//!     iteration: 100,
//!     loss_total: 0.5,
//!     loss_l1: 0.3,
//!     loss_ssim: 0.1,
//!     loss_lpips: Some(0.05),
//!     loss_reg: 0.05,
//!     num_gaussians: 50000,
//!     lr_position: 0.00016,
//!     lr_scaling: 0.005,
//!     lr_rotation: 0.001,
//!     memory_mb: Some(4096),
//!     elapsed_seconds: 120.5,
//! };
//!
//! writer.write_metrics(&metrics)?;
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Training metrics for a single iteration.
///
/// Captures all relevant training statistics including losses, model size,
/// learning rates, resource usage, and timing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Current training iteration.
    pub iteration: u32,
    /// Total combined loss.
    pub loss_total: f32,
    /// L1 photometric loss component.
    pub loss_l1: f32,
    /// SSIM structural similarity loss component.
    pub loss_ssim: f32,
    /// LPIPS perceptual loss component (optional).
    pub loss_lpips: Option<f32>,
    /// Regularization loss component.
    pub loss_reg: f32,
    /// Current number of Gaussians in the model.
    pub num_gaussians: u32,
    /// Position parameter learning rate.
    pub lr_position: f32,
    /// Scaling parameter learning rate.
    pub lr_scaling: f32,
    /// Rotation parameter learning rate.
    pub lr_rotation: f32,
    /// GPU memory usage in megabytes (optional).
    pub memory_mb: Option<u64>,
    /// Elapsed training time in seconds.
    pub elapsed_seconds: f32,
}

/// Output format for metrics export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsFormat {
    /// Comma-separated values format (human-readable, spreadsheet-compatible).
    Csv,
    /// JSON Lines format (one JSON object per line, machine-readable).
    JsonLines,
}

/// Writer for training metrics with real-time flushing.
///
/// Writes metrics to a file in CSV or JSON Lines format with automatic
/// flushing after each write to enable real-time monitoring.
pub struct MetricsWriter {
    format: MetricsFormat,
    writer: BufWriter<File>,
}

impl MetricsWriter {
    /// Create a new metrics writer.
    ///
    /// Creates the output file and writes CSV header if using CSV format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written to.
    pub fn new(path: &Path, format: MetricsFormat) -> Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("Failed to create metrics file: {}", path.display()))?;

        let mut writer = BufWriter::new(file);

        // Write CSV header
        if matches!(format, MetricsFormat::Csv) {
            writeln!(
                writer,
                "iteration,loss_total,loss_l1,loss_ssim,loss_lpips,loss_reg,num_gaussians,lr_position,lr_scaling,lr_rotation,memory_mb,elapsed_seconds"
            )
            .context("Failed to write CSV header")?;
        }

        Ok(Self { format, writer })
    }

    /// Write metrics for a single training iteration.
    ///
    /// Automatically flushes after writing to enable real-time monitoring.
    ///
    /// # Errors
    ///
    /// Returns an error if writing or flushing fails.
    pub fn write_metrics(&mut self, metrics: &TrainingMetrics) -> Result<()> {
        match self.format {
            MetricsFormat::Csv => {
                writeln!(
                    self.writer,
                    "{},{},{},{},{},{},{},{},{},{},{},{}",
                    metrics.iteration,
                    metrics.loss_total,
                    metrics.loss_l1,
                    metrics.loss_ssim,
                    metrics.loss_lpips.unwrap_or(0.0),
                    metrics.loss_reg,
                    metrics.num_gaussians,
                    metrics.lr_position,
                    metrics.lr_scaling,
                    metrics.lr_rotation,
                    metrics.memory_mb.unwrap_or(0),
                    metrics.elapsed_seconds,
                )
                .context("Failed to write CSV metrics line")?;
            }
            MetricsFormat::JsonLines => {
                let json = serde_json::to_string(metrics)
                    .context("Failed to serialize metrics to JSON")?;
                writeln!(self.writer, "{}", json).context("Failed to write JSON Lines metrics")?;
            }
        }

        // Flush periodically for real-time monitoring
        self.writer
            .flush()
            .context("Failed to flush metrics writer")?;
        Ok(())
    }

    /// Manually flush any buffered data to disk.
    ///
    /// Note: This is called automatically after each `write_metrics` call.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails.
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .context("Failed to flush metrics writer")?;
        Ok(())
    }
}

impl Drop for MetricsWriter {
    fn drop(&mut self) {
        // Best-effort flush on drop, ignore errors
        let _ = self.writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    #[allow(clippy::expect_used)]
    fn test_csv_format_header() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("oxigaf_metrics_test_csv_header.csv");
        let _ = fs::remove_file(&path); // Clean up any previous test

        let writer = MetricsWriter::new(&path, MetricsFormat::Csv);
        assert!(writer.is_ok(), "Failed to create CSV writer");

        drop(writer);

        let content = fs::read_to_string(&path).expect("Failed to read metrics file");
        assert!(
            content.contains("iteration,loss_total"),
            "CSV header not found"
        );

        // Clean up
        let _ = fs::remove_file(&path);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_csv_format_data() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("oxigaf_metrics_test_csv_data.csv");
        let _ = fs::remove_file(&path);

        let mut writer =
            MetricsWriter::new(&path, MetricsFormat::Csv).expect("Failed to create CSV writer");

        let metrics = TrainingMetrics {
            iteration: 1,
            loss_total: 1.234,
            loss_l1: 0.5,
            loss_ssim: 0.3,
            loss_lpips: Some(0.1),
            loss_reg: 0.334,
            num_gaussians: 50000,
            lr_position: 0.00016,
            lr_scaling: 0.005,
            lr_rotation: 0.001,
            memory_mb: Some(4096),
            elapsed_seconds: 120.5,
        };

        writer
            .write_metrics(&metrics)
            .expect("Failed to write metrics");
        drop(writer);

        let content = fs::read_to_string(&path).expect("Failed to read metrics file");
        assert!(content.contains("1,1.234"), "CSV data not found");
        assert!(content.contains("50000"), "Gaussian count not found");

        // Clean up
        let _ = fs::remove_file(&path);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_json_lines_format() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("oxigaf_metrics_test_jsonl.jsonl");
        let _ = fs::remove_file(&path);

        let mut writer = MetricsWriter::new(&path, MetricsFormat::JsonLines)
            .expect("Failed to create JSON Lines writer");

        let metrics = TrainingMetrics {
            iteration: 1,
            loss_total: 1.234,
            loss_l1: 0.5,
            loss_ssim: 0.3,
            loss_lpips: Some(0.1),
            loss_reg: 0.334,
            num_gaussians: 50000,
            lr_position: 0.00016,
            lr_scaling: 0.005,
            lr_rotation: 0.001,
            memory_mb: Some(4096),
            elapsed_seconds: 120.5,
        };

        writer
            .write_metrics(&metrics)
            .expect("Failed to write metrics");
        drop(writer);

        let content = fs::read_to_string(&path).expect("Failed to read metrics file");
        let parsed: Result<serde_json::Value, _> =
            serde_json::from_str(content.lines().next().expect("No lines in file"));
        assert!(parsed.is_ok(), "Failed to parse JSON");

        let json = parsed.expect("JSON parsing failed");
        assert_eq!(json["iteration"], 1);
        assert_eq!(json["num_gaussians"], 50000);

        // Clean up
        let _ = fs::remove_file(&path);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_multiple_writes() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("oxigaf_metrics_test_multiple.csv");
        let _ = fs::remove_file(&path);

        let mut writer =
            MetricsWriter::new(&path, MetricsFormat::Csv).expect("Failed to create CSV writer");

        for i in 0..5 {
            let metrics = TrainingMetrics {
                iteration: i,
                loss_total: 1.0 - (i as f32 * 0.1),
                loss_l1: 0.5,
                loss_ssim: 0.3,
                loss_lpips: None,
                loss_reg: 0.2,
                num_gaussians: 50000 + i * 100,
                lr_position: 0.00016,
                lr_scaling: 0.005,
                lr_rotation: 0.001,
                memory_mb: None,
                elapsed_seconds: i as f32 * 10.0,
            };

            writer
                .write_metrics(&metrics)
                .expect("Failed to write metrics");
        }

        drop(writer);

        let content = fs::read_to_string(&path).expect("Failed to read metrics file");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 6, "Expected header + 5 data lines"); // header + 5 iterations

        // Clean up
        let _ = fs::remove_file(&path);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_manual_flush() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("oxigaf_metrics_test_flush.csv");
        let _ = fs::remove_file(&path);

        let mut writer =
            MetricsWriter::new(&path, MetricsFormat::Csv).expect("Failed to create CSV writer");

        let metrics = TrainingMetrics {
            iteration: 1,
            loss_total: 1.234,
            loss_l1: 0.5,
            loss_ssim: 0.3,
            loss_lpips: Some(0.1),
            loss_reg: 0.334,
            num_gaussians: 50000,
            lr_position: 0.00016,
            lr_scaling: 0.005,
            lr_rotation: 0.001,
            memory_mb: Some(4096),
            elapsed_seconds: 120.5,
        };

        writer
            .write_metrics(&metrics)
            .expect("Failed to write metrics");

        // Manual flush should not fail
        assert!(writer.flush().is_ok(), "Manual flush failed");

        drop(writer);

        // Clean up
        let _ = fs::remove_file(&path);
    }
}
