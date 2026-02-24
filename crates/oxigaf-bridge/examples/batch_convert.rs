//! Batch convert multiple GAF checkpoints from ToRSh to OxiGAF
//!
//! This example demonstrates batch processing of multiple checkpoint files
//! with parallel conversion, progress tracking, and error recovery.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example batch_convert --features torsh -- \
//!   --input-dir checkpoints/ \
//!   --output-dir oxigaf/ \
//!   --precision fp16
//! ```
//!
//! # Arguments
//!
//! - `--input-dir`: Directory containing ToRSh safetensors files
//! - `--output-dir`: Output directory for OxiGAF converted files
//! - `--precision`: Target precision (fp32, fp16, bf16) - default: fp32
//! - `--pattern`: Glob pattern for input files (default: "*.safetensors")
//! - `--parallel`: Number of parallel conversions (default: 4)
//!
//! # Examples
//!
//! Convert all safetensors files in a directory:
//! ```bash
//! cargo run --example batch_convert --features torsh -- \
//!   --input-dir checkpoints/ \
//!   --output-dir oxigaf/
//! ```
//!
//! Convert only specific checkpoints with FP16:
//! ```bash
//! cargo run --example batch_convert --features torsh -- \
//!   --input-dir checkpoints/ \
//!   --output-dir oxigaf/ \
//!   --pattern "gaf_*.safetensors" \
//!   --precision fp16
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use oxigaf_bridge::{Precision, WeightConverter};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "batch_convert")]
#[command(about = "Batch convert multiple GAF checkpoints")]
struct Args {
    /// Input directory containing ToRSh safetensors files
    #[arg(short, long)]
    input_dir: PathBuf,

    /// Output directory for OxiGAF converted files
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Target precision: fp32, fp16, or bf16 (default: fp32)
    #[arg(short, long, default_value = "fp32")]
    precision: String,

    /// Glob pattern for input files (default: "*.safetensors")
    #[arg(long, default_value = "*.safetensors")]
    pattern: String,

    /// Number of parallel conversions (default: 4)
    #[arg(long, default_value = "4")]
    parallel: usize,

    /// Continue on errors (default: false)
    #[arg(long)]
    continue_on_error: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

struct ConversionResult {
    input_path: PathBuf,
    #[allow(dead_code)]
    output_path: PathBuf,
    success: bool,
    duration: std::time::Duration,
    error: Option<String>,
}

fn main() -> Result<()> {
    // Parse arguments
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    }

    // Validate input directory exists
    if !args.input_dir.exists() {
        anyhow::bail!(
            "Input directory does not exist: {}",
            args.input_dir.display()
        );
    }

    // Parse precision
    let precision = match args.precision.to_lowercase().as_str() {
        "fp32" => Precision::FP32,
        "fp16" => Precision::FP16,
        "bf16" => Precision::BF16,
        _ => anyhow::bail!(
            "Invalid precision '{}'. Must be one of: fp32, fp16, bf16",
            args.precision
        ),
    };

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   OxiGAF Batch Converter - ToRSh → OxiGAF                ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Find all input files matching pattern
    let input_files = find_checkpoint_files(&args.input_dir, &args.pattern)?;

    if input_files.is_empty() {
        println!(
            "No files found matching pattern '{}' in {}",
            args.pattern,
            args.input_dir.display()
        );
        return Ok(());
    }

    println!("Found {} checkpoint(s) to convert", input_files.len());
    println!("Input dir:  {}", args.input_dir.display());
    println!("Output dir: {}", args.output_dir.display());
    println!("Precision:  {:?}", precision);
    println!("Parallel:   {} worker(s)", args.parallel);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            args.output_dir.display()
        )
    })?;

    // Setup progress bar
    let progress = Arc::new(ProgressBar::new(input_files.len() as u64));
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .map_err(|e| anyhow::anyhow!("Failed to set progress style: {}", e))?
            .progress_chars("█▓▒░ "),
    );

    // Setup counters
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));

    // Configure rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.parallel)
        .build_global()
        .ok(); // Ignore error if already initialized

    let start_time = Instant::now();

    // Convert files in parallel
    let results: Vec<ConversionResult> = input_files
        .par_iter()
        .map(|input_path| {
            let output_path = get_output_path(input_path, &args.input_dir, &args.output_dir);
            let converter = WeightConverter::new().with_precision(precision);

            let file_start = Instant::now();
            let result = converter.torsh_to_oxigaf(input_path, &output_path);
            let duration = file_start.elapsed();

            let success = result.is_ok();
            let error = result
                .err()
                .map(|e: oxigaf_bridge::BridgeError| e.to_string());

            if success {
                success_count.fetch_add(1, Ordering::SeqCst);
                progress.set_message(format!(
                    "✓ {}",
                    input_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?")
                ));
            } else {
                error_count.fetch_add(1, Ordering::SeqCst);
                progress.set_message(format!(
                    "✗ {}",
                    input_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?")
                ));
            }

            progress.inc(1);

            ConversionResult {
                input_path: input_path.clone(),
                output_path,
                success,
                duration,
                error,
            }
        })
        .collect();

    progress.finish_with_message("Complete");

    let total_time = start_time.elapsed();

    // Report results
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Conversion Results");
    println!("═══════════════════════════════════════════════════════════");

    let success = success_count.load(Ordering::SeqCst);
    let errors = error_count.load(Ordering::SeqCst);

    println!("Total files:     {}", input_files.len());
    println!("Successful:      {}", success);
    println!("Failed:          {}", errors);
    println!("Total duration:  {:?}", total_time);
    println!();

    // Show detailed results
    if args.verbose || errors > 0 {
        println!("Detailed Results:");
        for result in &results {
            let status = if result.success { "✓" } else { "✗" };
            let filename = result
                .input_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");

            if result.success {
                println!("  {} {} ({:?})", status, filename, result.duration);
            } else {
                println!(
                    "  {} {} - {}",
                    status,
                    filename,
                    result.error.as_deref().unwrap_or("Unknown error")
                );
            }
        }
        println!();
    }

    // Exit with error if any conversions failed and not continuing on error
    if errors > 0 && !args.continue_on_error {
        anyhow::bail!("{} conversion(s) failed", errors);
    }

    println!("Output directory: {}", args.output_dir.display());
    println!();

    if success > 0 {
        println!("✓ Batch conversion complete!");
    }

    Ok(())
}

/// Find all checkpoint files matching the pattern in the input directory
fn find_checkpoint_files(dir: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
    let full_pattern = dir.join(pattern);
    let pattern_str = full_pattern
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid path: {:?}", full_pattern))?;

    let entries = glob::glob(pattern_str)
        .with_context(|| format!("Failed to glob pattern: {}", pattern_str))?;

    let mut files = Vec::new();
    for entry in entries {
        let path = entry.with_context(|| "Failed to read glob entry")?;
        if path.is_file() {
            files.push(path);
        }
    }

    // Sort for deterministic order
    files.sort();

    Ok(files)
}

/// Generate output path by preserving relative structure
fn get_output_path(input_path: &Path, input_dir: &Path, output_dir: &Path) -> PathBuf {
    let relative = input_path.strip_prefix(input_dir).unwrap_or(input_path);

    output_dir.join(relative)
}
