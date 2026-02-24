//! Convert GAF checkpoint from ToRSh to OxiGAF format
//!
//! This example demonstrates basic weight conversion workflow from ToRSh
//! safetensors format to OxiGAF native format with precision control.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example convert_gaf_checkpoint --features torsh -- \
//!   --input gaf_checkpoint.safetensors \
//!   --output oxigaf/ \
//!   --precision fp16
//! ```
//!
//! # Arguments
//!
//! - `--input`: Path to ToRSh safetensors checkpoint file
//! - `--output`: Output directory or file path for OxiGAF format
//! - `--precision`: Target precision (fp32, fp16, bf16) - default: fp32
//!
//! # Examples
//!
//! Convert with default FP32 precision:
//! ```bash
//! cargo run --example convert_gaf_checkpoint --features torsh -- \
//!   --input checkpoints/gaf_v1.safetensors \
//!   --output oxigaf/unet.safetensors
//! ```
//!
//! Convert with FP16 precision for reduced memory:
//! ```bash
//! cargo run --example convert_gaf_checkpoint --features torsh -- \
//!   --input checkpoints/gaf_v1.safetensors \
//!   --output oxigaf/unet_fp16.safetensors \
//!   --precision fp16
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use oxigaf_bridge::{Precision, WeightConverter};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "convert_gaf_checkpoint")]
#[command(about = "Convert GAF checkpoint from ToRSh to OxiGAF format")]
struct Args {
    /// Path to ToRSh safetensors checkpoint file
    #[arg(short, long)]
    input: PathBuf,

    /// Output path for OxiGAF format (file or directory)
    #[arg(short, long)]
    output: PathBuf,

    /// Target precision: fp32, fp16, or bf16 (default: fp32)
    #[arg(short, long, default_value = "fp32")]
    precision: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    // Parse arguments
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    // Validate input exists
    if !args.input.exists() {
        anyhow::bail!("Input file does not exist: {}", args.input.display());
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
    println!("║   OxiGAF Weight Converter - ToRSh → OxiGAF               ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Input:     {}", args.input.display());
    println!("Output:    {}", args.output.display());
    println!("Precision: {:?}", precision);
    println!();

    // Create output directory if needed
    if let Some(parent) = args.output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create output directory: {}", parent.display())
            })?;
            println!("Created output directory: {}", parent.display());
        }
    }

    // Create converter
    let converter = WeightConverter::new().with_precision(precision);

    // Perform conversion with timing
    println!("Converting weights...");
    let start = Instant::now();

    converter
        .torsh_to_oxigaf(&args.input, &args.output)
        .with_context(|| "Weight conversion failed")?;

    let elapsed = start.elapsed();

    // Report success
    println!();
    println!("✓ Conversion complete!");
    println!();
    println!("Duration:  {:?}", elapsed);
    println!("Output:    {}", args.output.display());

    // Display file size if available
    if args.output.exists() {
        let metadata = std::fs::metadata(&args.output)?;
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("Size:      {:.2} MB", size_mb);
    }

    println!();
    println!("Next steps:");
    println!("  • Verify conversion: cargo run --example validate_conversion --features torsh -- --checkpoint {}", args.output.display());
    println!("  • Load in OxiGAF:    Use WeightConverter::oxigaf_to_torsh() or load directly with Candle");

    Ok(())
}
