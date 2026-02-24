//! Validate ToRSh ↔ OxiGAF round-trip conversion accuracy
//!
//! This example performs a complete round-trip conversion (ToRSh → OxiGAF → ToRSh)
//! and validates that the weights are preserved with acceptable precision loss.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example validate_conversion --features torsh -- \
//!   --checkpoint gaf_checkpoint.safetensors
//! ```
//!
//! # Arguments
//!
//! - `--checkpoint`: Path to ToRSh safetensors checkpoint file to validate
//! - `--precision`: Precision for intermediate conversion (fp32, fp16, bf16) - default: fp32
//! - `--tolerance`: Maximum allowed error (default: 1e-6 for FP32, 1e-3 for FP16)
//!
//! # Examples
//!
//! Validate with FP32 precision (most accurate):
//! ```bash
//! cargo run --example validate_conversion --features torsh -- \
//!   --checkpoint checkpoints/gaf_v1.safetensors
//! ```
//!
//! Validate with FP16 precision (tests quantization error):
//! ```bash
//! cargo run --example validate_conversion --features torsh -- \
//!   --checkpoint checkpoints/gaf_v1.safetensors \
//!   --precision fp16 \
//!   --tolerance 1e-3
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use oxigaf_bridge::{Precision, WeightConverter};
use safetensors::SafeTensors;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "validate_conversion")]
#[command(about = "Validate ToRSh ↔ OxiGAF round-trip conversion")]
struct Args {
    /// Path to ToRSh safetensors checkpoint file
    #[arg(short, long)]
    checkpoint: PathBuf,

    /// Precision for conversion: fp32, fp16, or bf16 (default: fp32)
    #[arg(short, long, default_value = "fp32")]
    precision: String,

    /// Maximum allowed absolute error (auto-set based on precision if not provided)
    #[arg(short, long)]
    tolerance: Option<f32>,

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
            .with_max_level(tracing::Level::WARN)
            .init();
    }

    // Validate input exists
    if !args.checkpoint.exists() {
        anyhow::bail!(
            "Checkpoint file does not exist: {}",
            args.checkpoint.display()
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

    // Set tolerance based on precision
    let tolerance = args.tolerance.unwrap_or(match precision {
        Precision::FP32 => 1e-6,
        Precision::FP16 => 1e-3,
        Precision::BF16 => 1e-2,
    });

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   OxiGAF Conversion Validator - Round-trip Test          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Checkpoint: {}", args.checkpoint.display());
    println!("Precision:  {:?}", precision);
    println!("Tolerance:  {:.2e}", tolerance);
    println!();

    // Create temporary paths
    let temp_dir = std::env::temp_dir();
    let oxigaf_path = temp_dir.join("validate_oxigaf.safetensors");
    let roundtrip_path = temp_dir.join("validate_roundtrip.safetensors");

    // Create converter
    let converter = WeightConverter::new().with_precision(precision);

    println!("[1/3] Converting ToRSh → OxiGAF...");
    let start = Instant::now();
    converter
        .torsh_to_oxigaf(&args.checkpoint, &oxigaf_path)
        .with_context(|| "ToRSh → OxiGAF conversion failed")?;
    let step1_time = start.elapsed();
    println!("      ✓ Complete ({:?})", step1_time);

    println!("[2/3] Converting OxiGAF → ToRSh...");
    let start = Instant::now();
    converter
        .oxigaf_to_torsh(&oxigaf_path, &roundtrip_path)
        .with_context(|| "OxiGAF → ToRSh conversion failed")?;
    let step2_time = start.elapsed();
    println!("      ✓ Complete ({:?})", step2_time);

    println!("[3/3] Validating round-trip accuracy...");
    let start = Instant::now();

    // Load original checkpoint
    let original_data =
        std::fs::read(&args.checkpoint).with_context(|| "Failed to read original checkpoint")?;
    let original_tensors = SafeTensors::deserialize(&original_data)
        .with_context(|| "Failed to parse original checkpoint")?;

    // Load round-trip checkpoint
    let roundtrip_data =
        std::fs::read(&roundtrip_path).with_context(|| "Failed to read round-trip checkpoint")?;
    let roundtrip_tensors = SafeTensors::deserialize(&roundtrip_data)
        .with_context(|| "Failed to parse round-trip checkpoint")?;

    // Validate tensor count matches
    let original_names: Vec<&str> = original_tensors.names();
    let roundtrip_names: Vec<&str> = roundtrip_tensors.names();

    if original_names.len() != roundtrip_names.len() {
        anyhow::bail!(
            "Tensor count mismatch: {} original vs {} round-trip",
            original_names.len(),
            roundtrip_names.len()
        );
    }

    // Compare each tensor
    let mut max_error = 0.0f32;
    let mut max_error_tensor = String::new();
    let mut total_elements = 0usize;
    let mut errors_above_threshold = 0usize;

    for name in &original_names {
        let original_tensor = original_tensors
            .tensor(name)
            .with_context(|| format!("Failed to get original tensor: {}", name))?;

        let roundtrip_tensor = roundtrip_tensors
            .tensor(name)
            .with_context(|| format!("Failed to get round-trip tensor: {}", name))?;

        // Verify shapes match
        if original_tensor.shape() != roundtrip_tensor.shape() {
            anyhow::bail!(
                "Shape mismatch for tensor '{}': {:?} vs {:?}",
                name,
                original_tensor.shape(),
                roundtrip_tensor.shape()
            );
        }

        // Compare data (convert to f32 for comparison)
        let original_f32: Vec<f32> =
            tensor_to_f32(original_tensor.data(), original_tensor.dtype())?;
        let roundtrip_f32: Vec<f32> =
            tensor_to_f32(roundtrip_tensor.data(), roundtrip_tensor.dtype())?;

        for (i, (&v1, &v2)) in original_f32.iter().zip(roundtrip_f32.iter()).enumerate() {
            let error = (v1 - v2).abs();
            total_elements += 1;

            if error > max_error {
                max_error = error;
                max_error_tensor = name.to_string();
            }

            if error > tolerance {
                errors_above_threshold += 1;
                if args.verbose {
                    println!(
                        "      Warning: Large error in '{}' at index {}: {:.6e} (original={}, roundtrip={})",
                        name, i, error, v1, v2
                    );
                }
            }
        }
    }

    let step3_time = start.elapsed();
    println!("      ✓ Complete ({:?})", step3_time);

    // Cleanup temporary files
    let _ = std::fs::remove_file(&oxigaf_path);
    let _ = std::fs::remove_file(&roundtrip_path);

    // Report results
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Validation Results");
    println!("═══════════════════════════════════════════════════════════");
    println!("Total tensors:      {}", original_names.len());
    println!("Total elements:     {}", total_elements);
    println!("Max absolute error: {:.6e}", max_error);
    println!("Worst tensor:       {}", max_error_tensor);
    println!("Errors > tolerance: {}", errors_above_threshold);
    println!();

    // Determine pass/fail
    if max_error <= tolerance {
        println!("✓ PASSED - Round-trip conversion is accurate!");
        println!("  All errors are within tolerance ({:.2e})", tolerance);
        Ok(())
    } else {
        println!("✗ FAILED - Round-trip conversion has excessive errors!");
        println!(
            "  Max error {:.2e} exceeds tolerance {:.2e}",
            max_error, tolerance
        );
        println!();
        println!("Suggestions:");
        println!("  • Try higher precision (--precision fp32)");
        println!("  • Increase tolerance if current precision is acceptable");
        println!("  • Check for normalization layer precision settings");
        anyhow::bail!("Validation failed: max error exceeds tolerance");
    }
}

/// Convert tensor bytes to Vec<f32> based on dtype
fn tensor_to_f32(data: &[u8], dtype: safetensors::tensor::Dtype) -> Result<Vec<f32>> {
    use safetensors::tensor::Dtype;

    match dtype {
        Dtype::F32 => {
            let slice: &[f32] = bytemuck::cast_slice(data);
            Ok(slice.to_vec())
        }
        Dtype::F16 => {
            use half::f16;
            let slice: &[f16] = bytemuck::cast_slice(data);
            Ok(slice.iter().map(|x: &f16| x.to_f32()).collect())
        }
        Dtype::BF16 => {
            use half::bf16;
            let slice: &[bf16] = bytemuck::cast_slice(data);
            Ok(slice.iter().map(|x: &bf16| x.to_f32()).collect())
        }
        _ => anyhow::bail!("Unsupported dtype for validation: {:?}", dtype),
    }
}
