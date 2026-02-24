//! Convert FLAME model from .npy format to .safetensors format.
//!
//! This example demonstrates how to convert a FLAME model stored as multiple
//! .npy files into a single .safetensors file for easier distribution and loading.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example convert_npy_to_safetensors --features safetensors \
//!     -- path/to/flame/npy/dir output.safetensors
//! ```
//!
//! # Input
//!
//! The input directory should contain the following .npy files:
//! - v_template.npy
//! - faces.npy
//! - shapedirs.npy
//! - expressiondirs.npy
//! - posedirs.npy
//! - J_regressor.npy
//! - kintree_table.npy
//! - weights.npy
//!
//! # Output
//!
//! A single .safetensors file containing all model data with metadata.

use oxigaf_flame::conversion::convert_npy_to_safetensors;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <npy_dir> <output.safetensors>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  {} flame_2020/generic_model flame_2020.safetensors",
            args[0]
        );
        std::process::exit(1);
    }

    let npy_dir = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    // Validate input directory
    if !npy_dir.is_dir() {
        eprintln!(
            "Error: Input directory does not exist: {}",
            npy_dir.display()
        );
        std::process::exit(1);
    }

    println!("Converting FLAME model...");
    println!("  Input:  {}", npy_dir.display());
    println!("  Output: {}", output_path.display());
    println!();

    // Perform conversion
    convert_npy_to_safetensors(&npy_dir, &output_path)?;

    println!("Conversion complete!");
    println!();
    println!("The safetensors file can now be loaded with:");
    println!(
        "  FlameModel::load_safetensors(\"{}\")",
        output_path.display()
    );

    Ok(())
}
