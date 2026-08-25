//! Convert a FLAME `.pkl` head model to the directory of `.npy` files
//! `oxigaf-flame` loads — the pure-Rust replacement for
//! `scripts/convert_flame.py`.
//!
//! Writes `v_template`, `faces`, `shapedirs`, `expressiondirs`, `posedirs`,
//! `j_regressor`, `kintree_table` and `lbs_weights` as `.npy`, performing
//! the same identity/expression split of `shapedirs` and the same
//! densification of the SciPy-sparse `J_regressor` the Python script did.
//!
//! Unlike the Python script this needs no Python, no NumPy and no SciPy —
//! and it never executes anything the pickle names.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example convert_flame_pkl -- \
//!   --model FLAME2023.pkl --output-dir flame_model/
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use oxigaf_bridge::convert_flame_model;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "convert_flame_pkl")]
#[command(about = "Convert a FLAME .pkl model to .npy files (pure Rust)")]
struct Args {
    /// Path to the input FLAME `.pkl`.
    #[arg(short, long)]
    model: PathBuf,

    /// Directory to write the `.npy` files into.
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Verbose output.
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_max_level(if args.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .init();

    if !args.model.exists() {
        anyhow::bail!("Model does not exist: {}", args.model.display());
    }

    println!("Converting {} ...", args.model.display());
    let written = convert_flame_model(&args.model, &args.output_dir)
        .with_context(|| format!("Failed to convert {}", args.model.display()))?;

    println!();
    println!("Saved to {}/", args.output_dir.display());
    for (name, shape) in &written {
        println!("  {name:<20} {shape:?}");
    }
    Ok(())
}
