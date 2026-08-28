//! Convert a PyTorch `.pt` / `.pth` checkpoint to per-component
//! `.safetensors` — the pure-Rust replacement for
//! `scripts/convert_weights.py`.
//!
//! Tensors are partitioned into `unet` / `vae` / `clip` / `other` by the
//! same prefixes the Python script recognized
//! (`model.diffusion_model.`, `unet.`, `first_stage_model.`, `vae.`,
//! `cond_stage_model.`, `clip.`), the component prefix is stripped, and each
//! non-empty group is written as `<component>.safetensors` with names in the
//! dot-separated form `candle_nn::VarBuilder::pp` walks.
//!
//! Unlike the Python script this needs no Python, no PyTorch, and no
//! `torch.load` — and it never executes anything the pickle names.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example convert_pytorch -- \
//!   --checkpoint model.pt --output-dir weights/ --precision fp16
//! ```
//!
//! `--precision` is optional; omit it to keep each tensor's original dtype
//! (the Python script always forced FP16).

use anyhow::{Context, Result};
use clap::Parser;
use oxigaf_bridge::{convert_pytorch_checkpoint, Precision};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "convert_pytorch")]
#[command(about = "Convert a PyTorch .pt checkpoint to .safetensors (pure Rust)")]
struct Args {
    /// Path to the input `.pt` / `.pth` checkpoint.
    #[arg(short, long)]
    checkpoint: PathBuf,

    /// Directory to write `<component>.safetensors` into.
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Stored precision: fp32, fp16, or bf16. Omit to keep the source dtype.
    #[arg(short, long)]
    precision: Option<String>,

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

    if !args.checkpoint.exists() {
        anyhow::bail!("Checkpoint does not exist: {}", args.checkpoint.display());
    }

    let precision = match args.precision.as_deref() {
        None => None,
        Some(text) => Some(match text.to_lowercase().as_str() {
            "fp32" => Precision::FP32,
            "fp16" => Precision::FP16,
            "bf16" => Precision::BF16,
            other => anyhow::bail!("Invalid precision '{other}'. Use fp32, fp16, or bf16"),
        }),
    };

    println!("Converting {} ...", args.checkpoint.display());
    let report = convert_pytorch_checkpoint(&args.checkpoint, &args.output_dir, precision)
        .with_context(|| format!("Failed to convert {}", args.checkpoint.display()))?;

    println!();
    for (component, tensors, params) in &report.components {
        println!(
            "  {component:<8} {tensors:>5} tensors, {params:>12} params -> {}/{component}.safetensors",
            args.output_dir.display()
        );
    }
    println!();
    println!("Done: {} tensors written.", report.total_tensors());
    Ok(())
}
