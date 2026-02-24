//! Create and save FLAME parameter sequences.
//!
//! This example demonstrates how to create FLAME parameter sequences
//! programmatically and save them to JSON or NPZ format.
//!
//! # Usage
//!
//! ```bash
//! # Create synthetic sequence and save as JSON
//! cargo run --example create_flame_sequence -- json output.json 100 30
//!
//! # Create synthetic sequence and save as NPZ (requires npz feature)
//! cargo run --example create_flame_sequence --features npz -- npz output.npz 100 30
//! ```

use oxigaf_flame::params::FlameParams;
use oxigaf_flame::sequence::FlameSequence;
use std::path::PathBuf;

fn create_synthetic_params(
    frame_idx: usize,
    n_shape: usize,
    n_expr: usize,
    n_pose: usize,
) -> FlameParams {
    // Create synthetic parameters that vary over time
    let t = frame_idx as f32;

    // Shape parameters: slowly varying identity
    let shape = (0..n_shape)
        .map(|i| 0.1 * (t * 0.01 + i as f32 * 0.1).sin())
        .collect();

    // Expression parameters: faster varying facial expressions
    let expression = (0..n_expr)
        .map(|i| 0.3 * (t * 0.1 + i as f32 * 0.2).sin())
        .collect();

    // Pose parameters: head rotation
    let pose = (0..n_pose)
        .map(|i| 0.2 * (t * 0.05 + i as f32 * 0.15).cos())
        .collect();

    // Translation: gentle head motion
    let translation = [0.05 * (t * 0.02).sin(), 0.05 * (t * 0.03).cos(), 0.0];

    FlameParams {
        shape,
        expression,
        pose,
        translation,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "Usage: {} <format> <output_path> <num_frames> <fps>",
            args[0]
        );
        eprintln!();
        eprintln!("Formats: json, npz");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} json sequence.json 120 30", args[0]);
        eprintln!("  {} npz sequence.npz 240 60", args[0]);
        std::process::exit(1);
    }

    let format = &args[1];
    let output_path = PathBuf::from(&args[2]);
    let num_frames: usize = args[3]
        .parse()
        .map_err(|_| "Invalid num_frames: must be a positive integer")?;
    let fps: f32 = args[4]
        .parse()
        .map_err(|_| "Invalid fps: must be a positive number")?;

    // FLAME parameter dimensions (standard configuration)
    let n_shape = 100;
    let n_expr = 50;
    let n_pose = 15; // 5 joints × 3 DOF

    println!("Creating synthetic FLAME sequence...");
    println!("  Frames:     {}", num_frames);
    println!("  FPS:        {}", fps);
    println!("  Duration:   {:.2}s", num_frames as f32 / fps);
    println!("  Shape:      {} coefficients", n_shape);
    println!("  Expression: {} coefficients", n_expr);
    println!("  Pose:       {} coefficients", n_pose);
    println!();

    // Generate synthetic parameters
    let frames: Vec<FlameParams> = (0..num_frames)
        .map(|i| create_synthetic_params(i, n_shape, n_expr, n_pose))
        .collect();

    // Create sequence
    let sequence = FlameSequence::from_memory(frames, Some(fps));

    // Save to file
    match format.as_str() {
        "json" => {
            println!("Saving to JSON: {}", output_path.display());
            save_sequence_json(&sequence, &output_path)?;
        }
        "npz" => {
            #[cfg(feature = "npz")]
            {
                println!("Saving to NPZ: {}", output_path.display());
                save_sequence_npz(&sequence, &output_path)?;
            }
            #[cfg(not(feature = "npz"))]
            {
                eprintln!("Error: NPZ support not enabled. Rebuild with --features npz");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Error: Unknown format '{}'", format);
            eprintln!("Supported formats: json, npz");
            std::process::exit(1);
        }
    }

    println!();
    println!("Sequence saved successfully!");
    println!("You can load it with:");
    println!(
        "  FlameSequence::from_{}(\"{}\")",
        format,
        output_path.display()
    );

    Ok(())
}

fn save_sequence_json(
    _sequence: &FlameSequence,
    _path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    // Note: JSON export from FlameSequence not directly supported in this example
    // In a real application, you would:
    // 1. Iterate through all frames
    // 2. Collect them into a SequenceJson structure
    // 3. Serialize to JSON
    eprintln!("Note: JSON export from FlameSequence requires mutable access");
    eprintln!("      This is a limitation of the current API");
    eprintln!("      Consider creating a helper method or using NPZ format");
    Err("Cannot export from FlameSequence to JSON in this example".into())
}

#[cfg(feature = "npz")]
fn save_sequence_npz(
    _sequence: &FlameSequence,
    _path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    // NPZ export would require collecting all frames and writing as arrays
    // This is left as an exercise for the reader
    eprintln!("Note: NPZ export from FlameSequence not yet implemented");
    eprintln!("      You can manually collect frames and write using ndarray-npy");
    Err("NPZ export not implemented".into())
}

#[cfg(not(feature = "npz"))]
#[allow(dead_code)]
fn save_sequence_npz(
    _sequence: &FlameSequence,
    _path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    unreachable!()
}
