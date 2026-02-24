//! Load and process FLAME parameter sequences.
//!
//! This example demonstrates how to load FLAME parameter sequences from
//! different formats (JSON, NPZ, directory of files) and access frame data.
//!
//! # Usage
//!
//! ```bash
//! # Load from JSON file
//! cargo run --example load_flame_sequence -- json sequence.json
//!
//! # Load from NPZ file (requires npz feature)
//! cargo run --example load_flame_sequence --features npz -- npz sequence.npz
//!
//! # Load from directory of per-frame JSON files
//! cargo run --example load_flame_sequence -- dir frames/ "frame_{:04}.json" 100
//! ```

use oxigaf_flame::sequence::FlameSequence;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  {} json <sequence.json>", args[0]);
        eprintln!("  {} npz <sequence.npz>", args[0]);
        eprintln!("  {} dir <directory> <pattern> <num_frames>", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} json my_sequence.json", args[0]);
        eprintln!("  {} npz tracked_params.npz", args[0]);
        eprintln!("  {} dir frames/ \"frame_{{:04}}.json\" 100", args[0]);
        std::process::exit(1);
    }

    let format = &args[1];
    let mut sequence = match format.as_str() {
        "json" => {
            let path = PathBuf::from(&args[2]);
            println!("Loading sequence from JSON: {}", path.display());
            FlameSequence::from_json(&path)?
        }
        "npz" => {
            #[cfg(feature = "npz")]
            {
                let path = PathBuf::from(&args[2]);
                println!("Loading sequence from NPZ: {}", path.display());
                FlameSequence::from_npz(&path)?
            }
            #[cfg(not(feature = "npz"))]
            {
                eprintln!("Error: NPZ support not enabled. Rebuild with --features npz");
                std::process::exit(1);
            }
        }
        "dir" => {
            if args.len() < 5 {
                eprintln!("Error: Directory format requires 3 arguments");
                eprintln!("Usage: {} dir <directory> <pattern> <num_frames>", args[0]);
                std::process::exit(1);
            }
            let dir = PathBuf::from(&args[2]);
            let pattern = &args[3];
            let num_frames: usize = args[4]
                .parse()
                .map_err(|_| "Invalid num_frames: must be a positive integer")?;
            println!("Loading sequence from directory: {}", dir.display());
            println!("  Pattern: {}", pattern);
            println!("  Frames:  {}", num_frames);
            FlameSequence::from_directory(&dir, pattern, num_frames, None)?
        }
        _ => {
            eprintln!("Error: Unknown format '{}'", format);
            eprintln!("Supported formats: json, npz, dir");
            std::process::exit(1);
        }
    };

    println!();
    println!("Sequence loaded successfully!");
    println!("  Total frames: {}", sequence.num_frames());
    if let Some(fps) = sequence.fps() {
        println!("  FPS:          {}", fps);
        println!("  Duration:     {:.2}s", sequence.num_frames() as f32 / fps);
    }

    // Access first frame to get parameter dimensions
    let first_frame = sequence.get_frame(0)?;
    println!();
    println!("Parameter dimensions:");
    println!("  Shape:      {} coefficients", first_frame.shape.len());
    println!(
        "  Expression: {} coefficients",
        first_frame.expression.len()
    );
    println!("  Pose:       {} coefficients", first_frame.pose.len());
    println!(
        "  Translation: [{:.3}, {:.3}, {:.3}]",
        first_frame.translation[0], first_frame.translation[1], first_frame.translation[2]
    );

    // Demonstrate frame access
    println!();
    println!("Frame access examples:");

    // Access middle frame
    if sequence.num_frames() > 1 {
        let mid_idx = sequence.num_frames() / 2;
        let mid_frame = sequence.get_frame(mid_idx)?;
        println!("  Frame {} shape[0]: {:.6}", mid_idx, mid_frame.shape[0]);
    }

    // Demonstrate interpolation
    if sequence.num_frames() > 2 {
        let interp_frame = sequence.interpolate(1.5)?;
        println!(
            "  Interpolated frame 1.5 shape[0]: {:.6}",
            interp_frame.shape[0]
        );
    }

    // Demonstrate iteration over first 5 frames
    println!();
    println!("First 5 frames:");
    for (i, result) in sequence.iter().take(5).enumerate() {
        let frame = result?;
        println!(
            "  Frame {}: shape[0]={:.6}, expr[0]={:.6}",
            i, frame.shape[0], frame.expression[0]
        );
    }

    println!();
    println!("Done!");

    Ok(())
}
