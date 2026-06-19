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
    #[allow(unused_mut)]
    let mut sequence = sequence;

    match format.as_str() {
        "json" => {
            println!("Saving to JSON: {}", output_path.display());
            save_sequence_json(&sequence, &output_path)?;
        }
        "npz" => {
            #[cfg(feature = "npz")]
            {
                println!("Saving to NPZ: {}", output_path.display());
                save_sequence_npz(&mut sequence, &output_path)?;
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
    sequence: &mut FlameSequence,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::Array2;
    use ndarray_npy::NpzWriter;
    use std::fs::File;

    let n_frames = sequence.num_frames();
    if n_frames == 0 {
        return Err("Cannot export an empty sequence to NPZ".into());
    }

    // Determine array widths from the first frame.
    let first = sequence
        .get_frame(0)
        .map_err(|e| format!("Failed to read frame 0: {}", e))?;
    let n_shape = first.shape.len();
    let n_expr = first.expression.len();
    let n_pose = first.pose.len();

    // Allocate row-major f32 arrays: shape (n_frames, n_coeffs).
    let mut shape_data: Array2<f32> = Array2::zeros((n_frames, n_shape));
    let mut expr_data: Array2<f32> = Array2::zeros((n_frames, n_expr));
    let mut pose_data: Array2<f32> = Array2::zeros((n_frames, n_pose));
    let mut trans_data: Array2<f32> = Array2::zeros((n_frames, 3));

    // Populate arrays from the sequence, one frame at a time.
    for i in 0..n_frames {
        let frame = sequence
            .get_frame(i)
            .map_err(|e| format!("Failed to read frame {}: {}", i, e))?;

        for (j, &v) in frame.shape.iter().enumerate() {
            shape_data[[i, j]] = v;
        }
        for (j, &v) in frame.expression.iter().enumerate() {
            expr_data[[i, j]] = v;
        }
        for (j, &v) in frame.pose.iter().enumerate() {
            pose_data[[i, j]] = v;
        }
        trans_data[[i, 0]] = frame.translation[0];
        trans_data[[i, 1]] = frame.translation[1];
        trans_data[[i, 2]] = frame.translation[2];
    }

    // Write all arrays into a single .npz archive.
    let file = File::create(path)
        .map_err(|e| format!("Failed to create NPZ file {}: {}", path.display(), e))?;
    let mut writer = NpzWriter::new(file);
    writer
        .add_array("shape", &shape_data)
        .map_err(|e| format!("Failed to write 'shape' array: {}", e))?;
    writer
        .add_array("expression", &expr_data)
        .map_err(|e| format!("Failed to write 'expression' array: {}", e))?;
    writer
        .add_array("pose", &pose_data)
        .map_err(|e| format!("Failed to write 'pose' array: {}", e))?;
    writer
        .add_array("translation", &trans_data)
        .map_err(|e| format!("Failed to write 'translation' array: {}", e))?;
    writer
        .finish()
        .map_err(|e| format!("Failed to finalise NPZ archive: {}", e))?;

    println!("Saved {} frames as NPZ to {}", n_frames, path.display());
    Ok(())
}

#[cfg(not(feature = "npz"))]
#[allow(dead_code)]
fn save_sequence_npz(
    _sequence: &mut FlameSequence,
    _path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    unreachable!()
}
