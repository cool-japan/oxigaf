//! # Basic FLAME Model Example
//!
//! This example demonstrates the fundamental FLAME parametric head model workflow:
//!
//! 1. Load the FLAME model from disk
//! 2. Configure shape/expression/pose parameters
//! 3. Run the forward pass to generate a posed mesh
//! 4. Render a normal map and save to disk
//!
//! ## Prerequisites
//!
//! Before running this example, you need FLAME model files converted to `.npy` format.
//! See `scripts/convert_flame.py` for conversion from the original `.pkl` files.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example basic_flame -- --model-dir /path/to/flame/model
//! ```

use std::path::PathBuf;

use oxigaf::prelude::*;

/// Parse command-line arguments to get model directory path.
/// Returns the path or a default for demonstration.
fn parse_args() -> PathBuf {
    // Check for --model-dir argument
    let args: Vec<String> = std::env::args().collect();
    for (i, arg) in args.iter().enumerate() {
        if arg == "--model-dir" {
            if let Some(path) = args.get(i + 1) {
                return PathBuf::from(path);
            }
        }
    }
    // Default path for demonstration (user should override)
    PathBuf::from("data/flame_model")
}

fn main() -> oxigaf::Result<()> {
    // =========================================================================
    // Step 1: Load the FLAME model
    // =========================================================================
    //
    // The FLAME model consists of:
    // - Template mesh vertices (5023 vertices for standard FLAME)
    // - Shape blend shapes (identity variations)
    // - Expression blend shapes (facial expressions)
    // - Pose corrective blend shapes
    // - Joint regressor for kinematic chain
    // - LBS skinning weights
    //
    // These are loaded from .npy files in the specified directory.

    let model_dir = parse_args();

    println!("OxiGAF Basic FLAME Example");
    println!("==========================");
    println!();
    println!("Loading FLAME model from: {}", model_dir.display());

    // Attempt to load the model. In production, handle the error gracefully.
    let model = match FlameModel::load(&model_dir) {
        Ok(m) => {
            println!(
                "Model loaded successfully: {} vertices, {} faces",
                m.num_vertices(),
                m.faces.len()
            );
            m
        }
        Err(e) => {
            // Provide helpful error message with instructions
            eprintln!("Error loading FLAME model: {}", e);
            eprintln!();
            eprintln!("To run this example, you need FLAME model files:");
            eprintln!("  1. Download FLAME from https://flame.is.tue.mpg.de/");
            eprintln!("  2. Convert using: python scripts/convert_flame.py");
            eprintln!("  3. Run with: cargo run --example basic_flame -- --model-dir /path/to/converted/model");
            eprintln!();
            eprintln!("Alternatively, running in demo mode with synthetic data...");

            // For demonstration, we'll show the API without actual model files
            demonstrate_api_without_model();
            return Ok(());
        }
    };

    // =========================================================================
    // Step 2: Configure FLAME parameters
    // =========================================================================
    //
    // FLAME is controlled by several parameter types:
    // - Shape (beta): Identity-specific features (face shape, head size)
    // - Expression (psi): Facial expressions (smile, frown, surprise)
    // - Pose (theta): Joint rotations (head, neck, jaw, eyes)
    // - Translation: Global 3D offset
    //
    // We use the builder pattern for clear, type-safe parameter construction.

    println!();
    println!("Creating FLAME parameters...");

    // Example 1: Neutral pose (all zeros)
    let neutral_params = FlameParams::neutral();
    println!("  Created neutral parameters");

    // Example 2: Using the builder pattern for a smiling expression with slight head tilt
    let smiling_params = FlameParams::builder()
        // Shape coefficients affect identity (face width, nose length, etc.)
        // Typically use first 10-100 components from PCA
        .shape(vec![0.5, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        // Expression coefficients control facial movements
        // First few coefficients: smile, mouth open, brow raise, etc.
        .expression(vec![0.8, 0.3, -0.1, 0.0, 0.0])
        // Head rotation: slight tilt to the right (axis-angle: [rx, ry, rz])
        .root_rotation([0.1, 0.0, 0.05])
        // Jaw slightly open for natural smile
        .jaw_rotation(0.08)
        // Small translation to center in view
        .translation([0.0, -0.05, 0.0])
        .build();
    println!("  Created smiling expression parameters");

    // Validate parameters are within reasonable ranges
    if smiling_params.validate() {
        println!("  Parameters validated: within typical ranges");
    } else {
        println!("  Warning: Some parameters outside typical ranges");
    }

    // =========================================================================
    // Step 3: Run forward pass to generate posed mesh
    // =========================================================================
    //
    // The forward pass applies:
    // 1. Shape blend shapes to template
    // 2. Expression blend shapes
    // 3. Pose corrective blend shapes
    // 4. Linear Blend Skinning (LBS) for articulation
    // 5. Global translation
    //
    // Result: A posed mesh with vertices and computed normals

    println!();
    println!("Running FLAME forward pass...");

    let neutral_mesh = model.forward(&neutral_params);
    println!(
        "  Neutral mesh: {} vertices, {} faces",
        neutral_mesh.vertices.len(),
        neutral_mesh.faces.len()
    );

    let smiling_mesh = model.forward(&smiling_params);
    println!(
        "  Smiling mesh: {} vertices, {} faces",
        smiling_mesh.vertices.len(),
        smiling_mesh.faces.len()
    );

    // Compute vertex displacement between poses
    let max_displacement = neutral_mesh
        .vertices
        .iter()
        .zip(smiling_mesh.vertices.iter())
        .map(|(n, s)| {
            let dx = s.x - n.x;
            let dy = s.y - n.y;
            let dz = s.z - n.z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0f32, f32::max);
    println!("  Max vertex displacement: {:.4} units", max_displacement);

    // =========================================================================
    // Step 4: Render normal map
    // =========================================================================
    //
    // Normal maps encode surface orientation as RGB colors:
    // - R = (normal.x + 1) / 2 * 255
    // - G = (normal.y + 1) / 2 * 255
    // - B = (normal.z + 1) / 2 * 255
    //
    // These are used as conditioning input for the diffusion model.

    println!();
    println!("Rendering normal maps...");

    // Create a front-facing camera
    let camera = Camera::default_front(512, 512);
    println!(
        "  Camera: {}x{}, focal={:.1}",
        camera.width, camera.height, camera.focal_x
    );

    // Render normal maps for both poses
    let neutral_normal_map = NormalMapRenderer::render(&neutral_mesh, &camera);
    let smiling_normal_map = NormalMapRenderer::render(&smiling_mesh, &camera);
    println!("  Rendered normal maps: {}x{}", camera.width, camera.height);

    // =========================================================================
    // Step 5: Save output images
    // =========================================================================
    //
    // Save the normal maps to disk for inspection or further processing.

    println!();
    println!("Saving output images...");

    // Use system temp directory for output
    let output_dir = std::env::temp_dir().join("oxigaf_examples");
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("Warning: Could not create output directory: {}", e);
    }

    let neutral_path = output_dir.join("normal_map_neutral.png");
    let smiling_path = output_dir.join("normal_map_smiling.png");

    match neutral_normal_map.save(&neutral_path) {
        Ok(()) => println!("  Saved neutral normal map: {}", neutral_path.display()),
        Err(e) => eprintln!("  Error saving neutral image: {}", e),
    }

    match smiling_normal_map.save(&smiling_path) {
        Ok(()) => println!("  Saved smiling normal map: {}", smiling_path.display()),
        Err(e) => eprintln!("  Error saving smiling image: {}", e),
    }

    // =========================================================================
    // Summary
    // =========================================================================

    println!();
    println!("Example completed successfully!");
    println!();
    println!("Key takeaways:");
    println!("  - FlameModel::load() loads model from .npy files");
    println!("  - FlameParams::builder() provides type-safe parameter construction");
    println!("  - model.forward(&params) generates posed mesh via LBS");
    println!("  - NormalMapRenderer::render() creates normal map images");
    println!();
    println!("Output files saved to: {}", output_dir.display());

    Ok(())
}

/// Demonstrate the API when model files are not available.
/// Shows the parameter building and validation APIs.
fn demonstrate_api_without_model() {
    println!();
    println!("API Demonstration (without model files)");
    println!("=======================================");
    println!();

    // Show parameter creation
    println!("1. Creating neutral parameters:");
    let neutral = FlameParams::neutral();
    println!(
        "   Shape: {} coefficients, Expression: {} coefficients",
        neutral.shape.len(),
        neutral.expression.len()
    );
    println!(
        "   Pose: {} values (5 joints x 3 axis-angle)",
        neutral.pose.len()
    );
    println!("   Translation: {:?}", neutral.translation);

    // Show builder pattern
    println!();
    println!("2. Using FlameParamsBuilder:");
    let params = FlameParams::builder()
        .shape(vec![1.0, -0.5, 0.3])
        .expression(vec![0.8, 0.2])
        .jaw_rotation(0.15)
        .left_eye_rotation([0.1, 0.0, 0.0])
        .translation([0.0, 0.1, 0.0])
        .build();

    println!(
        "   Built params with {} shape, {} expression coefficients",
        params.shape.len(),
        params.expression.len()
    );

    // Show validation
    println!();
    println!("3. Parameter validation:");
    if params.validate() {
        println!("   Parameters are within typical ranges");
    }

    // Show joint access
    println!();
    println!("4. Accessing joint poses:");
    for joint_idx in 0..FlameParams::NUM_JOINTS {
        let joint_name = match joint_idx {
            0 => "Root",
            1 => "Neck",
            2 => "Jaw",
            3 => "Left Eye",
            4 => "Right Eye",
            _ => "Unknown",
        };
        let pose = params.joint_pose(joint_idx);
        println!(
            "   {}: [{:.3}, {:.3}, {:.3}]",
            joint_name, pose[0], pose[1], pose[2]
        );
    }

    println!();
    println!("To run with actual model files, see the prerequisites in the example source.");
}
