//! # Gaussian Splatting Render Example
//!
//! This example demonstrates the GPU-accelerated 3D Gaussian Splatting rasterizer:
//!
//! 1. Create a simple Gaussian model with a few primitives
//! 2. Initialize the wgpu-based rasterizer
//! 3. Set up camera parameters
//! 4. Render a single frame and save to disk
//!
//! ## 3D Gaussian Splatting Overview
//!
//! Each Gaussian primitive is defined by:
//! - **Position**: 3D center point (x, y, z)
//! - **Rotation**: Quaternion (x, y, z, w) defining orientation
//! - **Scale**: Log-scale values (sx, sy, sz) - exponentiated before use
//! - **Opacity**: Inverse-sigmoid opacity - passed through sigmoid
//! - **SH Coefficients**: Spherical harmonics for view-dependent color
//!
//! The rasterizer uses compute shaders for:
//! - Projection and covariance computation
//! - Tile-based sorting by depth
//! - Per-pixel alpha blending
//!
//! ## Running
//!
//! ```bash
//! cargo run --example gaussian_render
//! ```
//!
//! Note: Requires a GPU with Vulkan, Metal, or DX12 support.

use oxigaf::prelude::*;
use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

/// Create a simple test scene with a few Gaussians.
///
/// This generates Gaussians arranged in a grid pattern with varying colors
/// to demonstrate the rendering pipeline.
fn create_test_gaussians() -> GaussianModel {
    // We'll create a 3x3 grid of Gaussians with different colors
    let grid_size = 3;
    let spacing = 0.15f32;
    let offset = -(grid_size as f32 - 1.0) * spacing / 2.0;

    let mut gaussians = Vec::new();
    let mut sh_coeffs = Vec::new();

    // SH degree 0 means just DC component (3 coefficients for RGB)
    let sh_degree = 0u32;
    let _sh_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;

    for row in 0..grid_size {
        for col in 0..grid_size {
            // Position in a grid, centered at origin, facing camera (at z=0.5)
            let x = offset + col as f32 * spacing;
            let y = offset + row as f32 * spacing;
            let z = 0.0; // In front of camera

            // Create Gaussian attributes
            // - Log-scale of -4 gives exp(-4) ~ 0.018 scale
            // - Opacity of 2.0 gives sigmoid(2.0) ~ 0.88
            let gaussian = GaussianAttributes {
                position: [x, y, z],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
                scale: [-4.0, -4.0, -4.0],      // Uniform log-scale
                opacity: 2.0,                   // Inverse-sigmoid opacity
            };
            gaussians.push(gaussian);

            // Generate color based on grid position (using SH DC term)
            // Map row/col to RGB for visual distinction
            let r = (col as f32 / (grid_size - 1) as f32 - 0.5) * 0.5; // Red varies with column
            let g = (row as f32 / (grid_size - 1) as f32 - 0.5) * 0.5; // Green varies with row
            let b = 0.3; // Some blue for all

            // SH coefficient 0 (DC) is multiplied by ~0.28 (Y_0^0 = 0.282...)
            // So we scale up to get desired final color
            let sh_scale = 1.0 / 0.2821;
            sh_coeffs.push(r * sh_scale);
            sh_coeffs.push(g * sh_scale);
            sh_coeffs.push(b * sh_scale);
        }
    }

    let num_gaussians = gaussians.len();

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        // For this simple example, we don't use FLAME binding
        face_indices: vec![0; num_gaussians],
        barycentric: vec![[1.0, 0.0, 0.0]; num_gaussians],
        local_offsets: vec![[0.0, 0.0, 0.0]; num_gaussians],
        is_rigid: vec![true; num_gaussians],
    }
}

/// Create a camera looking at the origin from a distance.
///
/// Returns a RenderCamera with view and projection matrices.
fn create_camera(width: u32, height: u32) -> RenderCamera {
    // Camera positioned at z=0.5, looking at origin
    let eye = [0.0f32, 0.0, 0.5];
    let center = [0.0f32, 0.0, 0.0];
    let up = [0.0f32, 1.0, 0.0];

    // Compute view matrix (look-at)
    let forward = [center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]];
    let forward_len =
        (forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]).sqrt();
    let forward = [
        forward[0] / forward_len,
        forward[1] / forward_len,
        forward[2] / forward_len,
    ];

    // Right = forward x up
    let right = [
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    ];
    let right_len = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
    let right = [
        right[0] / right_len,
        right[1] / right_len,
        right[2] / right_len,
    ];

    // True up = right x forward
    let true_up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ];

    // View matrix (column-major, 4x4)
    // Note: We look in -forward direction for right-handed coordinate system
    let mut view_matrix = [0.0f32; 16];
    // Column 0: right
    view_matrix[0] = right[0];
    view_matrix[1] = true_up[0];
    view_matrix[2] = -forward[0];
    view_matrix[3] = 0.0;
    // Column 1: up
    view_matrix[4] = right[1];
    view_matrix[5] = true_up[1];
    view_matrix[6] = -forward[1];
    view_matrix[7] = 0.0;
    // Column 2: -forward
    view_matrix[8] = right[2];
    view_matrix[9] = true_up[2];
    view_matrix[10] = -forward[2];
    view_matrix[11] = 0.0;
    // Column 3: translation
    let tx = -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]);
    let ty = -(true_up[0] * eye[0] + true_up[1] * eye[1] + true_up[2] * eye[2]);
    let tz = -(-forward[0] * eye[0] + -forward[1] * eye[1] + -forward[2] * eye[2]);
    view_matrix[12] = tx;
    view_matrix[13] = ty;
    view_matrix[14] = tz;
    view_matrix[15] = 1.0;

    // Projection matrix (perspective, column-major)
    let focal = width as f32 * 1.5;
    let near = 0.01f32;
    let far = 100.0f32;
    let w = width as f32;
    let h = height as f32;
    let cx = w / 2.0;
    let cy = h / 2.0;

    let mut proj_matrix = [0.0f32; 16];
    proj_matrix[0] = 2.0 * focal / w; // m[0][0]
    proj_matrix[5] = 2.0 * focal / h; // m[1][1]
    proj_matrix[8] = -(2.0 * cx / w - 1.0); // m[2][0]
    proj_matrix[9] = -(2.0 * cy / h - 1.0); // m[2][1]
    proj_matrix[10] = -(far + near) / (far - near); // m[2][2]
    proj_matrix[11] = -1.0; // m[2][3]
    proj_matrix[14] = -2.0 * far * near / (far - near); // m[3][2]

    RenderCamera {
        view_matrix,
        proj_matrix,
        position: eye,
        focal: [focal, focal],
    }
}

/// Main entry point - runs the async rendering pipeline.
fn main() {
    println!("OxiGAF Gaussian Render Example");
    println!("===============================");
    println!();

    // Run the async rendering code using pollster for blocking
    match pollster::block_on(run_render()) {
        Ok(()) => {
            println!();
            println!("Example completed successfully!");
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Async rendering function.
///
/// The rasterizer initialization is async because wgpu device creation is async.
async fn run_render() -> oxigaf::Result<()> {
    // =========================================================================
    // Step 1: Create Gaussian model
    // =========================================================================

    println!("Creating test Gaussian model...");
    let model = create_test_gaussians();
    println!(
        "  Created {} Gaussians, SH degree {}",
        model.len(),
        model.sh_degree
    );
    println!("  SH coefficients: {} total", model.sh_coeffs.len());

    // =========================================================================
    // Step 2: Configure and initialize rasterizer
    // =========================================================================

    println!();
    println!("Initializing GPU rasterizer...");

    // Configure the rasterizer
    let config = RasterConfig::new()
        .with_resolution(512, 512) // Output image size
        .with_sh_degree(0) // Match model's SH degree
        .with_background([0.1, 0.1, 0.15]) // Dark blue-gray background
        .with_depth_output(true); // Enable depth buffer

    println!(
        "  Resolution: {}x{}",
        config.image_width, config.image_height
    );
    println!("  SH degree: {}", config.sh_degree);
    println!(
        "  Background: [{:.2}, {:.2}, {:.2}]",
        config.background[0], config.background[1], config.background[2]
    );

    // Initialize the rasterizer (async - requests GPU device)
    let mut rasterizer = Rasterizer::new(config.clone()).await?;
    println!("  GPU rasterizer initialized");

    // =========================================================================
    // Step 3: Set up camera
    // =========================================================================

    println!();
    println!("Setting up camera...");

    let camera = create_camera(config.image_width, config.image_height);
    println!(
        "  Camera position: [{:.2}, {:.2}, {:.2}]",
        camera.position[0], camera.position[1], camera.position[2]
    );
    println!(
        "  Focal length: [{:.1}, {:.1}]",
        camera.focal[0], camera.focal[1]
    );

    // =========================================================================
    // Step 4: Render frame
    // =========================================================================

    println!();
    println!("Rendering frame...");

    // Upload Gaussian data to GPU
    rasterizer.upload_gaussians(&model);
    println!("  Uploaded Gaussians to GPU");

    // Run forward pass
    let output = rasterizer.forward(&model, &camera)?;
    println!("  Forward pass complete");
    println!("  Output: {}x{} RGBA", output.width, output.height);

    // =========================================================================
    // Step 5: Save rendered image
    // =========================================================================

    println!();
    println!("Saving output...");

    // Convert to image
    let image = rasterizer.download_image(&output);

    // Save to temp directory
    let output_dir = std::env::temp_dir().join("oxigaf_examples");
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("Warning: Could not create output directory: {}", e);
    }

    let output_path = output_dir.join("gaussian_render.png");
    match image.save(&output_path) {
        Ok(()) => println!("  Saved render: {}", output_path.display()),
        Err(e) => eprintln!("  Error saving image: {}", e),
    }

    // Also save depth buffer as grayscale image
    let depth_path = output_dir.join("gaussian_depth.png");
    if let Err(e) = save_depth_image(&output.depth_data, output.width, output.height, &depth_path) {
        eprintln!("  Error saving depth: {}", e);
    } else {
        println!("  Saved depth: {}", depth_path.display());
    }

    // =========================================================================
    // Summary
    // =========================================================================

    println!();
    println!("Key takeaways:");
    println!("  - GaussianModel holds per-Gaussian attributes + SH coefficients");
    println!("  - RasterConfig controls resolution, SH degree, and options");
    println!("  - Rasterizer::new() is async (GPU device creation)");
    println!("  - forward() runs the full GPU rasterization pipeline");
    println!("  - download_image() converts GPU output to image::RgbaImage");
    println!();
    println!("Output files saved to: {}", output_dir.display());

    Ok(())
}

/// Save depth buffer as a grayscale PNG image.
///
/// Normalizes depth values to [0, 255] range for visualization.
fn save_depth_image(
    depth_data: &[f32],
    width: u32,
    height: u32,
    path: &std::path::Path,
) -> std::result::Result<(), String> {
    use image::{GrayImage, Luma};

    // Find min/max depth for normalization
    let mut min_depth = f32::INFINITY;
    let mut max_depth = f32::NEG_INFINITY;
    for &d in depth_data {
        if d.is_finite() && d > 0.0 {
            min_depth = min_depth.min(d);
            max_depth = max_depth.max(d);
        }
    }

    // Handle case where all depths are invalid
    if !min_depth.is_finite() || !max_depth.is_finite() || max_depth <= min_depth {
        min_depth = 0.0;
        max_depth = 1.0;
    }

    let range = max_depth - min_depth;

    // Create grayscale image
    let mut img = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let d = depth_data.get(idx).copied().unwrap_or(f32::INFINITY);

            let value = if d.is_finite() && d > 0.0 {
                // Invert so closer = brighter
                let normalized = 1.0 - (d - min_depth) / range;
                (normalized.clamp(0.0, 1.0) * 255.0) as u8
            } else {
                0 // Background (infinite depth) = black
            };

            img.put_pixel(x, y, Luma([value]));
        }
    }

    img.save(path).map_err(|e| e.to_string())
}
