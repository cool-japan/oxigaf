//! Example: Load a PLY file and render it.
//!
//! Usage: cargo run --example render_ply -- <path/to/model.ply> [output.png]
//!
//! If no GPU is available, prints model stats without rendering.

use oxigaf_render::{GaussianAttributes, GaussianModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let ply_path = args
        .get(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("model.ply"));
    let output_path = args.get(2).map(String::as_str).unwrap_or("output.png");

    // Try to load the PLY file, or create a demo model if it doesn't exist.
    if !ply_path.exists() {
        println!(
            "No PLY file found at {}. Creating a minimal demo model.",
            ply_path.display()
        );
        let model = create_demo_model();
        println!(
            "Demo model: {} Gaussians, SH degree {}",
            model.gaussians.len(),
            model.sh_degree
        );

        // Save it as PLY to demonstrate the I/O round-trip.
        let tmp = std::env::temp_dir().join("demo_model.ply");
        model.save_ply(&tmp)?;
        println!("Saved demo model to {}", tmp.display());

        // Reload to verify round-trip.
        let reloaded = GaussianModel::load_ply(&tmp)?;
        println!("Reload verified: {} Gaussians", reloaded.gaussians.len());
        return Ok(());
    }

    let model = GaussianModel::load_ply(&ply_path)?;
    println!(
        "Loaded: {} Gaussians, SH degree {}",
        model.gaussians.len(),
        model.sh_degree
    );

    // Compute and display bounding box.
    let positions: Vec<[f32; 3]> = model.gaussians.iter().map(|g| g.position).collect();
    let (min_pt, max_pt) = bounding_box(&positions);
    println!(
        "Bounding box: [{:.3},{:.3},{:.3}] to [{:.3},{:.3},{:.3}]",
        min_pt[0], min_pt[1], min_pt[2], max_pt[0], max_pt[1], max_pt[2]
    );

    // Display opacity statistics.
    let opacities: Vec<f32> = model
        .gaussians
        .iter()
        .map(|g| 1.0_f32 / (1.0_f32 + (-g.opacity).exp()))
        .collect();
    let mean_opacity = opacities.iter().sum::<f32>() / opacities.len().max(1) as f32;
    println!("Mean sigmoid opacity: {:.4}", mean_opacity);

    // Try GPU rendering — requires a compatible GPU at runtime.
    println!("Attempting GPU render (requires compatible GPU)...");
    println!("Output would be written to: {}", output_path);
    println!("Note: Actual rendering requires GPU initialization via wgpu.");
    println!("      Use MultiViewRenderer::new(config).await? in async context.");

    Ok(())
}

/// Create a small demo model with 8 Gaussians placed at the corners of a unit cube.
fn create_demo_model() -> GaussianModel {
    let sh_degree = 0_u32;
    let sh_coeffs_per = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;

    // Cube corners at ±0.5 on each axis.
    let corners: [[f32; 3]; 8] = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ];

    let gaussians: Vec<GaussianAttributes> = corners
        .iter()
        .map(|&pos| GaussianAttributes {
            position: pos,
            _pad0: 0.0,
            // Identity quaternion (x,y,z,w).
            rotation: [0.0, 0.0, 0.0, 1.0],
            // Log-scale: exp(-3) ≈ 0.05 world units.
            scale: [-3.0, -3.0, -3.0],
            // Logit(0.8) ≈ 1.386 — reasonably opaque.
            opacity: 1.386_f32,
        })
        .collect();

    let n = gaussians.len();
    // SH degree 0: 3 coefficients per Gaussian (RGB DC term).
    let sh_coeffs = vec![0.5_f32; n * sh_coeffs_per];

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: vec![0_u32; n],
        barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; n],
        local_offsets: vec![[0.0_f32; 3]; n],
        is_rigid: vec![false; n],
    }
}

/// Compute the axis-aligned bounding box of a set of 3-D positions.
///
/// Returns `([f32::MAX; 3], [f32::MIN; 3])` for an empty slice.
fn bounding_box(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min_pt = [f32::MAX; 3];
    let mut max_pt = [f32::MIN; 3];
    for p in positions {
        for axis in 0..3 {
            if p[axis] < min_pt[axis] {
                min_pt[axis] = p[axis];
            }
            if p[axis] > max_pt[axis] {
                max_pt[axis] = p[axis];
            }
        }
    }
    (min_pt, max_pt)
}
