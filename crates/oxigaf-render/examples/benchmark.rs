//! Example: Benchmark CPU-side Gaussian operations.
//!
//! Usage: cargo run --example benchmark [num_gaussians]
//!
//! Tests: PLY I/O, density control, profiler timing.

use oxigaf_render::{
    DensityConfig, DensityController, GaussianAttributes, GaussianModel, PassProfiler,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    println!("Benchmarking with {} Gaussians", n);

    let profiler = PassProfiler::new();

    // Benchmark model creation.
    let model = profiler.time("create_model", || create_large_model(n));
    println!("Created model: {} Gaussians", model.gaussians.len());

    // Benchmark PLY I/O.
    let tmp = std::env::temp_dir().join("bench_model.ply");
    profiler.time("save_ply", || model.save_ply(&tmp))?;

    let loaded = profiler.time("load_ply", || GaussianModel::load_ply(&tmp))?;
    println!("Loaded model: {} Gaussians", loaded.gaussians.len());

    // Benchmark density control.
    let density_config = DensityConfig::default();
    let mut controller = DensityController::new(model.gaussians.len(), density_config);

    // Simulate high gradients for all Gaussians (above default threshold 2e-4).
    let grad_norms: Vec<f32> = vec![1.0_f32; n];
    controller.accumulator.accumulate(&grad_norms);

    let (densified, _new_acc) =
        profiler.time("clone_gaussians", || controller.clone_gaussians(&model));
    println!(
        "After cloning: {} Gaussians (were {})",
        densified.gaussians.len(),
        n
    );

    // Prune — all Gaussians have high opacity (logit(0.8) ≈ 1.386) so most survive.
    let pruned = profiler.time("prune_gaussians", || {
        controller.prune_gaussians(&model, None)
    });
    println!("After pruning: {} Gaussians", pruned.gaussians.len());

    // Benchmark split operation as well.
    // Reset accumulator and re-accumulate with high grads on a fresh controller.
    let mut split_controller =
        DensityController::new(model.gaussians.len(), DensityConfig::default());
    split_controller.accumulator.accumulate(&grad_norms);

    let (split_model, _) = profiler.time("split_gaussians", || {
        split_controller.split_gaussians(&model)
    });
    println!(
        "After splitting: {} Gaussians (were {})",
        split_model.gaussians.len(),
        n
    );

    // Advance frame counter and print report.
    profiler.next_frame();
    println!("\n{}", profiler.format_report());

    Ok(())
}

/// Create a model with `n` Gaussians arranged in a regular grid.
///
/// All Gaussians use log-scale `-3.0` (≈ 0.05 world units) and a high
/// opacity (logit ≈ 1.386 → sigmoid ≈ 0.8) so that pruning keeps them.
/// The scale is chosen to be below the default `scale_split_threshold`
/// (0.01 world units after exponentiation), making them candidates for
/// *cloning* rather than splitting.
fn create_large_model(n: usize) -> GaussianModel {
    let sh_degree = 0_u32;
    let sh_coeffs_per = 3_usize; // (0+1)^2 * 3

    let side = (n as f64).cbrt().ceil() as usize;
    let spacing = 0.1_f32;

    let gaussians: Vec<GaussianAttributes> = (0..n)
        .map(|i| {
            let ix = i % side;
            let iy = (i / side) % side;
            let iz = i / (side * side);
            GaussianAttributes {
                position: [
                    ix as f32 * spacing,
                    iy as f32 * spacing,
                    iz as f32 * spacing,
                ],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                // exp(-3) ≈ 0.050 < scale_split_threshold 0.01? Actually exp(-3) ≈ 0.0498 > 0.01
                // Use -6 so exp(-6) ≈ 0.0025 < 0.01 → clone path.
                scale: [-6.0, -6.0, -6.0],
                opacity: 1.386_f32,
            }
        })
        .collect();

    let count = gaussians.len();
    GaussianModel {
        gaussians,
        sh_coeffs: vec![0.5_f32; count * sh_coeffs_per],
        sh_degree,
        face_indices: vec![0_u32; count],
        barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; count],
        local_offsets: vec![[0.0_f32; 3]; count],
        is_rigid: vec![false; count],
    }
}
