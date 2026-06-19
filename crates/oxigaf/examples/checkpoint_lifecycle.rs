//! Checkpoint save → load → restore lifecycle example.
//!
//! Demonstrates the full round-trip:
//!   1. Build a small GaussianModel + GaussianOptimizer + MetricTracker
//!   2. Record synthetic metric entries
//!   3. Snapshot with `build_checkpoint`
//!   4. Persist to a temp file with `save_checkpoint`
//!   5. Reload with `load_checkpoint` (validates on deserialise)
//!   6. Restore model, optimizer states, and metrics history
//!   7. Verify that the restored iteration, Gaussian count, and metric
//!      count all match the original
//!
//! Fully CPU — no GPU or real assets required.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example checkpoint_lifecycle
//! ```

use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf::trainer::checkpoint::{
    build_checkpoint, load_checkpoint, restore_metrics, restore_model, restore_optimizer,
    save_checkpoint,
};
use oxigaf::trainer::metrics::MetricTracker;
use oxigaf::trainer::optimizer::GaussianOptimizer;
use oxigaf::trainer::OptimizerConfig;

// ---------------------------------------------------------------------------
// Helper: deterministic small GaussianModel
// ---------------------------------------------------------------------------

/// Build a `GaussianModel` with `n` Gaussians at SH degree 0 using a
/// linear congruential generator (no external RNG dependency).
fn build_small_model(n: usize) -> GaussianModel {
    let sh_degree: u32 = 0;
    let sh_per = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;

    let mut seed = 7919u64;
    let mut rng = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    let mut gaussians = Vec::with_capacity(n);
    let mut sh_coeffs = Vec::with_capacity(n * sh_per);

    for _ in 0..n {
        gaussians.push(GaussianAttributes {
            position: [rng() * 0.1, rng() * 0.1, rng() * 0.1],
            _pad0: 0.0,
            rotation: [rng() * 0.05, rng() * 0.05, rng() * 0.05, 1.0],
            scale: [-5.0, -5.0, -5.0], // exp(-5) ≈ 0.007
            opacity: -2.0,             // sigmoid(-2) ≈ 0.12
        });
        for _ in 0..sh_per {
            sh_coeffs.push(rng() * 0.1);
        }
    }

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: vec![0; n],
        barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; n],
        local_offsets: vec![[0.0, 0.0, 0.0]; n],
        is_rigid: vec![true; n],
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // Initialize logging (same style as basic_flame.rs / training_loop.rs)
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!("OxiGAF Checkpoint Resume Example");
    println!("=================================");
    println!();

    let tmp = std::env::temp_dir().join("oxigaf_checkpoint_lifecycle");
    std::fs::create_dir_all(&tmp).expect("failed to create temp dir");
    let ckpt_path = tmp.join("checkpoint_iter050.json");

    // =========================================================================
    // Step 1: Build a small GaussianModel (20 Gaussians)
    // =========================================================================

    println!("Step 1: Building GaussianModel (20 Gaussians)...");

    let num_gaussians: usize = 20;
    let model = build_small_model(num_gaussians);

    println!("  Gaussians: {}", model.len());
    println!("  SH degree: {}", model.sh_degree);
    println!("  SH coefficients total: {}", model.sh_coeffs.len());

    // =========================================================================
    // Step 2: Create optimizer
    // =========================================================================
    //
    // GaussianOptimizer::new(config, model) allocates per-parameter Adam state
    // sized to match the model.  The first argument is &OptimizerConfig, the
    // second is &GaussianModel.

    println!();
    println!("Step 2: Creating GaussianOptimizer...");

    let opt_config = OptimizerConfig::default();
    let optimizer = GaussianOptimizer::new(&opt_config, &model);

    let original_groups = optimizer.checkpoint_states();
    println!("  Optimizer parameter groups: {}", original_groups.len());
    for (name, m, _v, t) in &original_groups {
        println!("    - {}: {} params, step={}", name, m.len(), t);
    }

    // =========================================================================
    // Step 3: Create MetricTracker and record synthetic iterations
    // =========================================================================

    println!();
    println!("Step 3: Recording synthetic metrics (5 entries)...");

    let mut metrics = MetricTracker::new();
    let synthetic_entries: [(u32, f32, f32, f32); 5] = [
        (10, 18.5, 0.72, 0.045),
        (20, 20.1, 0.78, 0.038),
        (30, 21.4, 0.81, 0.032),
        (40, 22.0, 0.83, 0.029),
        (50, 22.8, 0.84, 0.026),
    ];
    for (iter, psnr, ssim, loss) in synthetic_entries {
        metrics.record(iter, psnr, ssim, loss);
    }

    println!("  Recorded {} metric entries.", metrics.len());
    if let Some(latest) = metrics.latest() {
        println!(
            "  Latest — iter={} psnr={:.2} ssim={:.3} loss={:.4}",
            latest.iteration, latest.psnr, latest.ssim, latest.loss
        );
    }

    let simulation_iteration: u32 = 50;

    // =========================================================================
    // Step 4: Build + save checkpoint
    // =========================================================================

    println!();
    println!("Step 4: Building and saving checkpoint...");

    let checkpoint_data = build_checkpoint(&model, &optimizer, simulation_iteration, &metrics);

    println!("  CheckpointData:");
    println!("    version:          {}", checkpoint_data.version);
    println!("    iteration:        {}", checkpoint_data.iteration);
    println!("    positions:        {}", checkpoint_data.positions.len());
    println!(
        "    optimizer groups: {}",
        checkpoint_data.optimizer_groups.len()
    );
    println!(
        "    metrics history:  {}",
        checkpoint_data.metrics_history.len()
    );

    save_checkpoint(&ckpt_path, &checkpoint_data).expect("failed to save checkpoint");

    let meta = std::fs::metadata(&ckpt_path).expect("stat ckpt");
    println!("  Saved to: {}", ckpt_path.display());
    println!("  File size: {} bytes", meta.len());

    // =========================================================================
    // Step 5: Load checkpoint + validate
    // =========================================================================
    //
    // load_checkpoint() reads JSON, deserialises, and calls validate() which
    // checks version compatibility, array length consistency, and NaN/Inf.

    println!();
    println!("Step 5: Loading and validating checkpoint...");

    let loaded = load_checkpoint(&ckpt_path).expect("failed to load checkpoint");

    // validate() is called internally by load_checkpoint; call again to show
    // it is idempotent and can be used as an explicit integrity gate.
    loaded.validate().expect("checkpoint validation failed");

    println!("  Loaded checkpoint is valid.");
    println!("    version:   {}", loaded.version);
    println!("    iteration: {}", loaded.iteration);
    println!("    Gaussians: {}", loaded.positions.len());

    // =========================================================================
    // Step 6: Restore model, optimizer, and metrics
    // =========================================================================

    println!();
    println!("Step 6: Restoring model, optimizer, and metrics...");

    let restored_model = restore_model(&loaded);
    println!(
        "  Restored model: {} Gaussians (expected {})",
        restored_model.len(),
        num_gaussians
    );

    let mut restored_optimizer = GaussianOptimizer::new(&opt_config, &restored_model);
    restore_optimizer(&loaded, &mut restored_optimizer);
    let restored_groups = restored_optimizer.checkpoint_states();
    println!(
        "  Restored optimizer groups: {} (expected {})",
        restored_groups.len(),
        original_groups.len()
    );

    let restored_metrics = restore_metrics(&loaded);
    println!(
        "  Restored metric entries: {} (expected {})",
        restored_metrics.len(),
        metrics.len()
    );

    // =========================================================================
    // Step 7: Verify round-trip integrity
    // =========================================================================

    println!();
    println!("Step 7: Verifying round-trip integrity...");

    assert_eq!(loaded.iteration, simulation_iteration, "Iteration mismatch");
    assert_eq!(
        restored_model.len(),
        num_gaussians,
        "Gaussian count mismatch"
    );
    assert_eq!(
        restored_groups.len(),
        original_groups.len(),
        "Optimizer group count mismatch"
    );
    assert_eq!(
        restored_metrics.len(),
        metrics.len(),
        "Metric entry count mismatch"
    );

    println!("  [OK] iteration restored: {}", loaded.iteration);
    println!("  [OK] Gaussian count:     {}", restored_model.len());
    println!("  [OK] optimizer groups:   {}", restored_groups.len());
    println!("  [OK] metric entries:     {}", restored_metrics.len());

    if let Some(latest) = restored_metrics.latest() {
        println!(
            "  [OK] latest metric — iter={} psnr={:.2} ssim={:.3} loss={:.4}",
            latest.iteration, latest.psnr, latest.ssim, latest.loss
        );
    }

    // =========================================================================
    // Bonus: how to resume full training from a checkpoint (GPU required)
    // =========================================================================
    //
    // ```rust,ignore
    // use oxigaf::prelude::*;
    //
    // let (device, queue) = adapter.request_device(...).await?;
    // let raster_config = RasterConfig::new().with_resolution(512, 512);
    // let training_config = TrainingConfig { total_iterations: 30_000, ..Default::default() };
    //
    // let mut trainer = Trainer::from_checkpoint(
    //     &ckpt_path,
    //     training_config,
    //     raster_config,
    //     device,
    //     queue,
    // )?;
    //
    // // Resumes from iteration 50 (trainer.current_iteration() == 50)
    // for _ in trainer.current_iteration()..training_config.total_iterations {
    //     let output = trainer.train_step()?;
    //     println!("iter {} loss {:.6}", output.iteration, output.loss.total);
    // }
    // ```

    println!();
    println!("  (See commented-out snippet in source for Trainer::from_checkpoint usage)");

    // Clean up
    let _ = std::fs::remove_dir_all(&tmp);

    // =========================================================================
    // Key Takeaways footer
    // =========================================================================

    println!();
    println!("Key Takeaways:");
    println!("  - build_checkpoint(&model, &optimizer, iter, &metrics) snapshots live state");
    println!("  - save_checkpoint(&path, &data)  serialises to JSON");
    println!("  - load_checkpoint(&path)          deserialises + validates automatically");
    println!("  - CheckpointData::validate()      checks version, array lengths, NaN/Inf");
    println!("  - restore_model(&data)            reconstructs GaussianModel from snapshot");
    println!("  - restore_optimizer(&data, &mut)  replays Adam m/v/t, no warm-up needed");
    println!("  - restore_metrics(&data)          rebuilds MetricTracker with full history");
    println!("  - Trainer::from_checkpoint()      is the production API (requires live GPU)");
}
