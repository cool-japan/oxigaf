//! End-to-end OxiGAF pipeline example.
//!
//! Demonstrates the full FLAME → diffusion → render → train → export pipeline
//! using the PipelineBuilder API. Heavy operations (GPU rendering, actual training)
//! are guarded so this example compiles and runs on any machine.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example end_to_end_pipeline
//! ```

use std::path::PathBuf;

fn main() -> oxigaf::Result<()> {
    // Initialize logging (same style as basic_flame.rs)
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!("OxiGAF End-to-End Pipeline Example");
    println!("====================================");
    println!();

    // =========================================================================
    // Step 1: Build pipeline configuration
    // =========================================================================
    //
    // PipelineBuilder is the single entry point for the full OxiGAF workflow.
    // Every option has a sensible default; only flame_model_path and output_dir
    // are required.

    println!("Step 1: Building pipeline configuration...");

    let output_dir = std::env::temp_dir().join("oxigaf_e2e_example");
    let flame_model_path = PathBuf::from("data/flame_model");

    // Attempt to create a validated PipelineConfig with all options set.
    // If the paths don't exist the build() call still succeeds — only
    // validate_config() enforces path presence on disk.
    let config = match oxigaf::PipelineBuilder::new()
        .flame_model_path(&flame_model_path)
        .output_dir(&output_dir)
        .num_views(8)
        .iterations(30_000)
        .build()
    {
        Ok(c) => {
            println!("  PipelineConfig built successfully");
            println!("    flame_model_path: {}", c.flame_model_path.display());
            println!("    output_dir:       {}", c.output_dir.display());
            println!("    num_views:        {}", c.num_views);
            println!("    iterations:       {}", c.iterations);
            c
        }
        Err(e) => {
            // This branch is unreachable for valid paths/counts, but handles
            // the case where num_views or iterations would be zero.
            eprintln!("  Pipeline config error: {}", e);
            return Err(e);
        }
    };

    // =========================================================================
    // Step 2: Validate configuration (always safe to call)
    // =========================================================================
    //
    // validate_config() checks that:
    //   - flame_model_path exists on disk
    //   - num_views >= 1
    //   - iterations >= 1

    println!();
    println!("Step 2: Validating configuration...");

    match oxigaf::validate_config(&config) {
        Ok(()) => {
            println!("  Configuration is valid — all paths exist and values in range.");
        }
        Err(oxigaf::OxigafError::PathNotFound(ref p)) => {
            println!(
                "  Note: flame_model_path does not exist yet: {} (expected in production)",
                p
            );
            println!("  Continuing in demo mode — GPU-heavy steps will be skipped.");
        }
        Err(e) => return Err(e),
    }

    // =========================================================================
    // Step 3: Check for missing asset files
    // =========================================================================
    //
    // verify_assets() returns the names of FLAME .npy files that are absent
    // from the model directory. An empty list means the directory is complete.

    println!();
    println!("Step 3: Verifying FLAME assets...");

    let missing = oxigaf::verify_assets(&config.flame_model_path);
    if missing.is_empty() {
        println!("  All expected FLAME asset files are present.");
    } else {
        println!("  Missing assets ({}):", missing.len());
        for name in &missing {
            println!("    - {}", name);
        }
        println!("  Download FLAME from https://flame.is.tue.mpg.de/ and convert to .npy");
    }

    // =========================================================================
    // Step 4: Detect GPU adapters
    // =========================================================================
    //
    // check_gpu() enumerates wgpu adapters. On headless / CI machines the
    // returned vector may be empty, and we fall back gracefully to demo mode.

    println!();
    println!("Step 4: Detecting GPU adapters...");

    let gpus = oxigaf::check_gpu()?;

    if gpus.is_empty() {
        println!("  No GPU adapter found — running in demo mode.");
        println!("  (Rendering and training steps will be skipped.)");
    } else {
        println!("  Found {} GPU adapter(s):", gpus.len());
        for (i, gpu) in gpus.iter().enumerate() {
            println!(
                "    [{}] {} ({}) via {}",
                i, gpu.name, gpu.device_type, gpu.backend
            );
        }
    }

    let has_gpu = !gpus.is_empty();

    // =========================================================================
    // Step 5: Detect best backend
    // =========================================================================
    //
    // detect_best_backend() uses compile-time target-OS information to choose
    // the preferred wgpu backend without performing any driver enumeration.

    println!();
    println!("Step 5: Detecting best wgpu backend...");

    let backend = oxigaf::detect_best_backend();
    println!("  Preferred backend for this platform: {}", backend);

    // =========================================================================
    // Step 6: Quick-train (validation + path resolution; safe on any machine)
    // =========================================================================
    //
    // quick_train() validates the config and returns the resolved output_dir.
    // It does *not* invoke the GPU rasterizer or Adam optimiser; that is
    // handled by oxigaf_trainer::Trainer. Here we call it only to demonstrate
    // the convenience API and show the resolved output path.

    println!();
    println!("Step 6: Running quick_train (config validation)...");

    // Ensure output directory exists so quick_train can resolve it.
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("  Warning: could not create output directory: {}", e);
    }

    // quick_train requires flame_model_path to exist; if it doesn't, we skip.
    if flame_model_path.exists() {
        match oxigaf::quick_train(&flame_model_path, &output_dir) {
            Ok(resolved) => println!("  Output directory resolved to: {}", resolved.display()),
            Err(e) => println!(
                "  quick_train reported: {} (expected without real assets)",
                e
            ),
        }
    } else {
        println!(
            "  Skipping quick_train — flame_model_path not present ({})",
            flame_model_path.display()
        );
    }

    // =========================================================================
    // Step 7: Export demonstration (guarded by output_dir existence)
    // =========================================================================
    //
    // export() reads the model at model_path (parsed by its `.ply` or
    // `.safetensors` extension) and re-serialises it to output_path in the
    // given format. This is a real, synchronous CPU-side write — it does not
    // require a GPU runtime. We guard the call with an existence check so
    // this example never panics on a clean machine, and export to `.obj`
    // (not `.safetensors`, which export() only ever reads, never writes) so
    // the output extension actually matches ExportFormat::Obj.

    println!();
    println!("Step 7: Demonstrating export API...");

    let model_ply = output_dir.join("model.ply");
    let export_obj = output_dir.join("model.obj");

    if model_ply.exists() {
        match oxigaf::export(&model_ply, &export_obj, oxigaf::ExportFormat::Obj) {
            Ok(()) => println!(
                "  Exported {} -> {}",
                model_ply.display(),
                export_obj.display()
            ),
            Err(e) => println!("  Export failed: {} (expected without real model)", e),
        }
    } else {
        println!(
            "  Skipping export — model.ply not present at {}",
            model_ply.display()
        );
        println!("  In production, export() accepts ExportFormat::{{Ply, Gltf, Obj}}.");
    }

    // =========================================================================
    // GPU-gated: show what full pipeline steps would do
    // =========================================================================

    if has_gpu {
        println!();
        println!("GPU detected — in a full pipeline run you would:");
        println!("  1. Load FLAME model:        FlameModel::load(&config.flame_model_path)");
        println!("  2. Diffusion inference:     MultiViewDiffusionPipeline::new(diffusion_cfg)");
        println!("  3. Init Gaussians:          oxigaf_trainer::init::initialize_gaussians(...)");
        println!("  4. Build wgpu device:       adapter.request_device(...)");
        println!(
            "  5. Create Trainer:          Trainer::new(training_cfg, model, raster_cfg, ...)"
        );
        println!(
            "  6. Train loop:              for _ in 0..cfg.total_iterations {{ trainer.train_step()? }}"
        );
        println!("  7. Export model:            export(&ply_path, &out_path, ExportFormat::Ply)");
    }

    // =========================================================================
    // Key Takeaways footer
    // =========================================================================

    println!();
    println!("Key Takeaways:");
    println!("  - PipelineBuilder::new()  constructs a validated PipelineConfig");
    println!("  - validate_config(&cfg)   checks paths exist and values are in range");
    println!("  - verify_assets(&dir)     lists missing FLAME .npy files");
    println!("  - check_gpu()             enumerates wgpu adapters (may be empty on CI)");
    println!("  - detect_best_backend()   returns Metal/Vulkan/Dx12 based on OS");
    println!("  - quick_train(&p, &out)   validates config and resolves output path");
    println!("  - export(&src, &dst, fmt) reads &src and re-serialises it to &dst in `fmt`");
    println!();
    println!("For a full GPU training loop, see the `training_loop` example.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

    /// Regression test for the Step-7 format/extension pairing bug: exporting
    /// with `ExportFormat::Obj` must write real OBJ text to the `.obj`
    /// destination, not raw PLY bytes carried over under a mismatched
    /// extension (which is what the previous `.safetensors` destination +
    /// `ExportFormat::Ply` pairing would have produced).
    #[test]
    fn export_obj_writes_real_obj_not_raw_ply_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "oxigaf_e2e_export_regression_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let model = GaussianModel {
            gaussians: vec![GaussianAttributes {
                position: [0.1, 0.2, 0.3],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-5.0, -5.0, -5.0],
                opacity: -2.0,
            }],
            sh_coeffs: vec![0.1, 0.1, 0.1],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
            local_offsets: vec![[0.0, 0.0, 0.0]],
            is_rigid: vec![true],
        };

        let model_ply = dir.join("model.ply");
        model.save_ply(&model_ply).expect("save_ply should succeed");

        let export_obj = dir.join("model.obj");
        oxigaf::export(&model_ply, &export_obj, oxigaf::ExportFormat::Obj)
            .expect("export to Obj should succeed");

        let obj_bytes = std::fs::read(&export_obj).expect("read exported obj");
        let obj_text = String::from_utf8(obj_bytes).expect("obj output should be UTF-8 text");

        assert!(
            obj_text.starts_with("# Wavefront OBJ"),
            "expected OBJ output to start with the Wavefront OBJ comment header, got: {:?}",
            &obj_text[..obj_text.len().min(40)]
        );
        assert!(
            !obj_text.starts_with("ply\n") && !obj_text.starts_with("ply\r\n"),
            "exported .obj file must not contain raw PLY bytes"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
