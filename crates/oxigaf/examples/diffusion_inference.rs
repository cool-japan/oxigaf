//! # Multi-View Diffusion Inference Example
//!
//! This example demonstrates the multi-view diffusion pipeline:
//!
//! 1. Load the diffusion model (CLIP + U-Net + VAE)
//! 2. Prepare reference image conditioning
//! 3. Encode camera poses and normal maps
//! 4. Generate multi-view images with DDIM sampling
//! 5. Save output views to disk
//!
//! ## Pipeline Architecture
//!
//! The multi-view diffusion model follows this flow:
//!
//! ```text
//! Reference Image
//!       |
//!       v
//! [CLIP Encoder] --> IP Tokens
//!       |
//!       v
//! [Multi-View U-Net] <-- Camera Poses
//!       |                    ^
//!       v                    |
//! [DDIM Scheduler] <-- Normal Map Latents
//!       |
//!       v
//! [VAE Decoder]
//!       |
//!       v
//! Generated Views (N x H x W x 3)
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo run --example diffusion_inference -- --weights-dir /path/to/weights
//! ```
//!
//! Note: Requires pre-trained model weights in safetensors format.
//! See the project README for weight download instructions.

use std::path::{Path, PathBuf};

use oxigaf::prelude::*;

/// Parse command-line arguments.
fn parse_args() -> (PathBuf, Option<PathBuf>) {
    let args: Vec<String> = std::env::args().collect();
    let mut weights_dir = PathBuf::from("data/diffusion_weights");
    let mut reference_image = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--weights-dir" => {
                if let Some(path) = args.get(i + 1) {
                    weights_dir = PathBuf::from(path);
                    i += 1;
                }
            }
            "--reference" => {
                if let Some(path) = args.get(i + 1) {
                    reference_image = Some(PathBuf::from(path));
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    (weights_dir, reference_image)
}

fn main() {
    println!("OxiGAF Multi-View Diffusion Inference Example");
    println!("==============================================");
    println!();

    let (weights_dir, reference_image) = parse_args();

    println!("Weights directory: {}", weights_dir.display());
    if let Some(ref img) = reference_image {
        println!("Reference image: {}", img.display());
    }
    println!();

    // Check if weights exist, otherwise run demo mode
    if !weights_dir.exists() || !weights_dir.join("unet").exists() {
        println!("Model weights not found at: {}", weights_dir.display());
        println!();
        println!("Running in demonstration mode (API showcase only).");
        println!();
        demonstrate_api();
        return;
    }

    // Run the full inference pipeline
    match run_inference(&weights_dir, reference_image.as_deref()) {
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

/// Run the full diffusion inference pipeline.
fn run_inference(weights_dir: &Path, _reference_image: Option<&Path>) -> oxigaf::Result<()> {
    // =========================================================================
    // Step 1: Configure diffusion pipeline
    // =========================================================================

    println!("Configuring diffusion pipeline...");

    let config = DiffusionConfig {
        num_views: 4,            // Generate 4 views
        guidance_scale: 3.0,     // Classifier-free guidance scale
        num_inference_steps: 50, // DDIM steps
        image_size: 256,         // Output resolution
        ..Default::default()
    };

    println!("  Views: {}", config.num_views);
    println!("  Guidance scale: {}", config.guidance_scale);
    println!("  Inference steps: {}", config.num_inference_steps);
    println!("  Image size: {}x{}", config.image_size, config.image_size);

    // =========================================================================
    // Step 2: Select compute device
    // =========================================================================

    println!();
    println!("Selecting compute device...");

    // Try to use GPU if available, fallback to CPU
    let device = select_device();
    println!("  Using device: {:?}", device);

    // =========================================================================
    // Step 3: Load diffusion pipeline
    // =========================================================================

    println!();
    println!("Loading diffusion pipeline...");
    println!("  This may take a moment...");

    let mut pipeline = MultiViewDiffusionPipeline::load(config.clone(), weights_dir, &device)?;

    println!("  Pipeline loaded successfully");
    println!("  - CLIP encoder: ready");
    println!("  - U-Net: ready");
    println!("  - VAE decoder: ready");
    println!("  - DDIM scheduler: {} steps", config.num_inference_steps);

    // =========================================================================
    // Step 4: Prepare inputs
    // =========================================================================

    println!();
    println!("Preparing inputs...");

    // Create dummy reference image tensor (in production, load from file)
    // Shape: (1, 3, 224, 224) normalized to [-1, 1]
    let reference_image = create_dummy_reference_image(&device)?;
    println!("  Reference image: 1x3x224x224");

    // Create normal map latents (in production, encode from rendered normal maps)
    // Shape: (num_views, latent_channels, latent_size, latent_size)
    let normal_map_latents = create_dummy_normal_latents(&config, &device)?;
    println!(
        "  Normal latents: {}x{}x{}x{}",
        config.num_views, config.latent_channels, config.latent_size, config.latent_size
    );

    // Create camera poses (flattened 3x4 matrices)
    // Shape: (num_views, 12)
    let camera_poses = create_camera_poses(&config, &device)?;
    println!(
        "  Camera poses: {}x{}",
        config.num_views, config.camera_pose_dim
    );

    // =========================================================================
    // Step 5: Run inference
    // =========================================================================

    println!();
    println!("Running multi-view generation...");
    match &device {
        candle_core::Device::Cpu => println!("  (This may take several minutes on CPU)"),
        other => println!("  (Using {:?} — should be faster than CPU)", other),
    }

    let seed = 42u64;
    let output = pipeline.generate(&reference_image, &normal_map_latents, &camera_poses, seed)?;

    println!();
    println!("Generation complete!");
    println!(
        "  Generated {} views at {}x{}",
        output.images.len(),
        output.width,
        output.height
    );

    // =========================================================================
    // Step 6: Save outputs
    // =========================================================================

    println!();
    println!("Saving output views...");

    let output_dir = std::env::temp_dir()
        .join("oxigaf_examples")
        .join("diffusion");
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("Warning: Could not create output directory: {}", e);
    }

    for (i, image_tensor) in output.images.iter().enumerate() {
        let output_path = output_dir.join(format!("view_{}.png", i));

        match save_tensor_as_image(image_tensor, &output_path) {
            Ok(()) => println!("  Saved view {}: {}", i, output_path.display()),
            Err(e) => eprintln!("  Error saving view {}: {}", i, e),
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================

    println!();
    println!("Key takeaways:");
    println!("  - DiffusionConfig controls generation parameters");
    println!("  - MultiViewDiffusionPipeline::load() loads model weights");
    println!("  - generate() runs the full DDIM denoising loop");
    println!("  - Output contains multiple view tensors ready for training");
    println!();
    println!("Output files saved to: {}", output_dir.display());

    Ok(())
}

/// Demonstrate the API without requiring model weights.
fn demonstrate_api() {
    println!("API Demonstration");
    println!("-----------------");
    println!();

    // Show configuration options
    println!("1. DiffusionConfig options:");
    let config = DiffusionConfig::default();
    println!(
        "   num_views: {} (multi-view generation count)",
        config.num_views
    );
    println!(
        "   guidance_scale: {} (classifier-free guidance)",
        config.guidance_scale
    );
    println!(
        "   num_inference_steps: {} (DDIM denoising steps)",
        config.num_inference_steps
    );
    println!("   image_size: {} (output resolution)", config.image_size);
    println!(
        "   latent_size: {} (VAE latent size = image_size/8)",
        config.latent_size
    );
    println!(
        "   latent_channels: {} (VAE latent channels)",
        config.latent_channels
    );
    println!(
        "   cross_attention_dim: {} (text/image conditioning dim)",
        config.cross_attention_dim
    );
    println!(
        "   clip_embed_dim: {} (CLIP embedding dimension)",
        config.clip_embed_dim
    );

    // Show scheduler options
    println!();
    println!("2. DDIM Scheduler:");
    let scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);
    println!("   Total timesteps: 1000");
    println!("   Prediction type: VPrediction (velocity prediction)");
    println!("   Alternative: Epsilon (noise prediction)");

    // Configure inference steps
    let mut scheduler = scheduler;
    scheduler.set_timesteps(50);
    let timesteps = scheduler.timesteps();
    println!(
        "   Inference timesteps (first 5): {:?}...",
        &timesteps[..5.min(timesteps.len())]
    );

    // Show camera pose format
    println!();
    println!("3. Camera Pose Format:");
    println!("   Shape: (num_views, 12)");
    println!("   Each camera: flattened 3x4 extrinsic matrix [R|t]");
    println!("   Row-major order: [r00, r01, r02, tx, r10, r11, r12, ty, ...]");
    println!();
    println!("   Example poses for 4-view setup:");
    for i in 0..4 {
        let angle = i as f32 * std::f32::consts::PI / 2.0;
        println!(
            "     View {}: azimuth = {:.0} degrees",
            i,
            angle.to_degrees()
        );
    }

    // Show pipeline loading requirements
    println!();
    println!("4. Required weight files:");
    println!("   weights_dir/");
    println!("   ├── unet/diffusion_pytorch_model.safetensors");
    println!("   ├── vae/diffusion_pytorch_model.safetensors");
    println!("   └── image_encoder/model.safetensors");

    // Show example usage
    println!();
    println!("5. Example code:");
    println!("   ```rust");
    println!("   let config = DiffusionConfig::default();");
    println!("   let device = candle_core::Device::Cpu;");
    println!("   ");
    println!("   let mut pipeline = MultiViewDiffusionPipeline::load(");
    println!("       config,");
    println!("       Path::new(\"weights\"),");
    println!("       &device,");
    println!("   )?;");
    println!("   ");
    println!("   let output = pipeline.generate(");
    println!("       &reference_image,    // (1, 3, 224, 224)");
    println!("       &normal_latents,     // (N, 4, 32, 32)");
    println!("       &camera_poses,       // (N, 12)");
    println!("       seed,                // u64");
    println!("   )?;");
    println!("   ```");

    println!();
    println!("To run with actual model weights:");
    println!("  cargo run --example diffusion_inference -- --weights-dir /path/to/weights");
}

/// Select the best available compute device.
///
/// Prefers Metal, then CUDA, falling back to CPU. Which (if either) GPU
/// backend is actually compiled into `candle-core` is a build-time choice:
/// this crate's `Cargo.toml` does not enable candle's `cuda`/`metal`
/// features, so in the default build `metal_is_available()` /
/// `cuda_is_available()` are both `false` and this always resolves to CPU.
/// To get a real GPU device here, enable the corresponding feature on
/// `candle-core` directly in your own Cargo.toml (Cargo's feature
/// unification then turns it on for this dependency across the build) —
/// `Device::new_metal`/`Device::new_cuda` will then succeed instead of
/// returning `Err(NotCompiledWith*Support)`.
fn select_device() -> candle_core::Device {
    if candle_core::utils::metal_is_available() {
        if let Ok(device) = candle_core::Device::new_metal(0) {
            return device;
        }
    }
    if candle_core::utils::cuda_is_available() {
        if let Ok(device) = candle_core::Device::new_cuda(0) {
            return device;
        }
    }
    candle_core::Device::Cpu
}

/// Create a dummy reference image tensor for demonstration.
fn create_dummy_reference_image(
    device: &candle_core::Device,
) -> oxigaf::Result<candle_core::Tensor> {
    use candle_core::Tensor;

    // Create normalized image tensor: (1, 3, 224, 224) in range [-1, 1]
    Tensor::zeros((1, 3, 224, 224), candle_core::DType::F32, device).map_err(|e| {
        oxigaf::OxigafError::Diffusion(DiffusionError::Inference(format!(
            "Failed to create reference tensor: {}",
            e
        )))
    })
}

/// Create dummy normal map latents for demonstration.
fn create_dummy_normal_latents(
    config: &DiffusionConfig,
    device: &candle_core::Device,
) -> oxigaf::Result<candle_core::Tensor> {
    use candle_core::Tensor;

    // Normal map latents: (num_views, latent_channels, latent_size, latent_size)
    Tensor::zeros(
        (
            config.num_views,
            config.latent_channels,
            config.latent_size,
            config.latent_size,
        ),
        candle_core::DType::F32,
        device,
    )
    .map_err(|e| {
        oxigaf::OxigafError::Diffusion(DiffusionError::Inference(format!(
            "Failed to create normal latents: {}",
            e
        )))
    })
}

/// Create camera poses for multi-view generation.
///
/// Generates camera poses arranged uniformly around the subject.
fn create_camera_poses(
    config: &DiffusionConfig,
    device: &candle_core::Device,
) -> oxigaf::Result<candle_core::Tensor> {
    use candle_core::Tensor;

    let mut poses = Vec::with_capacity(config.num_views * config.camera_pose_dim);

    for i in 0..config.num_views {
        // Compute azimuth angle for this view
        let angle = (i as f32 / config.num_views as f32) * 2.0 * std::f32::consts::PI;

        // Rotation matrix for y-axis rotation (looking at origin from distance)
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // 3x4 extrinsic matrix [R|t], row-major
        // Rotation around Y-axis
        poses.push(cos_a); // r00
        poses.push(0.0); // r01
        poses.push(sin_a); // r02
        poses.push(0.0); // tx

        poses.push(0.0); // r10
        poses.push(1.0); // r11
        poses.push(0.0); // r12
        poses.push(0.0); // ty

        poses.push(-sin_a); // r20
        poses.push(0.0); // r21
        poses.push(cos_a); // r22
        poses.push(2.5); // tz (camera distance)
    }

    Tensor::from_slice(&poses, (config.num_views, config.camera_pose_dim), device).map_err(|e| {
        oxigaf::OxigafError::Diffusion(DiffusionError::Inference(format!(
            "Failed to create camera poses: {}",
            e
        )))
    })
}

/// Save a tensor as a PNG image.
fn save_tensor_as_image(
    tensor: &candle_core::Tensor,
    path: &std::path::Path,
) -> std::result::Result<(), String> {
    use image::{Rgb, RgbImage};

    // Get tensor dimensions (expected: 3, H, W)
    let dims = tensor.dims();
    if dims.len() != 3 || dims[0] != 3 {
        return Err(format!("Expected tensor shape (3, H, W), got {:?}", dims));
    }

    let height = dims[1] as u32;
    let width = dims[2] as u32;

    // Convert to f32 vector
    let data: Vec<f32> = tensor
        .flatten_all()
        .map_err(|e| format!("Failed to flatten tensor: {}", e))?
        .to_vec1()
        .map_err(|e| format!("Failed to convert tensor to vec: {}", e))?;

    // Create RGB image
    let mut img = RgbImage::new(width, height);
    let hw = (height * width) as usize;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            // Tensor is CHW format, so R is at [0..HW], G at [HW..2HW], B at [2HW..3HW]
            let r = (data.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0) * 255.0) as u8;
            let g = (data.get(hw + idx).copied().unwrap_or(0.0).clamp(0.0, 1.0) * 255.0) as u8;
            let b = (data
                .get(2 * hw + idx)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0)
                * 255.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    img.save(path)
        .map_err(|e| format!("Failed to save image: {}", e))
}
