//! Basic diffusion inference demonstration.
//!
//! Shows how to create a `DiffusionConfig`, enable sequential VAE and weight
//! offloading, compute an `OffloadSchedule`, and run a DDIM denoising step —
//! all with synthetic (randomly-initialised) latent tensors and no real weights.
//!
//! Run with:
//! ```text
//! cargo run --example basic_inference -p oxigaf-diffusion
//! ```

use candle_core::{DType, Device, Tensor};
use oxigaf_diffusion::{
    sequential_vae::SequentialVaeConfig,
    weight_offload::{recommend_strategy, MemoryBudget, OffloadSchedule, OffloadStrategy},
    DdimScheduler, DiffusionConfig, PredictionType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -----------------------------------------------------------------
    // 1. Build a DiffusionConfig (for displaying the pipeline parameters)
    // -----------------------------------------------------------------
    let cfg = DiffusionConfig::default();
    println!("=== OxiGAF Basic Inference Demo ===");
    println!("Pipeline config:");
    println!("  num_views         : {}", cfg.num_views);
    println!("  guidance_scale    : {}", cfg.guidance_scale);
    println!("  num_inference_steps: {}", cfg.num_inference_steps);
    println!("  image_size        : {}", cfg.image_size);
    println!("  latent_size       : {}", cfg.latent_size);
    println!("  latent_channels   : {}", cfg.latent_channels);
    println!("  vae_scale_factor  : {}", cfg.vae_scale_factor);
    println!();

    // -----------------------------------------------------------------
    // 2. Create a DDIM scheduler with 1000 training timesteps
    // -----------------------------------------------------------------
    let num_train_timesteps = 1000_usize;
    let mut scheduler = DdimScheduler::new(num_train_timesteps, PredictionType::VPrediction);

    // -----------------------------------------------------------------
    // 3. Set inference timesteps (20-step DDIM)
    // -----------------------------------------------------------------
    let num_inference_steps = 20_usize;
    scheduler.set_timesteps(num_inference_steps)?;

    let timesteps = scheduler.timesteps();
    println!(
        "DDIM scheduler: {} training steps, {} inference steps",
        num_train_timesteps, num_inference_steps
    );
    println!(
        "Timestep sequence (first 5): {:?}",
        &timesteps[..5.min(timesteps.len())]
    );
    println!();

    // -----------------------------------------------------------------
    // 4. Create synthetic latent tensors (CPU device, batch=1)
    // -----------------------------------------------------------------
    let device = Device::Cpu;
    let latent_channels = cfg.latent_channels;
    let latent_size = cfg.latent_size; // 32

    // Latent shape: [batch=1, C, H, W]
    let shape = (1_usize, latent_channels, latent_size, latent_size);
    let total_elems = shape.0 * shape.1 * shape.2 * shape.3;

    // Build a ramp of values [0, 1) as stand-in for random noise.
    let data: Vec<f32> = (0..total_elems)
        .map(|i| (i as f32) / (total_elems as f32))
        .collect();

    let latent = Tensor::from_vec(data.clone(), shape, &device)?;
    let noise = Tensor::from_vec(
        data.iter().rev().copied().collect::<Vec<f32>>(),
        shape,
        &device,
    )?;

    println!("Latent tensor shape : {:?}", latent.shape());
    println!("Latent dtype        : {:?}", latent.dtype());
    println!();

    // -----------------------------------------------------------------
    // 5. Demonstrate add_noise (forward diffusion at t=500)
    // -----------------------------------------------------------------
    let t_noise = 500_usize;
    let noisy_latent = scheduler.add_noise(&latent, &noise, t_noise)?;
    let noisy_mean = noisy_latent.mean_all()?.to_scalar::<f32>()?;
    println!("add_noise at t={t_noise}:");
    println!("  noisy_latent mean  : {noisy_mean:.6}");
    println!();

    // -----------------------------------------------------------------
    // 6. Run a single DDIM step with synthetic model output
    // -----------------------------------------------------------------
    // Use the noisy latent itself as a placeholder model prediction.
    let t_step = timesteps[0];
    let model_output = noisy_latent.clone();
    let denoised = scheduler.step(&model_output, t_step, &noisy_latent)?;
    let denoised_mean = denoised.mean_all()?.to_scalar::<f32>()?;

    println!("DDIM step at t={t_step}:");
    println!("  denoised mean      : {denoised_mean:.6}");
    println!();

    // -----------------------------------------------------------------
    // 7. Generate timestep embedding tensor for batch inference
    // -----------------------------------------------------------------
    let batch_size = 4_usize;
    let ts_tensor = scheduler.timestep_tensor(t_step, batch_size, &device)?;
    println!("Timestep tensor (batch={batch_size}):");
    println!("  shape              : {:?}", ts_tensor.shape());
    println!("  dtype              : {:?}", ts_tensor.dtype());
    assert_eq!(ts_tensor.dtype(), DType::F32);
    println!();

    // -----------------------------------------------------------------
    // 8. SequentialVaeConfig — enable chunk-by-chunk VAE processing
    // -----------------------------------------------------------------
    let cfg_seq = DiffusionConfig {
        sequential_vae: true,
        vae_chunk_size: 2,
        ..Default::default()
    };

    let vae_cfg = SequentialVaeConfig::new(
        cfg_seq.vae_chunk_size, // chunk_size
        cfg_seq.latent_channels,
        cfg_seq.image_size,
        cfg_seq.image_size,
        cfg_seq.vae_scale_factor as f32,
    );
    vae_cfg.validate()?;

    println!("SequentialVaeConfig:");
    println!("  chunk_size    : {}", vae_cfg.chunk_size);
    println!("  latent_channels: {}", vae_cfg.latent_channels);
    println!(
        "  image {}×{}  → latent {}×{}",
        vae_cfg.image_height,
        vae_cfg.image_width,
        vae_cfg.latent_height(),
        vae_cfg.latent_width()
    );
    println!("  latent_scale  : {}", vae_cfg.latent_scale);
    println!("  latent_elements: {}", vae_cfg.latent_element_count());
    println!();

    // -----------------------------------------------------------------
    // 9. OffloadSchedule::for_strategy() — compute the inference phase plan
    // -----------------------------------------------------------------
    for strategy in [
        OffloadStrategy::AllInMemory,
        OffloadStrategy::Sequential,
        OffloadStrategy::CacheOne,
    ] {
        let schedule = OffloadSchedule::for_strategy(strategy);
        println!("{}", schedule.format_schedule());
    }

    // -----------------------------------------------------------------
    // 10. MemoryBudget — recommend a strategy for given VRAM
    // -----------------------------------------------------------------
    for vram_gb in [4.0_f32, 8.0, 16.0, 24.0] {
        let budget = MemoryBudget::new(vram_gb * 1024.0, 0.1);
        let suggested = recommend_strategy(&budget);
        println!(
            "VRAM {:.0} GB  →  recommended strategy: {:?}",
            vram_gb, suggested
        );
    }
    println!();

    println!("=== Demo complete ===");
    Ok(())
}
