//! CFG guidance-scale comparison.
//!
//! Creates `DiffusionConfig` variants at guidance_scale = 1.0, 3.5 and 7.5,
//! encodes a batch of synthetic images through the sequential VAE for each
//! variant, and prints latent statistics scaled by the guidance value.
//!
//! Demonstrates how CFG affects the effective inference budget and how
//! guidance-scaled latent statistics differ between configurations.
//!
//! Run with:
//! ```text
//! cargo run --example cfg_comparison -p oxigaf-diffusion
//! ```

use oxigaf_diffusion::{
    sequential_vae::{encode_sequential, SequentialVaeConfig},
    DiffusionConfig,
};

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

fn std_dev(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let variance = xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / xs.len() as f32;
    variance.sqrt()
}

fn min_val(xs: &[f32]) -> f32 {
    xs.iter().cloned().fold(f32::INFINITY, f32::min)
}

fn max_val(xs: &[f32]) -> f32 {
    xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

// ---------------------------------------------------------------------------
// Pseudo-random data generation (xorshift32)
// ---------------------------------------------------------------------------

struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        (self.state as f32) / (u32::MAX as f32)
    }
}

// ---------------------------------------------------------------------------
// CFG explanation helper
// ---------------------------------------------------------------------------

/// Estimate how CFG guidance scale affects total inference cost.
///
/// When `guidance_scale > 1.0` the pipeline performs two U-Net forward passes
/// (conditional + unconditional) per denoising step, doubling cost.  At
/// `guidance_scale == 1.0` only the conditional pass is required.
fn inference_budget_factor(guidance_scale: f64) -> f64 {
    if (guidance_scale - 1.0).abs() < 1e-9 {
        1.0 // single forward pass
    } else {
        2.0 // conditional + unconditional pass
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiGAF CFG Guidance-Scale Comparison ===");
    println!();
    println!("CFG formula:  ε_cfg = ε_uncond + s × (ε_cond − ε_uncond)");
    println!("  s = 1.0 → no guidance (pure conditional, single forward pass)");
    println!("  s = 3.5 → moderate conditioning (2 forward passes per step)");
    println!("  s = 7.5 → strong conditioning  (2 forward passes per step)");
    println!();

    // ------------------------------------------------------------------
    // 1. Shared VAE + image configuration
    // ------------------------------------------------------------------
    let num_views: usize = 4;

    // Use the same VAE configuration for all variants
    let vae_cfg = SequentialVaeConfig::new(
        2,       // chunk_size
        4,       // latent_channels
        256,     // image_height
        256,     // image_width
        0.18215, // latent_scale
    );
    vae_cfg.validate()?;

    // Generate the same synthetic images for all guidance-scale variants
    let img_elems = vae_cfg.image_element_count();
    let mut rng = Xorshift32::new(0xCAFE_BABE);
    let images: Vec<Vec<f32>> = (0..num_views)
        .map(|_| (0..img_elems).map(|_| rng.next_f32()).collect())
        .collect();

    // ------------------------------------------------------------------
    // 2. Iterate over guidance scales
    // ------------------------------------------------------------------
    let guidance_scales = [1.0_f64, 3.5, 7.5];

    println!(
        "{:<6} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Scale", "Steps", "Budget×", "LatMean", "LatStd", "LatMin", "LatMax", "ScaledMean"
    );
    println!("{}", "-".repeat(80));

    for &guidance_scale in &guidance_scales {
        // Build a config variant for this guidance scale
        let cfg = DiffusionConfig {
            guidance_scale,
            num_views,
            ..Default::default()
        };

        // Encode images (same data, same VAE — latents are guidance-scale independent)
        let encoded = encode_sequential(&images, &vae_cfg)?;

        // Gather all latent values into a flat slice for statistics
        let all_latents: Vec<f32> = encoded.latents.iter().flatten().copied().collect();

        let lat_mean = mean(&all_latents);
        let lat_std = std_dev(&all_latents);
        let lat_min = min_val(&all_latents);
        let lat_max = max_val(&all_latents);

        // Guidance-scaled mean: show how the CFG formula scales predictions
        let scaled_mean = lat_mean * guidance_scale as f32;

        let budget_factor = inference_budget_factor(guidance_scale);

        println!(
            "{:<6.1} {:>8} {:>10.2} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>12.6}",
            guidance_scale,
            cfg.num_inference_steps,
            budget_factor,
            lat_mean,
            lat_std,
            lat_min,
            lat_max,
            scaled_mean,
        );
    }
    println!();

    // ------------------------------------------------------------------
    // 3. Detailed breakdown for the recommended scale (3.5)
    // ------------------------------------------------------------------
    println!("--- Detailed breakdown at guidance_scale = 3.5 ---");
    let cfg35 = DiffusionConfig {
        guidance_scale: 3.5,
        ..Default::default()
    };

    let encoded35 = encode_sequential(&images, &vae_cfg)?;

    println!(
        "Config: num_views={} guidance={} steps={} sequential_vae={}",
        cfg35.num_views, cfg35.guidance_scale, cfg35.num_inference_steps, cfg35.sequential_vae
    );
    println!();

    println!("Per-view latent statistics:");
    println!(
        "  {:<6} {:>10} {:>10} {:>10} {:>10}",
        "View", "Mean", "Std", "Min", "Max"
    );
    println!("  {}", "-".repeat(50));

    for (v, latent) in encoded35.latents.iter().enumerate() {
        let m = mean(latent);
        let s = std_dev(latent);
        let mn = min_val(latent);
        let mx = max_val(latent);
        println!(
            "  view {:1}  {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
            v, m, s, mn, mx
        );
    }
    println!();

    // ------------------------------------------------------------------
    // 4. Inference cost breakdown
    // ------------------------------------------------------------------
    println!("Inference cost breakdown:");
    println!(
        "  {:<6} {:>14} {:>18} {:>16}",
        "Scale", "Steps", "UNet Fwd-Passes", "Relative Cost"
    );
    println!("  {}", "-".repeat(58));

    for &gs in &guidance_scales {
        let steps = DiffusionConfig {
            guidance_scale: gs,
            ..DiffusionConfig::default()
        }
        .num_inference_steps;
        let passes_per_step = if (gs - 1.0).abs() < 1e-9 { 1 } else { 2 };
        let total_passes = steps * passes_per_step;
        let baseline = DiffusionConfig::default().num_inference_steps; // gs=1.0 baseline
        let relative = total_passes as f64 / baseline as f64;
        println!(
            "  {:<6.1} {:>14} {:>18} {:>16.2}×",
            gs, steps, total_passes, relative
        );
    }
    println!();

    println!("=== CFG comparison complete ===");
    Ok(())
}
