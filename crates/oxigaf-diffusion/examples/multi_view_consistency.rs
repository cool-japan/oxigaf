//! Multi-view consistency demonstration.
//!
//! Encodes a batch of synthetic RGB images through the sequential VAE,
//! decodes them back, and measures the cosine similarity between latent
//! pairs as well as pixel-level reconstruction error — all without real
//! model weights.
//!
//! Run with:
//! ```text
//! cargo run --example multi_view_consistency -p oxigaf-diffusion
//! ```

use oxigaf_diffusion::sequential_vae::{decode_sequential, encode_sequential, SequentialVaeConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple deterministic pseudo-random number generator (xorshift32).
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

/// Cosine similarity between two equal-length float slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Root-mean-square error between two equal-length slices.
fn rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();
    (sum_sq / a.len() as f32).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiGAF Multi-View Consistency Demo ===");
    println!();

    // ------------------------------------------------------------------
    // 1. Configure the sequential VAE for 4 views (256×256, chunk_size=2)
    // ------------------------------------------------------------------
    let num_views: usize = 4;
    let config = SequentialVaeConfig::new(
        2,       // chunk_size: process 2 views at a time
        4,       // latent_channels
        256,     // image_height
        256,     // image_width
        0.18215, // latent_scale (SD 2.1 default)
    );
    config.validate()?;

    println!("SequentialVaeConfig:");
    println!("  num_views      : {num_views}");
    println!("  chunk_size     : {}", config.chunk_size);
    println!(
        "  image size     : {}×{}",
        config.image_height, config.image_width
    );
    println!(
        "  latent size    : {}×{} × {} ch",
        config.latent_height(),
        config.latent_width(),
        config.latent_channels
    );
    println!("  latent_scale   : {}", config.latent_scale);
    println!();

    // ------------------------------------------------------------------
    // 2. Generate 4 synthetic RGB images using deterministic noise
    // ------------------------------------------------------------------
    let img_elems = config.image_element_count();
    let mut rng = Xorshift32::new(0xDEAD_BEEF);

    let images: Vec<Vec<f32>> = (0..num_views)
        .map(|_| (0..img_elems).map(|_| rng.next_f32()).collect())
        .collect();

    println!("Generated {num_views} synthetic images, each with {img_elems} elements (3×256×256).");
    println!();

    // ------------------------------------------------------------------
    // 3. Encode all views to latent space (sequential, chunk_size=2)
    // ------------------------------------------------------------------
    let encoded = encode_sequential(&images, &config)?;

    println!(
        "Encoded {} views → latent shape {}×{}×{}",
        encoded.num_views, encoded.latent_channels, encoded.latent_height, encoded.latent_width
    );
    println!(
        "Latent elements per view: {}",
        config.latent_element_count()
    );
    println!();

    // ------------------------------------------------------------------
    // 4. Decode latents back to pixel space
    // ------------------------------------------------------------------
    let decoded = decode_sequential(&encoded, &config)?;

    println!(
        "Decoded {} views → image shape {}×{}×{}",
        decoded.num_views, decoded.channels, decoded.height, decoded.width
    );
    println!();

    // ------------------------------------------------------------------
    // 5. Cosine similarity between every pair of latent vectors
    // ------------------------------------------------------------------
    println!("Cosine similarity between latent pairs:");
    println!("  {:<12} {:>12}", "Pair", "Similarity");
    println!("  {}", "-".repeat(26));

    for i in 0..num_views {
        for j in (i + 1)..num_views {
            let sim = cosine_similarity(&encoded.latents[i], &encoded.latents[j]);
            println!("  view {:1} vs {:1}  {:>12.6}", i, j, sim);
        }
    }
    println!();

    // ------------------------------------------------------------------
    // 6. Reconstruction error statistics (encode → decode round-trip)
    // ------------------------------------------------------------------
    println!("Pixel reconstruction error (encode → decode round-trip):");
    println!(
        "  {:<8} {:>10} {:>10} {:>10}",
        "View", "RMSE", "Min Δ", "Max Δ"
    );
    println!("  {}", "-".repeat(42));

    let mut total_rmse = 0.0_f32;
    for (v, (orig, recon)) in images
        .iter()
        .zip(decoded.images.iter())
        .enumerate()
        .take(num_views)
    {
        let err = rmse(orig, recon);
        total_rmse += err;

        let max_err = orig
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(f32::NEG_INFINITY, f32::max);
        let min_err = orig
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(f32::INFINITY, f32::min);

        println!(
            "  view {:1}   {:>10.6} {:>10.6} {:>10.6}",
            v, err, min_err, max_err
        );
    }
    let mean_rmse = total_rmse / num_views as f32;
    println!();
    println!("  Mean RMSE across {num_views} views: {mean_rmse:.6}");
    println!();

    // ------------------------------------------------------------------
    // 7. Memory savings from sequential processing
    // ------------------------------------------------------------------
    use oxigaf_diffusion::sequential_vae::{
        batch_memory_bytes, memory_reduction_ratio, peak_memory_bytes,
    };

    let seq_peak = peak_memory_bytes(&config, num_views);
    let batch_total = batch_memory_bytes(&config, num_views);
    let ratio = memory_reduction_ratio(&config, num_views);

    println!("Memory analysis:");
    println!(
        "  Sequential peak  : {:.2} MB",
        seq_peak as f32 / 1024.0 / 1024.0
    );
    println!(
        "  Batch total      : {:.2} MB",
        batch_total as f32 / 1024.0 / 1024.0
    );
    println!("  Reduction ratio  : {ratio:.2}×");
    println!();

    println!("=== Multi-view consistency demo complete ===");
    Ok(())
}
