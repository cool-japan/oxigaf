//! Batch generation demonstration.
//!
//! Shows how to queue multiple `GenerationRequest`s, process them with
//! `BatchGenerator`, and inspect the resulting `BatchStats`.
//!
//! Run with:
//! ```text
//! cargo run --example batch_generation -p oxigaf-diffusion
//! ```

use oxigaf_diffusion::{BatchGenConfig, BatchGenerator, BatchStats, GenerationRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiGAF Batch Generation Demo ===");
    println!();

    // -----------------------------------------------------------------
    // 1. Build a BatchGenConfig
    // -----------------------------------------------------------------
    let config = BatchGenConfig {
        max_batch_size: 8,
        max_views_per_request: 4,
        guidance_scale: 3.0,
        num_steps: 10,
        use_kv_cache: true,
        synchronous: true,
    };

    println!("BatchGenConfig:");
    println!("  max_batch_size       : {}", config.max_batch_size);
    println!("  max_views_per_request: {}", config.max_views_per_request);
    println!("  guidance_scale       : {}", config.guidance_scale);
    println!("  num_steps            : {}", config.num_steps);
    println!("  use_kv_cache         : {}", config.use_kv_cache);
    println!();

    // -----------------------------------------------------------------
    // 2. Create the batch generator
    // -----------------------------------------------------------------
    let gen = BatchGenerator::new(config);

    // -----------------------------------------------------------------
    // 3. Queue multiple GenerationRequests
    // -----------------------------------------------------------------
    let requests = vec![
        GenerationRequest {
            id: "req-001".into(),
            reference_image: vec![128u8; 256 * 256 * 3],
            image_width: 256,
            image_height: 256,
            num_views: 2,
            guidance_scale: None, // use config default
            num_steps: None,      // use config default
            seed: Some(42),
        },
        GenerationRequest {
            id: "req-002".into(),
            reference_image: vec![200u8; 256 * 256 * 3],
            image_width: 256,
            image_height: 256,
            num_views: 3,
            guidance_scale: Some(5.0), // override guidance
            num_steps: Some(20),       // override steps
            seed: Some(7),
        },
        GenerationRequest {
            id: "req-003".into(),
            reference_image: vec![64u8; 256 * 256 * 3],
            image_width: 256,
            image_height: 256,
            num_views: 1,
            guidance_scale: None,
            num_steps: None,
            seed: None,
        },
    ];

    println!("Queueing {} requests…", requests.len());
    for req in &requests {
        gen.queue(req.clone())?;
    }
    println!("Queue length: {}", gen.queue_len());
    println!();

    // -----------------------------------------------------------------
    // 4. Process the full batch
    // -----------------------------------------------------------------
    println!("Processing batch…");
    let results = gen.process_batch()?;
    println!("Queue after processing: {} (drained)", gen.queue_len());
    println!();

    for result in &results {
        println!(
            "Result id={} → {} views, time={:.2} ms",
            result.id,
            result.views.len(),
            result.total_time_ms
        );
        for view in &result.views {
            println!(
                "  view {} : {}×{} px, {} bytes",
                view.view_index,
                view.width,
                view.height,
                view.image_data.len()
            );
        }
    }
    println!();

    // -----------------------------------------------------------------
    // 5. Process a single request immediately (bypassing the queue)
    // -----------------------------------------------------------------
    let solo_req = GenerationRequest {
        id: "solo".into(),
        reference_image: vec![0u8; 64 * 64 * 3],
        image_width: 64,
        image_height: 64,
        num_views: 4,
        guidance_scale: None,
        num_steps: None,
        seed: None,
    };

    println!("Processing single request directly…");
    let solo_result = gen.process_one(solo_req)?;
    println!(
        "Solo result: {} views, all_views_generated={}",
        solo_result.views.len(),
        solo_result.all_views_generated()
    );
    println!();

    // -----------------------------------------------------------------
    // 6. Inspect cumulative BatchStats
    // -----------------------------------------------------------------
    let stats: BatchStats = gen.stats();
    print_stats(&stats);

    // -----------------------------------------------------------------
    // 7. Clear queue demonstration
    // -----------------------------------------------------------------
    gen.queue(GenerationRequest {
        id: "pending".into(),
        reference_image: vec![0u8; 64 * 64 * 3],
        image_width: 64,
        image_height: 64,
        num_views: 1,
        guidance_scale: None,
        num_steps: None,
        seed: None,
    })?;
    println!("Queued 1 pending request (queue len = {})", gen.queue_len());
    gen.clear_queue();
    println!("After clear_queue:  queue len = {}", gen.queue_len());
    println!();

    println!("=== Batch generation demo complete ===");
    Ok(())
}

/// Print a summary of cumulative generation statistics.
fn print_stats(stats: &BatchStats) {
    println!("=== BatchStats ===");
    println!("  total_requests        : {}", stats.total_requests);
    println!("  total_views_generated : {}", stats.total_views_generated);
    println!("  total_time_ms         : {:.2}", stats.total_time_ms);
    println!(
        "  avg_time_per_view_ms  : {:.4}",
        stats.average_time_per_view_ms()
    );
    println!(
        "  cache_hit_rate        : {:.2}%",
        stats.cache_hit_rate() * 100.0
    );
    println!("  cache_hits            : {}", stats.cache_hits);
    println!("  cache_misses          : {}", stats.cache_misses);
    println!();
}
