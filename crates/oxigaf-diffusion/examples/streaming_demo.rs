//! Streaming inference demonstration.
//!
//! Shows how to use `StreamingInference` to generate views one denoising step
//! at a time.  This is useful for applications that want to display progressive
//! partial results as each step completes rather than waiting for the full run.
//!
//! Run with:
//! ```text
//! cargo run --example streaming_demo -p oxigaf-diffusion
//! ```

use oxigaf_diffusion::{StreamingConfig, StreamingInference};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiGAF Streaming Inference Demo ===");
    println!();

    // -----------------------------------------------------------------
    // 1. Build a StreamingConfig
    // -----------------------------------------------------------------
    let config = StreamingConfig {
        image_width: 256,
        image_height: 256,
        guidance_scale: 3.0,
        num_steps: 5, // Use a small number so output is readable
    };

    println!("StreamingConfig:");
    println!("  image_width    : {}", config.image_width);
    println!("  image_height   : {}", config.image_height);
    println!("  guidance_scale : {}", config.guidance_scale);
    println!("  num_steps      : {}", config.num_steps);
    println!();

    // -----------------------------------------------------------------
    // 2. Create the streaming inference engine
    // -----------------------------------------------------------------
    let si = StreamingInference::new(config);

    // -----------------------------------------------------------------
    // 3. Iterate over 2 views, printing progress at each step
    // -----------------------------------------------------------------
    let num_views = 2_usize;
    println!("Generating {num_views} views…");
    println!();

    let mut last_view = usize::MAX;
    for step in si.step_iter(num_views) {
        // Print a header each time we move to a new view.
        if step.view_index != last_view {
            println!("--- View {} ---", step.view_index);
            last_view = step.view_index;
        }

        let pct = (step.progress_fraction() * 100.0) as u32;
        let sample_pixel = step.partial_image.first().copied().unwrap_or(0);

        println!(
            "  step {:2}/{} | progress {:3}% | first_pixel={} | is_final={}",
            step.step_index + 1,
            step.total_steps,
            pct,
            sample_pixel,
            step.is_final,
        );

        if step.is_final {
            let img_len = step.partial_image.len();
            println!(
                "  View {} complete — image buffer {} bytes",
                step.view_index, img_len
            );
            println!();
        }
    }

    println!("=== Streaming demo complete ===");
    Ok(())
}
