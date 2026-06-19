//! Flash attention vs standard attention benchmark.
//!
//! Compares two implementations of scaled dot-product attention:
//!
//! - **Standard** — O(N²) naive computation: build the full N×N score matrix,
//!   apply softmax row-by-row, then multiply by V.
//! - **Flash** — tiled computation with online log-sum-exp (O(N) memory):
//!   accumulates the weighted sum of V incrementally, block by block,
//!   maintaining running (max, denominator) for numerically stable softmax.
//!
//! Both implementations operate on `Vec<f32>` matrices and produce identical
//! outputs (verified to within 1e-3 tolerance).
//!
//! Run with:
//! ```text
//! cargo run --example flash_vs_standard -p oxigaf-diffusion --release
//! ```

use std::time::Instant;

use oxigaf_diffusion::DiffusionConfig;

// ---------------------------------------------------------------------------
// Pseudo-random number generator (xorshift32)
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
        // Map to [-1, 1) for realistic attention inputs
        (self.state as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Generate a flat row-major matrix of shape `[rows × cols]`.
fn rand_matrix(rows: usize, cols: usize, rng: &mut Xorshift32) -> Vec<f32> {
    (0..rows * cols).map(|_| rng.next_f32()).collect()
}

// ---------------------------------------------------------------------------
// Softmax (in-place, numerically stable)
// ---------------------------------------------------------------------------

fn softmax_inplace(xs: &mut [f32]) {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in xs.iter_mut() {
            *x /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Standard O(N²) attention
// ---------------------------------------------------------------------------
//
// Computes:  Output[i] = sum_j ( softmax(Q[i]·K[j] / sqrt(D)) ) * V[j]
//
// Arguments:
//   q, k, v  – flat row-major [seq_len × dim] matrices
//   seq_len  – number of tokens (N)
//   dim      – head dimension (D)
//
// Returns a flat [seq_len × dim] output matrix.

fn standard_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    let scale = 1.0_f32 / (dim as f32).sqrt();
    let mut output = vec![0.0_f32; seq_len * dim];
    let mut scores = vec![0.0_f32; seq_len]; // reused row buffer

    for i in 0..seq_len {
        let qi = &q[i * dim..(i + 1) * dim];

        // Compute dot products Q[i] · K[j] for all j
        for j in 0..seq_len {
            let kj = &k[j * dim..(j + 1) * dim];
            let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
            scores[j] = dot * scale;
        }

        // Stable softmax
        softmax_inplace(&mut scores[..seq_len]);

        // Weighted sum of V
        let out_i = &mut output[i * dim..(i + 1) * dim];
        for j in 0..seq_len {
            let vj = &v[j * dim..(j + 1) * dim];
            let w = scores[j];
            for d in 0..dim {
                out_i[d] += w * vj[d];
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Flash attention (tiled, O(N) memory)
// ---------------------------------------------------------------------------
//
// Implements the core Flash Attention accumulation loop:
//
//   For each query i:
//     m = -∞,  l = 0,  O = 0
//     For each key block b:
//       s[j] = Q[i]·K[j] / sqrt(D)   for j in block
//       m_new = max(m, max(s))
//       α     = exp(m - m_new)
//       w[j]  = exp(s[j] - m_new)
//       O     = α·O + Σ_j w[j]·V[j]
//       l     = α·l + Σ_j w[j]
//       m     = m_new
//     output[i] = O / l

fn flash_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    dim: usize,
    block_size: usize,
) -> Vec<f32> {
    let scale = 1.0_f32 / (dim as f32).sqrt();
    let mut output = vec![0.0_f32; seq_len * dim];

    // Buffer for local block scores (at most block_size elements)
    let mut local_scores = vec![0.0_f32; block_size];

    for i in 0..seq_len {
        let qi = &q[i * dim..(i + 1) * dim];

        // Online softmax running state
        let mut running_max = f32::NEG_INFINITY;
        let mut running_l = 0.0_f32;
        let mut running_o = vec![0.0_f32; dim];

        let mut j_start = 0;
        while j_start < seq_len {
            let j_end = (j_start + block_size).min(seq_len);
            let block_len = j_end - j_start;

            // Compute scaled dot products for this block
            let mut block_max = f32::NEG_INFINITY;
            for (b, j) in (j_start..j_end).enumerate() {
                let kj = &k[j * dim..(j + 1) * dim];
                let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                let s = dot * scale;
                local_scores[b] = s;
                if s > block_max {
                    block_max = s;
                }
            }

            // Update global running max
            let new_max = running_max.max(block_max);

            // Correction factor for previously accumulated state
            let alpha = (running_max - new_max).exp();

            // Compute block weights exp(s - new_max) and their sum
            let mut sum_w = 0.0_f32;
            for score in local_scores.iter_mut().take(block_len) {
                *score = (*score - new_max).exp();
                sum_w += *score;
            }

            // Rescale running output and accumulate new block contribution
            for o in running_o.iter_mut().take(dim) {
                *o *= alpha;
            }
            for (b, j) in (j_start..j_end).enumerate() {
                let w = local_scores[b];
                let vj = &v[j * dim..(j + 1) * dim];
                for d in 0..dim {
                    running_o[d] += w * vj[d];
                }
            }

            running_l = alpha * running_l + sum_w;
            running_max = new_max;

            j_start = j_end;
        }

        // Normalize
        let out_i = &mut output[i * dim..(i + 1) * dim];
        if running_l > 0.0 {
            for d in 0..dim {
                out_i[d] = running_o[d] / running_l;
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Verification helper
// ---------------------------------------------------------------------------

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ---------------------------------------------------------------------------
// Timing helper — runs a closure `reps` times and returns mean wall-clock ns.
// ---------------------------------------------------------------------------

fn time_ns<F: Fn() -> Vec<f32>>(reps: usize, f: F) -> (u64, Vec<f32>) {
    let mut last_result = Vec::new();
    let t0 = Instant::now();
    for _ in 0..reps {
        last_result = f();
    }
    let elapsed_ns = t0.elapsed().as_nanos() as u64;
    let mean_ns = elapsed_ns / reps as u64;
    (mean_ns, last_result)
}

fn main() {
    println!("=== OxiGAF Flash vs Standard Attention ===");
    println!();

    // ------------------------------------------------------------------
    // 1. Show config-level flag for flash attention
    // ------------------------------------------------------------------
    #[cfg(feature = "flash_attention")]
    let flash_cfg = DiffusionConfig {
        use_flash_attention: true,
        ..DiffusionConfig::default()
    };
    #[cfg(not(feature = "flash_attention"))]
    let flash_cfg = DiffusionConfig::default();

    let std_cfg = DiffusionConfig {
        use_flash_attention: false,
        ..DiffusionConfig::default()
    };

    println!("DiffusionConfig.use_flash_attention:");
    println!("  flash config   : {}", flash_cfg.use_flash_attention);
    println!("  standard config: {}", std_cfg.use_flash_attention);
    println!(
        "  flash block_size: {}",
        flash_cfg.flash_attention_block_size
    );
    println!();

    // ------------------------------------------------------------------
    // 2. Synthetic Q / K / V matrices
    // ------------------------------------------------------------------
    let seq_len: usize = 128;
    let dim: usize = 64;
    let block_size: usize = flash_cfg.flash_attention_block_size;
    let reps: usize = 10;

    let mut rng = Xorshift32::new(0xFACE_B00C);
    let q = rand_matrix(seq_len, dim, &mut rng);
    let k = rand_matrix(seq_len, dim, &mut rng);
    let v = rand_matrix(seq_len, dim, &mut rng);

    println!("Attention parameters:");
    println!("  seq_len        : {seq_len}");
    println!("  dim            : {dim}");
    println!("  block_size     : {block_size}");
    println!("  repetitions    : {reps}");
    println!();

    // ------------------------------------------------------------------
    // 3. Time standard attention
    // ------------------------------------------------------------------
    let (std_ns, std_out) = time_ns(reps, || standard_attention(&q, &k, &v, seq_len, dim));

    // ------------------------------------------------------------------
    // 4. Time flash attention
    // ------------------------------------------------------------------
    let (flash_ns, flash_out) = time_ns(reps, || {
        flash_attention(&q, &k, &v, seq_len, dim, block_size)
    });

    // ------------------------------------------------------------------
    // 5. Verify correctness
    // ------------------------------------------------------------------
    let tolerance = 1e-3_f32;
    let max_diff = max_abs_diff(&std_out, &flash_out);
    let within_tolerance = max_diff < tolerance;

    println!("Correctness check:");
    println!("  max |std − flash| : {max_diff:.2e}");
    println!("  tolerance         : {tolerance:.2e}");
    println!("  within tolerance  : {within_tolerance}");
    println!();

    assert!(
        within_tolerance,
        "Flash and standard attention diverged: max diff {max_diff:.4e} > {tolerance:.4e}"
    );

    // ------------------------------------------------------------------
    // 6. Timing results
    // ------------------------------------------------------------------
    println!("Timing (mean over {reps} runs):");
    println!(
        "  standard attention : {std_ns:>10} ns  ({:.3} ms)",
        std_ns as f64 / 1e6
    );
    println!(
        "  flash attention    : {flash_ns:>10} ns  ({:.3} ms)",
        flash_ns as f64 / 1e6
    );

    let speedup = std_ns as f64 / flash_ns as f64;
    println!("  speedup (flash/std): {speedup:.3}×");
    println!();

    // ------------------------------------------------------------------
    // 7. Output statistics
    // ------------------------------------------------------------------
    let out_mean: f32 = std_out.iter().sum::<f32>() / std_out.len() as f32;
    let out_std: f32 = {
        let var: f32 = std_out
            .iter()
            .map(|x| (x - out_mean) * (x - out_mean))
            .sum::<f32>()
            / std_out.len() as f32;
        var.sqrt()
    };
    let out_min = std_out.iter().cloned().fold(f32::INFINITY, f32::min);
    let out_max = std_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("Output statistics (standard path):");
    println!("  mean  : {out_mean:.6}");
    println!("  std   : {out_std:.6}");
    println!("  min   : {out_min:.6}");
    println!("  max   : {out_max:.6}");
    println!();

    println!("=== Flash vs standard attention demo complete ===");
}
