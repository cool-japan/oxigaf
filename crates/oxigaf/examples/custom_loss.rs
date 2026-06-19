//! Custom loss functions alongside the built-in trainer loss API.
//!
//! Demonstrates:
//! - Implementing a Charbonnier loss (smooth L1 approximation) from scratch
//! - Using the built-in `l1_loss`, `ssim_loss`, `scale_reg`, `opacity_reg`
//!   functions from `oxigaf::trainer::loss`
//! - Combining custom and built-in losses into a weighted objective
//!
//! All computation is CPU-only — no GPU or real assets required.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example custom_loss
//! ```

use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf::trainer::loss::{gaussian_kernel_1d, l1_loss, opacity_reg, scale_reg, ssim_loss};

// ============================================================================
// Charbonnier loss implementation
// ============================================================================

/// Charbonnier loss: a smooth L1 approximation.
///
/// For each pair (p, t) computes `sqrt((p - t)^2 + eps^2) - eps`, then
/// returns the mean over all elements.  As `eps → 0` this converges to L1;
/// at large `eps` it behaves like L2/2.  The `eps` subtraction makes the
/// minimum value zero (loss is 0 when pred == target).
///
/// # Arguments
///
/// * `pred`   - predicted values (flat, any channel layout)
/// * `target` - ground-truth values (same length as `pred`)
/// * `eps`    - smoothness constant; typical values: 1e-3 … 1e-1
///
/// # Panics
///
/// Does not panic. Returns 0.0 if `pred` is empty.
pub fn charbonnier_loss(pred: &[f32], target: &[f32], eps: f32) -> f32 {
    let n = pred.len().min(target.len());
    if n == 0 {
        return 0.0;
    }
    let eps2 = eps * eps;
    let sum: f32 = pred[..n]
        .iter()
        .zip(target[..n].iter())
        .map(|(&p, &t)| {
            let diff = p - t;
            (diff * diff + eps2).sqrt() - eps
        })
        .sum();
    sum / n as f32
}

// ============================================================================
// Synthetic image helpers
// ============================================================================

/// Generate a synthetic 4×4 RGB image in HWC flat layout (H=4, W=4, C=3).
///
/// Uses a deterministic LCG to avoid external RNG dependencies.
fn synthetic_image_4x4(seed_init: u64) -> Vec<f32> {
    let mut seed = seed_init;
    let n = 4 * 4 * 3;
    let mut buf = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [0, 1]
        let v = ((seed >> 33) as f32) / (u32::MAX as f32);
        buf.push(v);
    }
    buf
}

/// Build a small demo GaussianModel with `n` Gaussians at SH degree 0.
///
/// Mirrors the `create_demo_model` helper in `training_loop.rs`.
fn create_demo_model(n: usize) -> GaussianModel {
    let sh_degree: u32 = 0;
    let sh_per = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;

    let mut seed = 99u64;
    let mut rng = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    let mut gaussians = Vec::with_capacity(n);
    let mut sh_coeffs = Vec::with_capacity(n * sh_per);

    for _ in 0..n {
        gaussians.push(GaussianAttributes {
            position: [rng() * 0.15, rng() * 0.15, rng() * 0.15],
            _pad0: 0.0,
            rotation: [rng() * 0.05, rng() * 0.05, rng() * 0.05, 1.0],
            scale: [-4.5, -4.5, -4.5], // log-space: exp(-4.5) ≈ 0.011
            opacity: -1.5,             // sigmoid(-1.5) ≈ 0.18
        });
        for _ in 0..sh_per {
            sh_coeffs.push(rng() * 0.3);
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

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("OxiGAF Custom Loss Example");
    println!("==========================");
    println!();

    // =========================================================================
    // Step 1: Build synthetic 4×4 rendered prediction and ground-truth images
    // =========================================================================
    //
    // In production these would come from the GPU rasterizer (prediction) and
    // the dataset (target).  Here we use deterministic synthetic values so the
    // example runs on any machine without assets or a GPU.

    println!("Step 1: Creating synthetic 4×4 RGB images (HWC layout)...");

    let width: usize = 4;
    let height: usize = 4;
    let pred = synthetic_image_4x4(1234);
    let target = synthetic_image_4x4(5678);

    println!("  pred[0..6]   = {:?}", &pred[..6.min(pred.len())]);
    println!("  target[0..6] = {:?}", &target[..6.min(target.len())]);

    // =========================================================================
    // Step 2: Build a small GaussianModel for regularisation terms
    // =========================================================================

    println!();
    println!("Step 2: Building demo GaussianModel (16 Gaussians)...");

    let model = create_demo_model(16);
    println!(
        "  Gaussians: {}, SH degree: {}",
        model.len(),
        model.sh_degree
    );

    // =========================================================================
    // Step 3: Charbonnier loss (custom implementation)
    // =========================================================================
    //
    // eps = 1e-3 is a standard choice that keeps the loss nearly identical to
    // L1 while guaranteeing a continuous gradient everywhere (no kink at zero).

    println!();
    println!("Step 3: Computing Charbonnier loss...");

    let eps = 1e-3_f32;
    let charb = charbonnier_loss(&pred, &target, eps);
    println!("  charbonnier_loss(eps={:.0e}) = {:.6}", eps, charb);

    // Show the relationship to L1 at the same eps: should be close but slightly
    // smaller because sqrt(x^2+eps^2)-eps < |x| for all x != 0.
    let l1 = l1_loss(&pred, &target);
    println!("  l1_loss (built-in)          = {:.6}", l1);
    println!(
        "  Δ (L1 − Charb)             = {:.6}  (Charb ≤ L1 always)",
        l1 - charb
    );

    // =========================================================================
    // Step 4: SSIM loss (built-in, from oxigaf::trainer::loss)
    // =========================================================================
    //
    // ssim_loss requires a pre-computed 1-D Gaussian kernel (separable filter).
    // gaussian_kernel_1d(11, 1.5) matches the SSIM paper defaults.

    println!();
    println!("Step 4: Computing SSIM loss...");

    let kernel = gaussian_kernel_1d(11, 1.5);
    let ssim = ssim_loss(&pred, &target, width, height, &kernel);
    println!("  ssim_loss (1 − SSIM) = {:.6}", ssim);
    println!("  (0.0 = identical images, 2.0 = maximally different)");

    // =========================================================================
    // Step 5: Scale regularisation (built-in)
    // =========================================================================
    //
    // scale_reg computes mean squared log-scale, penalising Gaussians that are
    // either extremely large or extremely small in any axis.

    println!();
    println!("Step 5: Computing scale regularisation...");

    let sc_reg = scale_reg(&model);
    println!("  scale_reg  = {:.6}", sc_reg);

    // =========================================================================
    // Step 6: Opacity regularisation (built-in)
    // =========================================================================
    //
    // opacity_reg uses binary entropy −(σ log σ + (1−σ) log(1−σ)) to encourage
    // Gaussians to be either fully opaque or fully transparent.

    println!();
    println!("Step 6: Computing opacity regularisation...");

    let op_reg = opacity_reg(&model);
    println!("  opacity_reg = {:.6}", op_reg);

    // =========================================================================
    // Step 7: Weighted combination
    // =========================================================================
    //
    // A practical training objective blends the reconstruction loss with
    // regularisation terms.  The weights below are typical starting values:
    //   w_ssim     = 0.20 (SSIM weighted lower for small images)
    //   w_scale    = 0.01 (light scale penalty)
    //   w_opacity  = 0.01 (light opacity penalty)

    println!();
    println!("Step 7: Combining into weighted objective...");

    let w_ssim = 0.20_f32;
    let w_scale = 0.01_f32;
    let w_opacity = 0.01_f32;

    let total_loss = charb + w_ssim * ssim + w_scale * sc_reg + w_opacity * op_reg;

    println!();
    println!("  Loss breakdown:");
    println!("  ┌──────────────────────────────────────────────────────┐");
    println!(
        "  │  Charbonnier (1.00)   {:.6}                        │",
        charb
    );
    println!(
        "  │  SSIM        ({:.2}) × {:.6} = {:.6}            │",
        w_ssim,
        ssim,
        w_ssim * ssim
    );
    println!(
        "  │  ScaleReg    ({:.2}) × {:.6} = {:.6}            │",
        w_scale,
        sc_reg,
        w_scale * sc_reg
    );
    println!(
        "  │  OpacityReg  ({:.2}) × {:.6} = {:.6}            │",
        w_opacity,
        op_reg,
        w_opacity * op_reg
    );
    println!("  ├──────────────────────────────────────────────────────┤");
    println!(
        "  │  TOTAL LOSS           {:.6}                        │",
        total_loss
    );
    println!("  └──────────────────────────────────────────────────────┘");

    // =========================================================================
    // Key Takeaways footer
    // =========================================================================

    println!();
    println!("Key Takeaways:");
    println!(
        "  - charbonnier_loss(pred, target, eps) is a drop-in L1 replacement with smooth grad"
    );
    println!("  - oxigaf::trainer::loss::l1_loss(pred, target) gives plain mean-absolute-error");
    println!("  - ssim_loss(pred, target, w, h, &kernel) returns 1−SSIM (lower = better match)");
    println!("  - Use gaussian_kernel_1d(11, 1.5) for the standard SSIM 11×11 Gaussian kernel");
    println!("  - scale_reg(&model) / opacity_reg(&model) add structural priors to the objective");
    println!("  - Typical weights: L1/Charb=1.0, SSIM=0.2, scale=0.01, opacity=0.01");
}
