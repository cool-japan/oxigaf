//! Rendering quality metrics for 3D Gaussian Splatting evaluation.
//!
//! Provides PSNR, SSIM, and MS-SSIM metrics that operate on raw pixel buffers
//! (f32 RGBA, row-major). All metrics exclude the alpha channel.

use crate::RenderError;

// ─── Gaussian kernel ────────────────────────────────────────────────────────

/// Build a normalized 11×11 Gaussian kernel with the given `sigma`.
///
/// Values are laid out row-major; the kernel sums to exactly 1.0.
fn gaussian_kernel_11x11(sigma: f32) -> Vec<f32> {
    const K: usize = 11;
    const HALF: i32 = 5; // (K - 1) / 2

    let sigma_sq = (sigma * sigma) as f64;
    let mut kernel = Vec::with_capacity(K * K);
    let mut sum = 0.0f64;

    for ky in 0..K {
        for kx in 0..K {
            let dy = (ky as i32 - HALF) as f64;
            let dx = (kx as i32 - HALF) as f64;
            let val = (-(dx * dx + dy * dy) / (2.0 * sigma_sq)).exp();
            kernel.push(val);
            sum += val;
        }
    }

    // Normalise so weights sum to 1.
    for v in &mut kernel {
        *v /= sum;
    }

    kernel.into_iter().map(|v| v as f32).collect()
}

// ─── Downsample ─────────────────────────────────────────────────────────────

/// Downsample an RGBA image by 2× using a 2×2 box filter.
///
/// Returns the downsampled image together with its new `(width, height)`.
fn downsample_2x(image: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let new_w = width / 2;
    let new_h = height / 2;
    let mut out = vec![0.0f32; new_w * new_h * 4];

    for ny in 0..new_h {
        for nx in 0..new_w {
            let x = nx * 2;
            let y = ny * 2;
            for c in 0..4 {
                // Average the four source pixels (handle out-of-bounds by clamping).
                let fetch = |iy: usize, ix: usize| -> f32 {
                    let iy_c = iy.min(height - 1);
                    let ix_c = ix.min(width - 1);
                    image[(iy_c * width + ix_c) * 4 + c]
                };
                let val =
                    (fetch(y, x) + fetch(y, x + 1) + fetch(y + 1, x) + fetch(y + 1, x + 1)) * 0.25;
                out[(ny * new_w + nx) * 4 + c] = val;
            }
        }
    }

    (out, new_w, new_h)
}

// ─── PSNR ───────────────────────────────────────────────────────────────────

/// Compute Peak Signal-to-Noise Ratio (PSNR) in dB.
///
/// Both slices must be row-major RGBA f32 of the same length.
/// Only RGB channels are used (alpha at stride position 3 is skipped).
///
/// * `max_val`: maximum possible pixel value (1.0 for normalised, 255.0 for u8).
///
/// Returns `f32::INFINITY` when the images are identical.
pub fn compute_psnr(
    predicted: &[f32],
    reference: &[f32],
    max_val: f32,
) -> Result<f32, RenderError> {
    if predicted.is_empty() || reference.is_empty() {
        return Err(RenderError::MismatchedBufferSizes {
            expected: 0,
            actual: 0,
        });
    }
    if predicted.len() != reference.len() {
        return Err(RenderError::MismatchedBufferSizes {
            expected: reference.len(),
            actual: predicted.len(),
        });
    }

    // Accumulate squared error over RGB channels only.
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;

    for (i, (&p, &r)) in predicted.iter().zip(reference.iter()).enumerate() {
        if i % 4 == 3 {
            // Skip alpha
            continue;
        }
        let diff = (p as f64) - (r as f64);
        sum_sq += diff * diff;
        count += 1;
    }

    if count == 0 {
        return Err(RenderError::MismatchedBufferSizes {
            expected: 1,
            actual: 0,
        });
    }

    let mse = sum_sq / (count as f64);
    if mse == 0.0 {
        return Ok(f32::INFINITY);
    }

    let max_sq = (max_val as f64) * (max_val as f64);
    Ok((10.0 * (max_sq / mse).log10()) as f32)
}

// ─── SSIM (single-scale) ────────────────────────────────────────────────────

/// Compute Structural Similarity Index (SSIM) between two RGBA images.
///
/// Uses an 11×11 Gaussian window with σ = 1.5, averaged over R, G, B channels.
///
/// Returns a value in \[−1, 1\]; 1.0 means identical.
pub fn compute_ssim(
    predicted: &[f32],
    reference: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, RenderError> {
    if predicted.len() != reference.len() {
        return Err(RenderError::MismatchedBufferSizes {
            expected: reference.len(),
            actual: predicted.len(),
        });
    }
    if predicted.len() != width * height * 4 {
        return Err(RenderError::MismatchedBufferSizes {
            expected: width * height * 4,
            actual: predicted.len(),
        });
    }

    let kernel = gaussian_kernel_11x11(1.5);
    let ssim_total = ssim_over_channels(predicted, reference, width, height, &kernel);
    Ok(ssim_total as f32)
}

/// Compute mean SSIM averaged across R, G, B channels.
fn ssim_over_channels(
    pred: &[f32],
    refer: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) -> f64 {
    // SSIM stability constants (L = 1.0)
    const L: f64 = 1.0;
    const C1: f64 = (0.01 * L) * (0.01 * L); // 1e-4
    const C2: f64 = (0.03 * L) * (0.03 * L); // 9e-4

    const K: usize = 11;
    const HALF: i32 = 5;

    let mut channel_ssims = [0.0f64; 3];

    for ch in 0..3usize {
        let mut ssim_sum = 0.0f64;
        let mut pixel_count = 0u64;

        for cy in 0..height {
            for cx in 0..width {
                let mut mu_x = 0.0f64;
                let mut mu_y = 0.0f64;
                let mut sigma_x2 = 0.0f64;
                let mut sigma_y2 = 0.0f64;
                let mut sigma_xy = 0.0f64;

                // Convolve 11×11 window (zero-pad outside boundaries).
                for ky in 0..K {
                    let py = cy as i32 + (ky as i32 - HALF);
                    if py < 0 || py >= height as i32 {
                        continue;
                    }
                    let py = py as usize;
                    for kx in 0..K {
                        let px = cx as i32 + (kx as i32 - HALF);
                        if px < 0 || px >= width as i32 {
                            continue;
                        }
                        let px = px as usize;
                        let w = kernel[ky * K + kx] as f64;
                        let x_val = pred[(py * width + px) * 4 + ch] as f64;
                        let y_val = refer[(py * width + px) * 4 + ch] as f64;
                        mu_x += w * x_val;
                        mu_y += w * y_val;
                        sigma_x2 += w * x_val * x_val;
                        sigma_y2 += w * y_val * y_val;
                        sigma_xy += w * x_val * y_val;
                    }
                }

                // Convert weighted second moments to variances/covariance.
                sigma_x2 -= mu_x * mu_x;
                sigma_y2 -= mu_y * mu_y;
                sigma_xy -= mu_x * mu_y;

                // Clamp small negative due to floating-point noise.
                if sigma_x2 < 0.0 {
                    sigma_x2 = 0.0;
                }
                if sigma_y2 < 0.0 {
                    sigma_y2 = 0.0;
                }

                let numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
                let denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2);

                ssim_sum += numerator / denominator;
                pixel_count += 1;
            }
        }

        if pixel_count == 0 {
            channel_ssims[ch] = 1.0;
        } else {
            channel_ssims[ch] = ssim_sum / (pixel_count as f64);
        }
    }

    (channel_ssims[0] + channel_ssims[1] + channel_ssims[2]) / 3.0
}

// ─── MS-SSIM ────────────────────────────────────────────────────────────────

/// Compute Multi-Scale SSIM (MS-SSIM) at 3 scales.
///
/// Uses a weighted average:
///   MS-SSIM = (w0·SSIM_0 + w1·SSIM_1 + w2·SSIM_2) / (w0 + w1 + w2)
/// with w = [0.0448, 0.2856, 0.3001] (3-scale Wang et al. approximation).
///
/// Stops early if the image becomes too small (< 11 pixels wide or tall) for SSIM.
pub fn compute_ms_ssim(
    predicted: &[f32],
    reference: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, RenderError> {
    if predicted.len() != reference.len() {
        return Err(RenderError::MismatchedBufferSizes {
            expected: reference.len(),
            actual: predicted.len(),
        });
    }
    if predicted.len() != width * height * 4 {
        return Err(RenderError::MismatchedBufferSizes {
            expected: width * height * 4,
            actual: predicted.len(),
        });
    }

    const WEIGHTS: [f64; 3] = [0.0448, 0.2856, 0.3001];
    const MIN_DIM: usize = 11;

    let kernel = gaussian_kernel_11x11(1.5);

    let mut pred_cur: Vec<f32> = predicted.to_vec();
    let mut ref_cur: Vec<f32> = reference.to_vec();
    let mut w_cur = width;
    let mut h_cur = height;

    let mut weighted_ssim_sum = 0.0f64;
    let mut weight_sum = 0.0f64;

    for &weight in WEIGHTS.iter() {
        if w_cur < MIN_DIM || h_cur < MIN_DIM {
            break;
        }

        let ssim_val = ssim_over_channels(&pred_cur, &ref_cur, w_cur, h_cur, &kernel);
        weighted_ssim_sum += weight * ssim_val;
        weight_sum += weight;

        // Downsample for next scale.
        let (new_pred, nw, nh) = downsample_2x(&pred_cur, w_cur, h_cur);
        let (new_ref, _, _) = downsample_2x(&ref_cur, w_cur, h_cur);
        pred_cur = new_pred;
        ref_cur = new_ref;
        w_cur = nw;
        h_cur = nh;
    }

    if weight_sum == 0.0 {
        // Image was already too small; fall back to single-scale SSIM.
        return compute_ssim(predicted, reference, width, height);
    }

    Ok((weighted_ssim_sum / weight_sum) as f32)
}

// ─── RenderQualityMetrics ───────────────────────────────────────────────────

/// Complete set of rendering quality metrics computed between a predicted and
/// a reference RGBA image.
#[derive(Debug, Clone)]
pub struct RenderQualityMetrics {
    /// Peak Signal-to-Noise Ratio in dB.
    pub psnr: f32,
    /// Structural Similarity Index in \[−1, 1\].
    pub ssim: f32,
    /// Multi-Scale SSIM in \[−1, 1\].
    pub ms_ssim: f32,
    /// Mean absolute pixel difference (RGB only).
    pub mean_abs_error: f32,
    /// Maximum absolute pixel difference (RGB only).
    pub max_abs_error: f32,
}

impl RenderQualityMetrics {
    /// Compute all quality metrics for `predicted` vs `reference`.
    ///
    /// Both slices must be row-major RGBA f32 of length `width * height * 4`.
    pub fn compute(
        predicted: &[f32],
        reference: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Self, RenderError> {
        if predicted.len() != width * height * 4 {
            return Err(RenderError::MismatchedBufferSizes {
                expected: width * height * 4,
                actual: predicted.len(),
            });
        }
        if predicted.len() != reference.len() {
            return Err(RenderError::MismatchedBufferSizes {
                expected: predicted.len(),
                actual: reference.len(),
            });
        }

        let psnr = compute_psnr(predicted, reference, 1.0)?;
        let ssim = compute_ssim(predicted, reference, width, height)?;
        let ms_ssim = compute_ms_ssim(predicted, reference, width, height)?;

        // MAE and max AE over RGB.
        let mut sum_abs = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut rgb_count = 0usize;

        for (i, (&p, &r)) in predicted.iter().zip(reference.iter()).enumerate() {
            if i % 4 == 3 {
                continue;
            }
            let diff = (p - r).abs();
            sum_abs += diff as f64;
            if diff > max_abs {
                max_abs = diff;
            }
            rgb_count += 1;
        }

        let mean_abs_error = if rgb_count == 0 {
            0.0
        } else {
            (sum_abs / rgb_count as f64) as f32
        };

        Ok(Self {
            psnr,
            ssim,
            ms_ssim,
            mean_abs_error,
            max_abs_error: max_abs,
        })
    }

    /// Returns `true` when PSNR > 30 dB and SSIM > 0.9.
    pub fn is_high_quality(&self) -> bool {
        self.psnr > 30.0 && self.ssim > 0.9
    }

    /// Return a human-readable formatted report of all metrics.
    pub fn format_report(&self) -> String {
        format!(
            "┌─────────────────────────────────┐\n\
             │   Render Quality Metrics        │\n\
             ├─────────────────┬───────────────┤\n\
             │ PSNR            │ {:>10.4} dB │\n\
             │ SSIM            │ {:>13.6} │\n\
             │ MS-SSIM         │ {:>13.6} │\n\
             │ Mean Abs Error  │ {:>13.6} │\n\
             │ Max Abs Error   │ {:>13.6} │\n\
             └─────────────────┴───────────────┘",
            self.psnr, self.ssim, self.ms_ssim, self.mean_abs_error, self.max_abs_error
        )
    }
}

// ─── MetricThresholds ───────────────────────────────────────────────────────

/// Quality thresholds for pass/fail assessment of rendering metrics.
#[derive(Debug, Clone)]
pub struct MetricThresholds {
    /// Minimum acceptable PSNR in dB.
    pub min_psnr: f32,
    /// Minimum acceptable SSIM.
    pub min_ssim: f32,
    /// Minimum acceptable MS-SSIM.
    pub min_ms_ssim: f32,
    /// Maximum acceptable mean absolute error.
    pub max_mae: f32,
}

impl Default for MetricThresholds {
    fn default() -> Self {
        Self {
            min_psnr: 25.0,
            min_ssim: 0.85,
            min_ms_ssim: 0.90,
            max_mae: 0.05,
        }
    }
}

impl MetricThresholds {
    /// Strict thresholds: PSNR ≥ 35 dB, SSIM ≥ 0.95, MS-SSIM ≥ 0.97, MAE ≤ 0.01.
    pub fn strict() -> Self {
        Self {
            min_psnr: 35.0,
            min_ssim: 0.95,
            min_ms_ssim: 0.97,
            max_mae: 0.01,
        }
    }

    /// Permissive thresholds: PSNR ≥ 20 dB, SSIM ≥ 0.70, MS-SSIM ≥ 0.80, MAE ≤ 0.10.
    pub fn permissive() -> Self {
        Self {
            min_psnr: 20.0,
            min_ssim: 0.70,
            min_ms_ssim: 0.80,
            max_mae: 0.10,
        }
    }

    /// Check whether `metrics` satisfies all thresholds.
    ///
    /// Returns a list of failure descriptions; empty means all pass.
    pub fn check(&self, metrics: &RenderQualityMetrics) -> Vec<String> {
        let mut failures = Vec::new();

        if metrics.psnr < self.min_psnr && metrics.psnr.is_finite() {
            failures.push(format!(
                "PSNR {:.4} dB is below minimum {:.4} dB",
                metrics.psnr, self.min_psnr
            ));
        }
        if metrics.ssim < self.min_ssim {
            failures.push(format!(
                "SSIM {:.6} is below minimum {:.6}",
                metrics.ssim, self.min_ssim
            ));
        }
        if metrics.ms_ssim < self.min_ms_ssim {
            failures.push(format!(
                "MS-SSIM {:.6} is below minimum {:.6}",
                metrics.ms_ssim, self.min_ms_ssim
            ));
        }
        if metrics.mean_abs_error > self.max_mae {
            failures.push(format!(
                "MAE {:.6} exceeds maximum {:.6}",
                metrics.mean_abs_error, self.max_mae
            ));
        }

        failures
    }
}

// ─── compare_renders ────────────────────────────────────────────────────────

/// Compute quality metrics for each render paired with the corresponding
/// reference image.
///
/// `renders` and `reference` must have the same number of elements.
pub fn compare_renders(
    renders: &[Vec<f32>],
    reference: &[Vec<f32>],
    width: usize,
    height: usize,
) -> Result<Vec<RenderQualityMetrics>, RenderError> {
    if renders.len() != reference.len() {
        return Err(RenderError::MismatchedBufferSizes {
            expected: reference.len(),
            actual: renders.len(),
        });
    }

    renders
        .iter()
        .zip(reference.iter())
        .map(|(pred, refer)| RenderQualityMetrics::compute(pred, refer, width, height))
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Create a solid-colour RGBA image of `width × height`.
    fn solid_rgba(r: f32, g: f32, b: f32, a: f32, width: usize, height: usize) -> Vec<f32> {
        let n = width * height * 4;
        let mut buf = Vec::with_capacity(n);
        for _ in 0..(width * height) {
            buf.push(r);
            buf.push(g);
            buf.push(b);
            buf.push(a);
        }
        buf
    }

    /// Add a constant offset to all RGB channels (not alpha).
    fn add_noise(image: &[f32], delta: f32) -> Vec<f32> {
        image
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i % 4 == 3 {
                    v
                } else {
                    (v + delta).clamp(0.0, 1.0)
                }
            })
            .collect()
    }

    // ── PSNR tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_psnr_identical_images() {
        let img = solid_rgba(0.5, 0.5, 0.5, 1.0, 32, 32);
        let psnr = compute_psnr(&img, &img, 1.0).expect("psnr failed");
        assert!(
            psnr.is_infinite(),
            "identical images must yield infinite PSNR"
        );
    }

    #[test]
    fn test_psnr_all_zeros_vs_all_ones() {
        // MSE = 1.0 (over [0,1] RGB), PSNR = 10*log10(1) = 0 dB
        let zeros = solid_rgba(0.0, 0.0, 0.0, 0.0, 8, 8);
        let ones = solid_rgba(1.0, 1.0, 1.0, 1.0, 8, 8);
        let psnr = compute_psnr(&zeros, &ones, 1.0).expect("psnr failed");
        // MSE = 1.0, max_val = 1.0 → PSNR = 0 dB
        assert!((psnr - 0.0).abs() < 1e-4, "expected 0 dB, got {psnr}");
    }

    #[test]
    fn test_psnr_small_error() {
        let base = solid_rgba(0.5, 0.5, 0.5, 1.0, 16, 16);
        let noisy = add_noise(&base, 0.01);
        let psnr = compute_psnr(&noisy, &base, 1.0).expect("psnr failed");
        // MSE ~ 0.0001, PSNR ~ 40 dB
        assert!(psnr > 35.0 && psnr < 50.0, "expected ~40 dB, got {psnr}");
    }

    #[test]
    fn test_psnr_length_mismatch_error() {
        let a = vec![0.0f32; 16];
        let b = vec![0.0f32; 32];
        assert!(
            compute_psnr(&a, &b, 1.0).is_err(),
            "must error on length mismatch"
        );
    }

    // ── SSIM tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_ssim_identical_images() {
        let img = solid_rgba(0.5, 0.3, 0.8, 1.0, 32, 32);
        let ssim = compute_ssim(&img, &img, 32, 32).expect("ssim failed");
        assert!(
            (ssim - 1.0).abs() < 1e-4,
            "identical images must yield SSIM ≈ 1.0, got {ssim}"
        );
    }

    #[test]
    fn test_ssim_completely_different() {
        let black = solid_rgba(0.0, 0.0, 0.0, 1.0, 32, 32);
        let white = solid_rgba(1.0, 1.0, 1.0, 1.0, 32, 32);
        let ssim = compute_ssim(&black, &white, 32, 32).expect("ssim failed");
        assert!(
            ssim < 0.5,
            "completely different images must yield SSIM < 0.5, got {ssim}"
        );
    }

    #[test]
    fn test_ssim_small_perturbation() {
        let base = solid_rgba(0.5, 0.5, 0.5, 1.0, 32, 32);
        let noisy = add_noise(&base, 0.02);
        let ssim = compute_ssim(&noisy, &base, 32, 32).expect("ssim failed");
        assert!(
            ssim > 0.9,
            "small perturbation must yield SSIM > 0.9, got {ssim}"
        );
    }

    #[test]
    fn test_ssim_image_size_mismatch_error() {
        let a = solid_rgba(0.5, 0.5, 0.5, 1.0, 8, 8); // 8*8*4 = 256
        let b = solid_rgba(0.5, 0.5, 0.5, 1.0, 16, 16); // 16*16*4 = 1024
        assert!(
            compute_ssim(&a, &b, 8, 8).is_err(),
            "length mismatch must error"
        );
    }

    // ── MS-SSIM tests ────────────────────────────────────────────────────────

    #[test]
    fn test_ms_ssim_identical() {
        let img = solid_rgba(0.4, 0.6, 0.2, 1.0, 64, 64);
        let ms = compute_ms_ssim(&img, &img, 64, 64).expect("ms-ssim failed");
        assert!(
            (ms - 1.0).abs() < 1e-4,
            "identical images must yield MS-SSIM ≈ 1.0, got {ms}"
        );
    }

    #[test]
    fn test_ms_ssim_different() {
        let black = solid_rgba(0.0, 0.0, 0.0, 1.0, 64, 64);
        let white = solid_rgba(1.0, 1.0, 1.0, 1.0, 64, 64);
        let ms = compute_ms_ssim(&black, &white, 64, 64).expect("ms-ssim failed");
        assert!(
            ms < 0.5,
            "different images must yield MS-SSIM < 0.5, got {ms}"
        );
    }

    // ── Gaussian kernel test ─────────────────────────────────────────────────

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let kernel = gaussian_kernel_11x11(1.5);
        assert_eq!(kernel.len(), 121, "kernel must have 121 entries");
        let sum: f32 = kernel.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "kernel must sum to 1.0, got {sum}"
        );
    }

    // ── downsample_2x tests ──────────────────────────────────────────────────

    #[test]
    fn test_downsample_2x_shape() {
        let img = solid_rgba(0.5, 0.5, 0.5, 1.0, 32, 16);
        let (out, nw, nh) = downsample_2x(&img, 32, 16);
        assert_eq!(nw, 16);
        assert_eq!(nh, 8);
        assert_eq!(out.len(), 16 * 8 * 4);
    }

    #[test]
    fn test_downsample_2x_uniform_image() {
        // Averaging a uniform image must produce the same uniform colour.
        let img = solid_rgba(0.3, 0.7, 0.1, 0.5, 8, 8);
        let (out, nw, nh) = downsample_2x(&img, 8, 8);
        assert_eq!(nw, 4);
        assert_eq!(nh, 4);
        for chunk in out.chunks(4) {
            assert!((chunk[0] - 0.3).abs() < 1e-5, "R channel mismatch");
            assert!((chunk[1] - 0.7).abs() < 1e-5, "G channel mismatch");
            assert!((chunk[2] - 0.1).abs() < 1e-5, "B channel mismatch");
            assert!((chunk[3] - 0.5).abs() < 1e-5, "A channel mismatch");
        }
    }

    // ── RenderQualityMetrics tests ───────────────────────────────────────────

    #[test]
    fn test_render_quality_metrics_compute() {
        let img = solid_rgba(0.5, 0.5, 0.5, 1.0, 16, 16);
        let noisy = add_noise(&img, 0.01);
        let metrics =
            RenderQualityMetrics::compute(&noisy, &img, 16, 16).expect("metrics compute failed");
        assert!(
            metrics.psnr > 30.0,
            "expected PSNR > 30, got {}",
            metrics.psnr
        );
        assert!(
            metrics.ssim > 0.9,
            "expected SSIM > 0.9, got {}",
            metrics.ssim
        );
        assert!(
            metrics.ms_ssim > 0.9,
            "expected MS-SSIM > 0.9, got {}",
            metrics.ms_ssim
        );
        assert!(
            metrics.mean_abs_error < 0.02,
            "expected MAE < 0.02, got {}",
            metrics.mean_abs_error
        );
    }

    #[test]
    fn test_render_quality_metrics_is_high_quality() {
        let img = solid_rgba(0.5, 0.5, 0.5, 1.0, 16, 16);
        // Identical → PSNR = inf, SSIM ≈ 1.0 → high quality
        let m = RenderQualityMetrics::compute(&img, &img, 16, 16).expect("compute failed");
        assert!(m.is_high_quality(), "identical images must be high quality");

        // Completely different → not high quality
        let bad = solid_rgba(1.0, 0.0, 0.0, 0.0, 16, 16);
        let m2 = RenderQualityMetrics::compute(&bad, &img, 16, 16).expect("compute failed");
        assert!(
            !m2.is_high_quality(),
            "completely different must not be high quality"
        );
    }

    // ── MetricThresholds tests ───────────────────────────────────────────────

    #[test]
    fn test_metric_thresholds_default() {
        let t = MetricThresholds::default();
        assert_eq!(t.min_psnr, 25.0);
        assert_eq!(t.min_ssim, 0.85);
        assert_eq!(t.min_ms_ssim, 0.90);
        assert_eq!(t.max_mae, 0.05);
    }

    #[test]
    fn test_metric_thresholds_check_passing() {
        let img = solid_rgba(0.5, 0.5, 0.5, 1.0, 32, 32);
        let noisy = add_noise(&img, 0.005); // very small noise → high quality
        let metrics = RenderQualityMetrics::compute(&noisy, &img, 32, 32).expect("compute failed");
        let failures = MetricThresholds::default().check(&metrics);
        assert!(
            failures.is_empty(),
            "expected all pass but got failures: {failures:?}"
        );
    }

    #[test]
    fn test_metric_thresholds_check_failing() {
        let black = solid_rgba(0.0, 0.0, 0.0, 1.0, 32, 32);
        let white = solid_rgba(1.0, 1.0, 1.0, 1.0, 32, 32);
        let metrics = RenderQualityMetrics::compute(&black, &white, 32, 32).expect("compute");
        let failures = MetricThresholds::default().check(&metrics);
        assert!(!failures.is_empty(), "expected failures for black vs white");
    }

    // ── compare_renders test ─────────────────────────────────────────────────

    #[test]
    fn test_compare_renders_multiple() {
        let w = 16;
        let h = 16;
        let renders: Vec<Vec<f32>> = vec![
            solid_rgba(0.5, 0.5, 0.5, 1.0, w, h),
            solid_rgba(0.2, 0.8, 0.4, 1.0, w, h),
        ];
        let references: Vec<Vec<f32>> =
            vec![add_noise(&renders[0], 0.01), add_noise(&renders[1], 0.01)];
        let results = compare_renders(&renders, &references, w, h).expect("compare_renders failed");
        assert_eq!(results.len(), 2);
        for m in &results {
            assert!(m.psnr > 30.0, "expected PSNR > 30, got {}", m.psnr);
        }

        // Length mismatch must error.
        let one_render = vec![renders[0].clone()];
        assert!(compare_renders(&one_render, &references, w, h).is_err());
    }
}
