//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use rayon::prelude::*;

use super::types::{
    EvalComparison, EvalConfig, EvalError, EvalMetricKind, EvalSuiteResult, EvalTestItem,
    ViewEvalResult,
};

/// Build an 11×11 Gaussian kernel with σ = 1.5, normalised to sum ≈ 1.
pub fn eval_gaussian_kernel_11() -> [f32; 121] {
    let sigma = 1.5_f32;
    let half = 5_i32;
    let mut kernel = [0.0_f32; 121];
    let mut total = 0.0_f32;
    for y in -half..=half {
        for x in -half..=half {
            let v = (-(x * x + y * y) as f32 / (2.0 * sigma * sigma)).exp();
            let idx = ((y + half) * 11 + (x + half)) as usize;
            kernel[idx] = v;
            total += v;
        }
    }
    for v in kernel.iter_mut() {
        *v /= total;
    }
    kernel
}
/// Apply a 2-D convolution with a square `k_size × k_size` kernel to a
/// single-channel image.  Zero-padding is used at the borders so the output
/// has the same length as the input (`width * height`).
///
/// `image` and `kernel` are read with bounds-checked accessors: a caller
/// that supplies a buffer shorter than `width * height` (or `k_size *
/// k_size`) gets 0.0 for the missing samples instead of an index-out-of-
/// bounds panic. This function is `pub` and re-exported, so it must stay
/// well-defined for any external caller's dimensions rather than trusting
/// them.
pub fn eval_convolve(
    image: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    k_size: usize,
) -> Vec<f32> {
    let half = (k_size / 2) as i32;
    let mut out = vec![0.0_f32; width * height];
    for row in 0..height {
        for col in 0..width {
            let mut acc = 0.0_f32;
            for ky in 0..k_size {
                let iy = row as i32 + ky as i32 - half;
                if iy < 0 || iy >= height as i32 {
                    continue;
                }
                for kx in 0..k_size {
                    let ix = col as i32 + kx as i32 - half;
                    if ix < 0 || ix >= width as i32 {
                        continue;
                    }
                    let ki = ky * k_size + kx;
                    let pi = iy as usize * width + ix as usize;
                    let k = kernel.get(ki).copied().unwrap_or(0.0);
                    let p = image.get(pi).copied().unwrap_or(0.0);
                    acc += k * p;
                }
            }
            out[row * width + col] = acc;
        }
    }
    out
}
/// Downsample an RGB image by 2× using a box (average) filter.
/// Output dimensions are `(ceil(width/2), ceil(height/2))`.
/// Returns `(pixels, new_width, new_height)`.
pub fn eval_downsample_2x(image: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let new_w = width.div_ceil(2);
    let new_h = height.div_ceil(2);
    let mut out = vec![0.0_f32; new_w * new_h * 3];
    for oy in 0..new_h {
        for ox in 0..new_w {
            let src_x0 = ox * 2;
            let src_y0 = oy * 2;
            let src_x1 = (src_x0 + 1).min(width - 1);
            let src_y1 = (src_y0 + 1).min(height - 1);
            for c in 0..3 {
                let p00 = image[(src_y0 * width + src_x0) * 3 + c];
                let p10 = image[(src_y0 * width + src_x1) * 3 + c];
                let p01 = image[(src_y1 * width + src_x0) * 3 + c];
                let p11 = image[(src_y1 * width + src_x1) * 3 + c];
                let count = if (width == 1 && height == 1) || (src_x0 == src_x1 && src_y0 == src_y1)
                {
                    1.0_f32
                } else if src_x0 == src_x1 || src_y0 == src_y1 {
                    2.0
                } else {
                    4.0
                };
                let sum = p00 + p10 + p01 + p11;
                let avg = if count < 4.0 {
                    let mut s2 = 0.0_f32;
                    let mut cnt2 = 0usize;
                    let xs = if src_x0 == src_x1 {
                        vec![src_x0]
                    } else {
                        vec![src_x0, src_x1]
                    };
                    let ys = if src_y0 == src_y1 {
                        vec![src_y0]
                    } else {
                        vec![src_y0, src_y1]
                    };
                    for &sy in &ys {
                        for &sx in &xs {
                            s2 += image[(sy * width + sx) * 3 + c];
                            cnt2 += 1;
                        }
                    }
                    if cnt2 == 0 {
                        0.0
                    } else {
                        s2 / cnt2 as f32
                    }
                } else {
                    sum / 4.0
                };
                out[(oy * new_w + ox) * 3 + c] = avg;
            }
        }
    }
    (out, new_w, new_h)
}
/// Compute Sobel edge magnitude map.  Input is an interleaved RGB image
/// ([0, 1] range); output is a single-channel grayscale magnitude map of the
/// same spatial dimensions.
///
/// Reads `image` with a bounds-checked accessor: a caller-supplied buffer
/// shorter than `width * height * 3` yields 0.0 for the missing channels
/// instead of panicking (see [`eval_convolve`] for the same policy).
pub fn eval_sobel(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let mut gray = vec![0.0_f32; n];
    for (i, luma) in gray.iter_mut().enumerate() {
        let r = image.get(i * 3).copied().unwrap_or(0.0);
        let g = image.get(i * 3 + 1).copied().unwrap_or(0.0);
        let b = image.get(i * 3 + 2).copied().unwrap_or(0.0);
        *luma = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    let gx_kernel: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let gy_kernel: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let gx = eval_convolve(&gray, width, height, &gx_kernel, 3);
    let gy = eval_convolve(&gray, width, height, &gy_kernel, 3);
    let mut mag = vec![0.0_f32; n];
    for i in 0..n {
        mag[i] = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
    }
    mag
}

// ---------------------------------------------------------------------------
// SSIM internals
// ---------------------------------------------------------------------------

/// The 1-D Gaussian kernel (11 taps, σ = 1.5) whose outer product forms
/// [`eval_gaussian_kernel_11`]'s 2-D kernel — used to blur the SSIM window
/// as two cheap 1-D passes instead of one dense 2-D convolution.
fn gaussian_kernel_11_1d() -> [f32; 11] {
    let sigma = 1.5_f32;
    let half = 5_i32;
    let mut kernel = [0.0_f32; 11];
    let mut total = 0.0_f32;
    for x in -half..=half {
        let v = (-(x * x) as f32 / (2.0 * sigma * sigma)).exp();
        kernel[(x + half) as usize] = v;
        total += v;
    }
    for v in kernel.iter_mut() {
        *v /= total;
    }
    kernel
}

/// Separable 11×11 Gaussian blur (σ = 1.5): an 11-tap horizontal pass
/// followed by an 11-tap vertical pass, each zero-padded at the borders
/// exactly like [`eval_convolve`]. Because a Gaussian kernel is separable
/// (`G(x,y) = G(x)·G(y)`), this is numerically identical to
/// `eval_convolve(image, width, height, &eval_gaussian_kernel_11(), 11)` at
/// every position whose window does not touch the border — which is all
/// [`ssim_components`] ever samples once it crops the border ring away —
/// but costs O(22) multiply-adds per pixel instead of O(121).
fn gaussian_blur_11_separable(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let k1d = gaussian_kernel_11_1d();
    let half = 5_i32;

    let mut tmp = vec![0.0_f32; width * height];
    for row in 0..height {
        for col in 0..width {
            let mut acc = 0.0_f32;
            for (kx, &kw) in k1d.iter().enumerate() {
                let ix = col as i32 + kx as i32 - half;
                if ix < 0 || ix >= width as i32 {
                    continue;
                }
                acc += kw * image[row * width + ix as usize];
            }
            tmp[row * width + col] = acc;
        }
    }

    let mut out = vec![0.0_f32; width * height];
    for row in 0..height {
        for col in 0..width {
            let mut acc = 0.0_f32;
            for (ky, &kw) in k1d.iter().enumerate() {
                let iy = row as i32 + ky as i32 - half;
                if iy < 0 || iy >= height as i32 {
                    continue;
                }
                acc += kw * tmp[iy as usize * width + col];
            }
            out[row * width + col] = acc;
        }
    }
    out
}

/// Internal: compute the mean SSIM luminance term and mean contrast-
/// structure term for a single-channel image pair, using an 11×11 Gaussian
/// window (σ = 1.5). The full per-pixel SSIM is `luminance × contrast ×
/// structure`; multi-scale SSIM needs the contrast-structure term on its
/// own, so both are returned separately.
///
/// For images at least 11×11 in both dimensions, the average is restricted
/// to the region where the window is fully contained in the image (a
/// 5-pixel ring is excluded on every side), matching the "valid"
/// convolution mode used by reference SSIM implementations (skimage, the
/// original Wang et al. code) instead of biasing the result toward the
/// window's implicit zero-padding at the border. Images smaller than 11×11
/// in either dimension have no such interior region, so the average falls
/// back to the full (zero-padded) frame — this keeps the metric defined for
/// tiny images (thumbnails, unit tests) at the cost of some border bias,
/// which only matters below the window size and never affects any
/// realistic evaluation view.
fn ssim_components(chan: &[f32], ref_chan: &[f32], width: usize, height: usize) -> (f32, f32) {
    const HALF: usize = 5; // (11 / 2)
    let mu_x = gaussian_blur_11_separable(chan, width, height);
    let mu_y = gaussian_blur_11_separable(ref_chan, width, height);
    let n = width * height;
    let mut xx = vec![0.0_f32; n];
    let mut yy = vec![0.0_f32; n];
    let mut xy = vec![0.0_f32; n];
    for i in 0..n {
        xx[i] = chan[i] * chan[i];
        yy[i] = ref_chan[i] * ref_chan[i];
        xy[i] = chan[i] * ref_chan[i];
    }
    let mu_xx = gaussian_blur_11_separable(&xx, width, height);
    let mu_yy = gaussian_blur_11_separable(&yy, width, height);
    let mu_xy = gaussian_blur_11_separable(&xy, width, height);

    const C1: f32 = 0.0001;
    const C2: f32 = 0.0009;

    let valid = width > 2 * HALF && height > 2 * HALF;
    let (row_lo, row_hi) = if valid {
        (HALF, height - HALF)
    } else {
        (0, height)
    };
    let (col_lo, col_hi) = if valid {
        (HALF, width - HALF)
    } else {
        (0, width)
    };

    let mut l_sum = 0.0_f64;
    let mut cs_sum = 0.0_f64;
    let mut count = 0usize;
    for row in row_lo..row_hi {
        for col in col_lo..col_hi {
            let i = row * width + col;
            let mux = mu_x[i];
            let muy = mu_y[i];
            let sig_x = mu_xx[i] - mux * mux;
            let sig_y = mu_yy[i] - muy * muy;
            let sig_xy = mu_xy[i] - mux * muy;
            let l = (2.0 * mux * muy + C1) / (mux * mux + muy * muy + C1);
            let cs = (2.0 * sig_xy + C2) / (sig_x + sig_y + C2);
            l_sum += l as f64;
            cs_sum += cs as f64;
            count += 1;
        }
    }
    let count = count.max(1) as f64;
    ((l_sum / count) as f32, (cs_sum / count) as f32)
}

/// Average the SSIM luminance term and the contrast-structure term across
/// the three RGB channels of a pair of same-sized images. Returns
/// `(luminance_mean, contrast_structure_mean)`; the full SSIM value is
/// their product. `pred`/`gt` must already be validated as
/// `width * height * 3` interleaved RGB buffers (see [`validate_rgb_pair`]).
fn ssim_and_cs(pred: &[f32], gt: &[f32], width: usize, height: usize) -> (f32, f32) {
    let n = width * height;
    let mut l_total = 0.0_f32;
    let mut cs_total = 0.0_f32;
    for c in 0..3 {
        let mut pred_c = vec![0.0_f32; n];
        let mut gt_c = vec![0.0_f32; n];
        for i in 0..n {
            pred_c[i] = pred[i * 3 + c];
            gt_c[i] = gt[i * 3 + c];
        }
        let (l, cs) = ssim_components(&pred_c, &gt_c, width, height);
        l_total += l;
        cs_total += cs;
    }
    (l_total / 3.0, cs_total / 3.0)
}

/// Validate that `pred`/`gt` are equal-length flat interleaved RGB buffers
/// of exactly `width * height * 3` elements, returning a descriptive
/// [`EvalError`] otherwise. `label` identifies which metric rejected the
/// input, for a clearer error message.
fn validate_rgb_pair(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
    label: &str,
) -> Result<(), EvalError> {
    if pred.len() != gt.len() {
        return Err(EvalError::DimensionMismatch {
            pred: pred.len(),
            gt: gt.len(),
        });
    }
    let expected = width * height * 3;
    if pred.len() != expected {
        return Err(EvalError::MetricFailed(format!(
            "{label}: buffer length {} does not match {}×{}×3={}",
            pred.len(),
            width,
            height,
            expected
        )));
    }
    if pred.is_empty() {
        return Err(EvalError::MetricFailed(format!("{label}: empty image")));
    }
    Ok(())
}

/// PSNR between prediction and ground truth (flat interleaved RGB, \[0,1\]).
///
/// Returns `f32::INFINITY` when images are identical (MSE = 0).
pub fn eval_psnr(pred: &[f32], gt: &[f32]) -> Result<f32, EvalError> {
    if pred.len() != gt.len() {
        return Err(EvalError::DimensionMismatch {
            pred: pred.len(),
            gt: gt.len(),
        });
    }
    if pred.is_empty() {
        return Err(EvalError::DimensionMismatch { pred: 0, gt: 0 });
    }
    let mse: f64 = pred
        .iter()
        .zip(gt.iter())
        .map(|(p, g)| {
            let d = (*p - *g) as f64;
            d * d
        })
        .sum::<f64>()
        / pred.len() as f64;
    if mse == 0.0 {
        return Ok(f32::INFINITY);
    }
    Ok((10.0 * (1.0_f64 / mse).log10()) as f32)
}
/// Mean absolute error between prediction and ground truth.
pub fn eval_mae(pred: &[f32], gt: &[f32]) -> Result<f32, EvalError> {
    if pred.len() != gt.len() {
        return Err(EvalError::DimensionMismatch {
            pred: pred.len(),
            gt: gt.len(),
        });
    }
    if pred.is_empty() {
        return Err(EvalError::DimensionMismatch { pred: 0, gt: 0 });
    }
    let sum: f64 = pred
        .iter()
        .zip(gt.iter())
        .map(|(p, g)| (*p - *g).abs() as f64)
        .sum();
    Ok((sum / pred.len() as f64) as f32)
}
/// Root mean square error between prediction and ground truth.
pub fn eval_rmse(pred: &[f32], gt: &[f32]) -> Result<f32, EvalError> {
    if pred.len() != gt.len() {
        return Err(EvalError::DimensionMismatch {
            pred: pred.len(),
            gt: gt.len(),
        });
    }
    if pred.is_empty() {
        return Err(EvalError::DimensionMismatch { pred: 0, gt: 0 });
    }
    let mse: f64 = pred
        .iter()
        .zip(gt.iter())
        .map(|(p, g)| {
            let d = (*p - *g) as f64;
            d * d
        })
        .sum::<f64>()
        / pred.len() as f64;
    Ok(mse.sqrt() as f32)
}
/// SSIM with an 11×11 Gaussian window (σ=1.5), averaged over RGB channels.
///
/// Both `pred` and `gt` must have length `width * height * 3`. For images
/// at least 11×11 the average excludes the border ring the window cannot
/// fully cover without zero-padding (see the module-private
/// `ssim_components`); smaller images fall back to averaging the full padded
/// frame.
pub fn eval_ssim(pred: &[f32], gt: &[f32], width: usize, height: usize) -> Result<f32, EvalError> {
    validate_rgb_pair(pred, gt, width, height, "SSIM")?;
    let (l_mean, cs_mean) = ssim_and_cs(pred, gt, width, height);
    Ok(l_mean * cs_mean)
}

/// Core multi-scale SSIM computation (Wang, Simoncelli & Bovik, 2003): the
/// standard multiplicative combination `MS-SSIM = [l_M]^αM · Π[cs_j]^βj`,
/// with contrast-structure only at every scale but the last, and luminance
/// included solely at the coarsest scale actually used.
///
/// Uses as many of the 5 canonical scales (weights
/// `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`, renormalised to the subset
/// used) as the image supports: each extra scale halves the resolution,
/// and every scale's SSIM window needs at least an 11×11 image to avoid
/// degenerating into mostly zero-padding, so smaller images transparently
/// fall back to fewer scales (a plain single-scale SSIM in the smallest
/// case) rather than erroring or reporting a meaningless coarse-scale term.
/// This makes the metric resolution-dependent — the same pair of images
/// yields a different MS-SSIM value at 48×48 (3 scales) than at 512×512
/// (5 scales) — which matches how every reference MS-SSIM implementation
/// behaves (real implementations simply require a minimum input size).
///
/// `scale0` lets a caller that has already computed the full-resolution
/// SSIM components (e.g. [`eval_single_view`], which also needs plain
/// SSIM) pass them in directly instead of recomputing the identical
/// convolutions for the first scale.
fn eval_ssim_ms_impl(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
    scale0: Option<(f32, f32)>,
) -> f32 {
    const WEIGHTS: [f32; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
    let min_dim = width.min(height);
    let mut n_scales = WEIGHTS.len();
    while n_scales > 1 && (min_dim >> (n_scales - 1)) < 11 {
        n_scales -= 1;
    }
    let weight_sum: f32 = WEIGHTS[..n_scales].iter().sum();

    let mut p_cur = pred.to_vec();
    let mut g_cur = gt.to_vec();
    let mut w_cur = width;
    let mut h_cur = height;
    let mut ms_ssim = 1.0_f32;

    for (scale, &raw_weight) in WEIGHTS[..n_scales].iter().enumerate() {
        let weight = raw_weight / weight_sum;
        let (l_mean, cs_mean) = if scale == 0 {
            scale0.unwrap_or_else(|| ssim_and_cs(&p_cur, &g_cur, w_cur, h_cur))
        } else {
            ssim_and_cs(&p_cur, &g_cur, w_cur, h_cur)
        };
        // Clamp before raising to a fractional power: SSIM/CS terms can be
        // (rarely) negative for anti-correlated patches, and a negative
        // base with a non-integer exponent is NaN in IEEE 754.
        let term = if scale == n_scales - 1 {
            (l_mean * cs_mean).max(0.0)
        } else {
            cs_mean.max(0.0)
        };
        ms_ssim *= term.powf(weight);
        if scale < n_scales - 1 {
            // `eval_downsample_2x` is a pure function of (width, height), so
            // calling it with the same w_cur/h_cur for both images always
            // produces matching output dimensions — no mismatch to guard.
            let (pd, pw, ph) = eval_downsample_2x(&p_cur, w_cur, h_cur);
            let (gd, _, _) = eval_downsample_2x(&g_cur, w_cur, h_cur);
            p_cur = pd;
            g_cur = gd;
            w_cur = pw;
            h_cur = ph;
        }
    }
    ms_ssim
}

/// Multi-scale SSIM (Wang, Simoncelli & Bovik, 2003). See the module-private
/// `eval_ssim_ms_impl` for the exact formulation and its resolution-dependent
/// scale count.
pub fn eval_ssim_ms(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, EvalError> {
    validate_rgb_pair(pred, gt, width, height, "MS-SSIM")?;
    Ok(eval_ssim_ms_impl(pred, gt, width, height, None))
}
/// LPIPS approximation using Sobel gradient features.
///
/// Computes Sobel edge magnitude maps for both images and returns the
/// mean L1 distance between them.  Lower is better (closer perceptual match).
pub fn eval_lpips_approx(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, EvalError> {
    if pred.len() != gt.len() {
        return Err(EvalError::DimensionMismatch {
            pred: pred.len(),
            gt: gt.len(),
        });
    }
    let expected = width * height * 3;
    if pred.len() != expected {
        return Err(EvalError::MetricFailed(format!(
            "LPIPS: buffer length {} does not match {}×{}×3={}",
            pred.len(),
            width,
            height,
            expected
        )));
    }
    if pred.is_empty() {
        return Err(EvalError::MetricFailed("LPIPS: empty image".to_string()));
    }
    let pred_edges = eval_sobel(pred, width, height);
    let gt_edges = eval_sobel(gt, width, height);
    let n = width * height;
    let l1: f64 = pred_edges
        .iter()
        .zip(gt_edges.iter())
        .map(|(p, g)| (*p - *g).abs() as f64)
        .sum::<f64>()
        / n as f64;
    Ok(l1 as f32)
}
/// Every metric this module can compute, in [`EvalMetricKind`] declaration
/// order. Used as the default selection by [`eval_single_view`].
const ALL_METRICS: [EvalMetricKind; 6] = [
    EvalMetricKind::Psnr,
    EvalMetricKind::Ssim,
    EvalMetricKind::LpipsApprox,
    EvalMetricKind::Mae,
    EvalMetricKind::Rmse,
    EvalMetricKind::SsimMs,
];

/// Compute all metrics for a single view pair.
///
/// SSIM is computed once via the shared internal helper and its
/// full-resolution components are reused as MS-SSIM's first scale, rather
/// than recomputing the identical convolutions twice.
///
/// Equivalent to [`eval_single_view_with_metrics`] with every
/// [`EvalMetricKind`] selected.
pub fn eval_single_view(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
    view_id: &str,
) -> Result<ViewEvalResult, EvalError> {
    eval_single_view_with_metrics(pred, gt, width, height, view_id, &ALL_METRICS)
}

/// Compute the selected `metrics` for a single view pair, leaving every
/// unselected metric as `f32::NAN`.
///
/// Skipping a metric skips its work: omitting [`EvalMetricKind::Ssim`] *and*
/// [`EvalMetricKind::SsimMs`] avoids the Gaussian-window convolutions
/// entirely (by far the most expensive part of a full evaluation), and
/// omitting [`EvalMetricKind::LpipsApprox`] avoids both Sobel passes.
///
/// # PSNR is unconditional
///
/// [`EvalMetricKind::Psnr`] is always computed, whether or not it appears in
/// `metrics`. It is a single cheap pass over the buffers, and it is the
/// ranking key that the rest of [`EvalSuiteResult`] is *defined* in terms of:
/// `worst_views`, `best_views`, `std_psnr`, `min_psnr`/`max_psnr`,
/// [`eval_psnr_histogram`] and [`eval_psnr_percentiles`] would all become
/// meaningless — and, because `NaN` is not finite, would silently read as
/// "every view is a pixel-perfect match" — if it were skipped. Listing it in
/// `metrics` is therefore implied rather than required.
///
/// # NaN convention
///
/// An unselected metric is reported as `f32::NAN` rather than `0.0` so that a
/// "not computed" value can never be mistaken for a real measurement: `0.0`
/// is a legitimate MAE/RMSE/LPIPS reading (a perfect match) and a legitimate
/// SSIM reading (no correlation), whereas `NaN` propagates visibly through
/// any aggregate and serialises to JSON `null`.
///
/// # Errors
///
/// Returns [`EvalError::DimensionMismatch`] or [`EvalError::MetricFailed`]
/// when `pred`/`gt` are not equal-length `width * height * 3` buffers.
pub fn eval_single_view_with_metrics(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
    view_id: &str,
    metrics: &[EvalMetricKind],
) -> Result<ViewEvalResult, EvalError> {
    validate_rgb_pair(pred, gt, width, height, "eval_single_view")?;

    // Unconditional: the ranking key for every aggregate in EvalSuiteResult.
    let psnr = eval_psnr(pred, gt)?;

    let wants = |kind: &EvalMetricKind| metrics.contains(kind);
    let want_ssim = wants(&EvalMetricKind::Ssim);
    let want_ssim_ms = wants(&EvalMetricKind::SsimMs);

    // The full-resolution SSIM components are shared between plain SSIM and
    // MS-SSIM's first scale, so compute them once if either is selected.
    let scale0 = if want_ssim || want_ssim_ms {
        Some(ssim_and_cs(pred, gt, width, height))
    } else {
        None
    };
    let ssim = match scale0 {
        Some((l_mean, cs_mean)) if want_ssim => l_mean * cs_mean,
        _ => f32::NAN,
    };
    let ssim_ms = if want_ssim_ms {
        eval_ssim_ms_impl(pred, gt, width, height, scale0)
    } else {
        f32::NAN
    };
    let lpips_approx = if wants(&EvalMetricKind::LpipsApprox) {
        eval_lpips_approx(pred, gt, width, height)?
    } else {
        f32::NAN
    };
    let mae = if wants(&EvalMetricKind::Mae) {
        eval_mae(pred, gt)?
    } else {
        f32::NAN
    };
    let rmse = if wants(&EvalMetricKind::Rmse) {
        eval_rmse(pred, gt)?
    } else {
        f32::NAN
    };

    Ok(ViewEvalResult {
        view_id: view_id.to_string(),
        psnr,
        ssim,
        lpips_approx,
        mae,
        rmse,
        ssim_ms,
        width,
        height,
        is_worst: false,
        is_best: false,
    })
}
/// Run evaluation on a batch of test items, returning aggregate statistics.
///
/// Per-view evaluation is parallelised with `rayon`: [`std::slice::Iter`]
/// via `par_iter` is an indexed parallel iterator, so
/// `collect::<Result<Vec<_>, _>>()` preserves input order and `per_view`
/// stays aligned with `items`.
///
/// # Metric selection
///
/// Only the metrics named in [`EvalConfig::metrics`] are computed; the rest
/// are reported as `f32::NAN` per view, and their aggregate means
/// (`mean_ssim`, `mean_lpips`, `mean_mae`) are `NaN` in turn. PSNR is always
/// computed because every ranking and distribution statistic in
/// [`EvalSuiteResult`] is defined in terms of it — see
/// [`eval_single_view_with_metrics`] for the full rationale and the `NaN`
/// convention. The default [`EvalConfig`] selects every metric, so callers
/// that do not narrow the list see no change.
///
/// # Errors
///
/// - [`EvalError::EmptyTestSet`] when `items` is empty.
/// - [`EvalError::InvalidConfig`] when `config.metrics` is empty — that would
///   request an evaluation with nothing in it, which is far more likely a
///   caller bug than a deliberate "PSNR only" request (spell that as
///   `vec![EvalMetricKind::Psnr]`).
/// - Whatever [`eval_single_view_with_metrics`] returns for malformed buffers.
pub fn eval_suite(
    items: &[EvalTestItem],
    config: &EvalConfig,
) -> Result<EvalSuiteResult, EvalError> {
    if items.is_empty() {
        return Err(EvalError::EmptyTestSet);
    }
    if config.metrics.is_empty() {
        return Err(EvalError::InvalidConfig(
            "metrics must name at least one EvalMetricKind; use \
             vec![EvalMetricKind::Psnr] for a PSNR-only run"
                .to_string(),
        ));
    }
    let per_view: Vec<ViewEvalResult> = items
        .par_iter()
        .map(|item| {
            eval_single_view_with_metrics(
                &item.pred,
                &item.gt,
                item.width,
                item.height,
                &item.view_id,
                &config.metrics,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let n = per_view.len();
    let mean_psnr = aggregate_mean_psnr(&per_view);
    let mean_ssim: f32 = per_view.iter().map(|v| v.ssim).sum::<f32>() / n as f32;
    let mean_lpips: f32 = per_view.iter().map(|v| v.lpips_approx).sum::<f32>() / n as f32;
    let mean_mae: f32 = per_view.iter().map(|v| v.mae).sum::<f32>() / n as f32;

    // `std_psnr` is computed over only the finite (non-perfect) views, for the
    // same reason as `aggregate_mean_psnr` below: a spread is undefined once
    // an infinity enters it, and folding a fabricated finite stand-in for
    // "infinite" into the statistic would let a single pixel-perfect view
    // dominate it.
    let finite_psnrs: Vec<f32> = per_view
        .iter()
        .map(|v| v.psnr)
        .filter(|p| p.is_finite())
        .collect();

    // `max_psnr` ranges over *every* view, infinities included. A
    // pixel-perfect view's PSNR is genuinely infinite rather than a stand-in
    // for something finite, so reporting the best *imperfect* view instead
    // would understate the maximum and contradict this field's own
    // documentation ("Maximum PSNR across views"). Consumers already expect
    // it: `analyze eval` pipes it through a `finite_or_null` helper because
    // JSON cannot represent infinity.
    let max_psnr = per_view
        .iter()
        .map(|v| v.psnr)
        .fold(f32::NEG_INFINITY, f32::max);

    // `min_psnr` is unaffected by the choice: an infinity is never the
    // minimum unless *every* view is perfect, which the empty-finite branch
    // below reports as an infinite minimum anyway.
    let (min_psnr, std_psnr) = if finite_psnrs.is_empty() {
        // Every view is a pixel-perfect match.
        (f32::INFINITY, 0.0_f32)
    } else {
        let min_psnr = finite_psnrs.iter().cloned().fold(f32::INFINITY, f32::min);
        let variance: f32 = if finite_psnrs.len() > 1 {
            let mu = finite_psnrs.iter().sum::<f32>() / finite_psnrs.len() as f32;
            finite_psnrs
                .iter()
                .map(|p| {
                    let d = p - mu;
                    d * d
                })
                .sum::<f32>()
                / (finite_psnrs.len() - 1) as f32
        } else {
            0.0
        };
        (min_psnr, variance.sqrt())
    };

    let n_worst = config.n_worst_views.min(n);
    let n_best = config.n_best_views.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    // Rank by raw PSNR, not the finite-only subset above (whose length no
    // longer matches `n`): `f32::INFINITY` compares greater than any finite
    // value under `partial_cmp`, so perfect views sort correctly to the top
    // with no magic-number stand-in needed.
    indices.sort_by(|&a, &b| {
        per_view[a]
            .psnr
            .partial_cmp(&per_view[b].psnr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let worst_views: Vec<String> = indices[..n_worst]
        .iter()
        .map(|&i| per_view[i].view_id.clone())
        .collect();
    let best_views: Vec<String> = indices[n - n_best..]
        .iter()
        .rev()
        .map(|&i| per_view[i].view_id.clone())
        .collect();
    let worst_set: std::collections::HashSet<&str> =
        worst_views.iter().map(|s| s.as_str()).collect();
    let best_set: std::collections::HashSet<&str> = best_views.iter().map(|s| s.as_str()).collect();
    let mut per_view_marked = per_view;
    for v in per_view_marked.iter_mut() {
        v.is_worst = worst_set.contains(v.view_id.as_str());
        v.is_best = best_set.contains(v.view_id.as_str());
    }
    Ok(EvalSuiteResult {
        per_view: per_view_marked,
        mean_psnr,
        mean_ssim,
        mean_lpips,
        mean_mae,
        std_psnr,
        min_psnr,
        max_psnr,
        n_views: n,
        worst_views,
        best_views,
    })
}
/// Compute mean PSNR, excluding pixel-perfect (infinite) views from the
/// average so a single perfect match cannot dominate the aggregate via a
/// fabricated finite stand-in. If *every* view is a perfect match the mean
/// is `f32::INFINITY`.
fn aggregate_mean_psnr(views: &[ViewEvalResult]) -> f32 {
    if views.is_empty() {
        return 0.0;
    }
    let finite: Vec<f32> = views
        .iter()
        .map(|v| v.psnr)
        .filter(|p| p.is_finite())
        .collect();
    if finite.is_empty() {
        return f32::INFINITY;
    }
    finite.iter().sum::<f32>() / finite.len() as f32
}
/// Compare two evaluation suite results.
///
/// If the two results have different numbers of views, the per-view comparison
/// is skipped (only aggregate statistics are compared).
///
/// Infinite mean PSNR (every view a pixel-perfect match) is handled by
/// ordinary `f32` arithmetic: `∞ − finite = ∞`. The one case ordinary
/// arithmetic cannot resolve, `∞ − ∞ = NaN` (both sides all-perfect), is
/// special-cased to a `0.0` delta (a tie) rather than propagating `NaN`.
pub fn eval_compare(
    baseline: &EvalSuiteResult,
    candidate: &EvalSuiteResult,
) -> Result<EvalComparison, EvalError> {
    let delta_psnr = if baseline.mean_psnr.is_infinite() && candidate.mean_psnr.is_infinite() {
        0.0_f32
    } else {
        candidate.mean_psnr - baseline.mean_psnr
    };
    let delta_ssim = candidate.mean_ssim - baseline.mean_ssim;
    let delta_lpips = candidate.mean_lpips - baseline.mean_lpips;
    let (n_views_improved, n_views_degraded) = if baseline.n_views == candidate.n_views {
        let mut improved = 0usize;
        let mut degraded = 0usize;
        for (b, c) in baseline.per_view.iter().zip(candidate.per_view.iter()) {
            if c.psnr > b.psnr {
                improved += 1;
            } else if c.psnr < b.psnr {
                degraded += 1;
            }
        }
        (improved, degraded)
    } else {
        (0, 0)
    };
    let is_candidate_better = delta_psnr > 0.0;
    Ok(EvalComparison {
        baseline_mean_psnr: baseline.mean_psnr,
        candidate_mean_psnr: candidate.mean_psnr,
        delta_psnr,
        delta_ssim,
        delta_lpips,
        n_views_improved,
        n_views_degraded,
        is_candidate_better,
    })
}
/// Compute a histogram of PSNR values across the test views.
///
/// Returns `(bin_edges, counts)` where `bin_edges` has `n_bins + 1` entries,
/// derived from the range of the *finite* PSNR values only (a fabricated
/// finite stand-in for "infinite" would otherwise skew the bin range and
/// crowd every real view into a single low bin). Infinite (pixel-perfect)
/// views are placed in the last (best) bin directly, so `counts` still sums
/// to `results.per_view.len()`.
pub fn eval_psnr_histogram(results: &EvalSuiteResult, n_bins: usize) -> (Vec<f32>, Vec<usize>) {
    if n_bins == 0 || results.per_view.is_empty() {
        return (vec![], vec![]);
    }
    let finite_psnrs: Vec<f32> = results
        .per_view
        .iter()
        .map(|v| v.psnr)
        .filter(|p| p.is_finite())
        .collect();
    let (min_val, max_val) = if finite_psnrs.is_empty() {
        // Every view is a pixel-perfect match: there is no finite range to
        // bin against, so every view collapses into bin 0 via `range==0.0`
        // below (the loop still routes infinities to the last bin).
        (0.0_f32, 0.0_f32)
    } else {
        (
            finite_psnrs.iter().cloned().fold(f32::INFINITY, f32::min),
            finite_psnrs
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
        )
    };
    let mut edges = vec![0.0_f32; n_bins + 1];
    let range = max_val - min_val;
    for (i, edge_val) in edges.iter_mut().enumerate().take(n_bins + 1) {
        *edge_val = min_val + (i as f32 / n_bins as f32) * range;
    }
    let mut counts = vec![0usize; n_bins];
    for v in &results.per_view {
        let idx = if v.psnr.is_infinite() {
            n_bins - 1
        } else if range == 0.0 {
            0
        } else {
            let raw = ((v.psnr - min_val) / range * n_bins as f32) as usize;
            raw.min(n_bins - 1)
        };
        counts[idx] += 1;
    }
    (edges, counts)
}
/// Compute percentiles [P5, P25, P50, P75, P95] of the PSNR distribution.
///
/// Infinite (pixel-perfect) PSNR values sort correctly to the top of the
/// distribution under ordinary `f32` comparison — no finite stand-in value
/// is needed.
pub fn eval_psnr_percentiles(results: &EvalSuiteResult) -> [f32; 5] {
    let mut psnrs: Vec<f32> = results.per_view.iter().map(|v| v.psnr).collect();
    if psnrs.is_empty() {
        return [0.0; 5];
    }
    psnrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = psnrs.len();
    let percentile = |p: f32| -> f32 {
        let idx = (p / 100.0 * n as f32) as usize;
        psnrs[idx.min(n - 1)]
    };
    [
        percentile(5.0),
        percentile(25.0),
        percentile(50.0),
        percentile(75.0),
        percentile(95.0),
    ]
}
/// Render a PSNR value for human-readable output, spelling a pixel-perfect
/// match as `∞` rather than letting `{:.2}` print `inf`.
fn format_psnr(value: f32) -> String {
    if value.is_infinite() {
        "∞".to_string()
    } else {
        format!("{:.2}", value)
    }
}

/// Format a single view result as a human-readable string.
pub fn eval_format_view_result(result: &ViewEvalResult) -> String {
    let psnr_str = format_psnr(result.psnr);
    let flags = match (result.is_best, result.is_worst) {
        (true, _) => " [BEST]",
        (_, true) => " [WORST]",
        _ => "",
    };
    format!(
        "View {:30} | PSNR: {:>8} dB | SSIM: {:.4} | LPIPS: {:.4} | MAE: {:.4} | RMSE: {:.4} | MS-SSIM: {:.4}{}",
        result.view_id, psnr_str, result.ssim, result.lpips_approx, result.mae, result
        .rmse, result.ssim_ms, flags,
    )
}
/// Format the aggregate suite result as a multi-line report string.
pub fn eval_format_suite_result(result: &EvalSuiteResult) -> String {
    let mut lines = vec![
        "=== Evaluation Suite Results ===".to_string(),
        format!("Views evaluated : {}", result.n_views),
        format!(
            "Mean PSNR       : {} dB  (σ={:.2}, min={}, max={})",
            format_psnr(result.mean_psnr),
            result.std_psnr,
            format_psnr(result.min_psnr),
            format_psnr(result.max_psnr),
        ),
        format!("Mean SSIM       : {:.4}", result.mean_ssim),
        format!("Mean LPIPS      : {:.4}", result.mean_lpips),
        format!("Mean MAE        : {:.4}", result.mean_mae),
    ];
    if !result.worst_views.is_empty() {
        lines.push(format!(
            "Worst views     : {}",
            result.worst_views.join(", ")
        ));
    }
    if !result.best_views.is_empty() {
        lines.push(format!(
            "Best views      : {}",
            result.best_views.join(", ")
        ));
    }
    lines.join("\n")
}
/// Format a model comparison as a human-readable string.
pub fn eval_format_comparison(comparison: &EvalComparison) -> String {
    let sign = |v: f32| if v >= 0.0 { "+" } else { "" };
    let verdict = if comparison.is_candidate_better {
        "CANDIDATE IS BETTER"
    } else if comparison.delta_psnr == 0.0 {
        "TIE"
    } else {
        "BASELINE IS BETTER"
    };
    format!(
        "=== Model Comparison ===\n\
         Baseline PSNR  : {:.2} dB\n\
         Candidate PSNR : {:.2} dB\n\
         ΔPSNR          : {}{:.2} dB\n\
         ΔSSIM          : {}{:.4}\n\
         ΔLPIPS         : {}{:.4}  (negative = better)\n\
         Views improved : {}\n\
         Views degraded : {}\n\
         Verdict        : {}",
        comparison.baseline_mean_psnr,
        comparison.candidate_mean_psnr,
        sign(comparison.delta_psnr),
        comparison.delta_psnr,
        sign(comparison.delta_ssim),
        comparison.delta_ssim,
        sign(comparison.delta_lpips),
        comparison.delta_lpips,
        comparison.n_views_improved,
        comparison.n_views_degraded,
        verdict,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssim_uniform_images_match_closed_form_regardless_of_size() {
        // For two spatially-uniform images the SSIM window average has a
        // closed form: l = (2·v1·v2 + C1) / (v1² + v2² + C1), cs = 1 (zero
        // variance everywhere, since the image never varies). Because the
        // border ring is now cropped out of the average instead of being
        // zero-padded into it, this closed form must hold *exactly* for any
        // size >= 11x11 — a border-biased implementation would instead give
        // a result that drifts with image size, since a smaller image has
        // proportionally more border pixels pulled toward zero.
        const C1: f32 = 0.0001;
        let v1 = 0.2_f32;
        let v2 = 0.6_f32;
        let expected = (2.0 * v1 * v2 + C1) / (v1 * v1 + v2 * v2 + C1);
        for &size in &[11usize, 16, 21, 40] {
            let pred = vec![v1; size * size * 3];
            let gt = vec![v2; size * size * 3];
            let ssim = eval_ssim(&pred, &gt, size, size).expect("ssim ok");
            assert!(
                (ssim - expected).abs() < 1e-4,
                "size {size}: expected {expected}, got {ssim}"
            );
        }
    }

    #[test]
    fn test_ssim_tiny_image_still_defined() {
        // Below the 11×11 window, SSIM falls back to the full padded frame
        // rather than erroring, so tiny images (thumbnails, tests) remain
        // usable.
        let pred = vec![0.3_f32; 4 * 4 * 3];
        let gt = vec![0.7_f32; 4 * 4 * 3];
        let ssim = eval_ssim(&pred, &gt, 4, 4).expect("ssim ok");
        assert!((-1.0..=1.0).contains(&ssim));
    }

    #[test]
    fn test_eval_single_view_ssim_ms_matches_direct_call() {
        // eval_single_view reuses its own SSIM computation as MS-SSIM's
        // first scale instead of recomputing it; the result must be
        // identical to calling eval_ssim_ms directly.
        let pred: Vec<f32> = (0..48 * 48 * 3).map(|i| (i % 7) as f32 / 7.0).collect();
        let gt: Vec<f32> = (0..48 * 48 * 3).map(|i| (i % 5) as f32 / 5.0).collect();
        let direct = eval_ssim_ms(&pred, &gt, 48, 48).expect("ms-ssim ok");
        let via_single_view = eval_single_view(&pred, &gt, 48, 48, "v").expect("ok");
        assert!(
            (direct - via_single_view.ssim_ms).abs() < 1e-5,
            "direct={direct}, via_single_view={}",
            via_single_view.ssim_ms
        );
    }

    #[test]
    fn test_ssim_ms_adaptive_scale_count_stays_bounded() {
        // A 5-scale image (>=176px) and a 3-scale image (48px) must both
        // produce a well-defined value in the valid SSIM-like range; the
        // AM-GM bound `cs<=1, l<=1` on non-negative clamped terms keeps the
        // multiplicative product from ever exceeding 1.0.
        for &size in &[48usize, 176] {
            let pred: Vec<f32> = (0..size * size * 3)
                .map(|i| (i % 11) as f32 / 11.0)
                .collect();
            let gt: Vec<f32> = (0..size * size * 3)
                .map(|i| (i % 13) as f32 / 13.0)
                .collect();
            let ssim_ms = eval_ssim_ms(&pred, &gt, size, size).expect("ms-ssim ok");
            assert!(
                ssim_ms.is_finite() && ssim_ms <= 1.0001,
                "size {size}: ssim_ms={ssim_ms}"
            );
        }
    }

    #[test]
    fn test_eval_convolve_short_image_does_not_panic() {
        // A caller-supplied image shorter than width*height must not
        // panic; out-of-bounds samples are treated as 0.0.
        let short_image = vec![0.5_f32; 4];
        let k = eval_gaussian_kernel_11();
        let out = eval_convolve(&short_image, 6, 4, &k, 11);
        assert_eq!(out.len(), 24);
    }

    #[test]
    fn test_eval_sobel_short_image_does_not_panic() {
        let short_image = vec![0.5_f32; 3];
        let mag = eval_sobel(&short_image, 8, 8);
        assert_eq!(mag.len(), 64);
    }

    #[test]
    fn test_aggregate_mean_psnr_excludes_perfect_view_from_mean() {
        let cfg = EvalConfig::default();
        // One perfect (identical) view and one clearly-imperfect view: the
        // mean must reflect only the imperfect view's real PSNR, not a
        // 999.0 stand-in for the perfect one dragging the average up.
        let identical = vec![0.5_f32; 16 * 16 * 3];
        let pred_bad = vec![0.0_f32; 16 * 16 * 3];
        let gt_bad = vec![0.1_f32; 16 * 16 * 3];
        let items = vec![
            EvalTestItem {
                view_id: "perfect".to_string(),
                pred: identical.clone(),
                gt: identical,
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "imperfect".to_string(),
                pred: pred_bad.clone(),
                gt: gt_bad.clone(),
                width: 16,
                height: 16,
            },
        ];
        let result = eval_suite(&items, &cfg).expect("ok");
        let expected = eval_psnr(&pred_bad, &gt_bad).expect("psnr ok");
        assert!(
            (result.mean_psnr - expected).abs() < 0.01,
            "mean_psnr should equal the sole finite view's PSNR ({expected}), got {}",
            result.mean_psnr
        );
        assert!(
            result.max_psnr.is_infinite(),
            "max_psnr should reflect the perfect view"
        );
        assert!((result.min_psnr - expected).abs() < 0.01);
    }

    #[test]
    fn test_eval_compare_handles_one_sided_infinite_mean() {
        let cfg = EvalConfig::default();
        let identical = vec![0.5_f32; 16 * 16 * 3];
        let perfect_item = EvalTestItem {
            view_id: "v".to_string(),
            pred: identical.clone(),
            gt: identical,
            width: 16,
            height: 16,
        };
        let imperfect_item = EvalTestItem {
            view_id: "v".to_string(),
            pred: vec![0.0_f32; 16 * 16 * 3],
            gt: vec![0.1_f32; 16 * 16 * 3],
            width: 16,
            height: 16,
        };
        let perfect_result = eval_suite(&[perfect_item], &cfg).expect("ok");
        let imperfect_result = eval_suite(&[imperfect_item], &cfg).expect("ok");
        let cmp = eval_compare(&imperfect_result, &perfect_result).expect("compare ok");
        assert!(
            cmp.delta_psnr.is_infinite() && cmp.delta_psnr > 0.0,
            "delta should be +infinity, got {}",
            cmp.delta_psnr
        );
        assert!(cmp.is_candidate_better);

        // Both sides all-perfect must resolve to a tie (0.0), not NaN.
        let tie = eval_compare(&perfect_result_clone(&cfg), &perfect_result_clone(&cfg))
            .expect("compare ok");
        assert_eq!(tie.delta_psnr, 0.0);
        assert!(!tie.delta_psnr.is_nan());
    }

    /// Helper for `test_eval_compare_handles_one_sided_infinite_mean`: build
    /// a fresh all-perfect single-view suite result.
    fn perfect_result_clone(cfg: &EvalConfig) -> EvalSuiteResult {
        let identical = vec![0.5_f32; 16 * 16 * 3];
        let item = EvalTestItem {
            view_id: "v".to_string(),
            pred: identical.clone(),
            gt: identical,
            width: 16,
            height: 16,
        };
        eval_suite(&[item], cfg).expect("ok")
    }

    // -----------------------------------------------------------------------
    // max_psnr spans every view (regression: it was folded over the finite
    // subset only, so one pixel-perfect view among imperfect ones reported
    // the best *imperfect* PSNR as the maximum).
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_psnr_is_infinite_when_any_view_is_perfect() {
        let cfg = EvalConfig::default();
        let identical = vec![0.5_f32; 16 * 16 * 3];
        let items = vec![
            EvalTestItem {
                view_id: "perfect".to_string(),
                pred: identical.clone(),
                gt: identical,
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "imperfect".to_string(),
                pred: vec![0.0_f32; 16 * 16 * 3],
                gt: vec![0.1_f32; 16 * 16 * 3],
                width: 16,
                height: 16,
            },
        ];
        let result = eval_suite(&items, &cfg).expect("ok");
        assert!(
            result.max_psnr.is_infinite() && result.max_psnr > 0.0,
            "max_psnr must span every view, got {}",
            result.max_psnr
        );
        // min and std still exclude the infinity: the minimum is the one
        // imperfect view, and a spread over an infinity is undefined.
        assert!(
            result.min_psnr.is_finite(),
            "min_psnr must stay finite while an imperfect view exists, got {}",
            result.min_psnr
        );
        assert!(
            result.std_psnr.is_finite() && !result.std_psnr.is_nan(),
            "std_psnr must stay finite, got {}",
            result.std_psnr
        );
    }

    #[test]
    fn test_max_psnr_is_ordinary_maximum_without_perfect_views() {
        let cfg = EvalConfig::default();
        let items = vec![
            EvalTestItem {
                view_id: "closer".to_string(),
                pred: vec![0.0_f32; 16 * 16 * 3],
                gt: vec![0.05_f32; 16 * 16 * 3],
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "further".to_string(),
                pred: vec![0.0_f32; 16 * 16 * 3],
                gt: vec![0.5_f32; 16 * 16 * 3],
                width: 16,
                height: 16,
            },
        ];
        let result = eval_suite(&items, &cfg).expect("ok");
        let psnrs: Vec<f32> = result.per_view.iter().map(|v| v.psnr).collect();
        let expected_max = psnrs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let expected_min = psnrs.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!((result.max_psnr - expected_max).abs() < 1e-3);
        assert!((result.min_psnr - expected_min).abs() < 1e-3);
        assert!(result.max_psnr > result.min_psnr);
    }

    #[test]
    fn test_all_perfect_views_report_infinite_min_and_max() {
        let cfg = EvalConfig::default();
        let identical = vec![0.5_f32; 16 * 16 * 3];
        let items = vec![EvalTestItem {
            view_id: "v".to_string(),
            pred: identical.clone(),
            gt: identical,
            width: 16,
            height: 16,
        }];
        let result = eval_suite(&items, &cfg).expect("ok");
        assert!(result.min_psnr.is_infinite());
        assert!(result.max_psnr.is_infinite());
        assert_eq!(result.std_psnr, 0.0);
    }

    #[test]
    fn test_format_suite_result_spells_infinite_psnr_as_symbol() {
        // `{:.2}` would print "inf" for a pixel-perfect maximum; the report
        // uses the same ∞ spelling it already used for the mean.
        let cfg = EvalConfig::default();
        let identical = vec![0.5_f32; 16 * 16 * 3];
        let items = vec![
            EvalTestItem {
                view_id: "perfect".to_string(),
                pred: identical.clone(),
                gt: identical,
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "imperfect".to_string(),
                pred: vec![0.0_f32; 16 * 16 * 3],
                gt: vec![0.1_f32; 16 * 16 * 3],
                width: 16,
                height: 16,
            },
        ];
        let result = eval_suite(&items, &cfg).expect("ok");
        let text = eval_format_suite_result(&result);
        assert!(
            text.contains("max=∞"),
            "infinite maximum should render as ∞, got:\n{text}"
        );
        assert!(
            !text.contains("inf"),
            "raw `inf` must not leak into the report, got:\n{text}"
        );
    }

    // -----------------------------------------------------------------------
    // EvalConfig::metrics is honoured (regression: eval_suite used to compute
    // every metric unconditionally, making the field inert).
    // -----------------------------------------------------------------------

    /// Two 16×16 views with differing content, enough for real metric values.
    fn two_item_set() -> Vec<EvalTestItem> {
        vec![
            EvalTestItem {
                view_id: "a".to_string(),
                pred: (0..16 * 16 * 3).map(|i| (i % 7) as f32 / 7.0).collect(),
                gt: (0..16 * 16 * 3).map(|i| (i % 5) as f32 / 5.0).collect(),
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "b".to_string(),
                pred: vec![0.0_f32; 16 * 16 * 3],
                gt: vec![0.25_f32; 16 * 16 * 3],
                width: 16,
                height: 16,
            },
        ]
    }

    #[test]
    fn test_eval_suite_skips_unselected_metrics() {
        let cfg = EvalConfig {
            metrics: vec![EvalMetricKind::Psnr, EvalMetricKind::Mae],
            ..EvalConfig::default()
        };
        let result = eval_suite(&two_item_set(), &cfg).expect("ok");
        for view in &result.per_view {
            assert!(
                view.psnr.is_finite(),
                "PSNR is unconditional, got {} for {}",
                view.psnr,
                view.view_id
            );
            assert!(
                view.mae.is_finite(),
                "MAE was selected, got {} for {}",
                view.mae,
                view.view_id
            );
            assert!(
                view.ssim.is_nan(),
                "unselected SSIM must be NaN, got {}",
                view.ssim
            );
            assert!(
                view.ssim_ms.is_nan(),
                "unselected MS-SSIM must be NaN, got {}",
                view.ssim_ms
            );
            assert!(
                view.lpips_approx.is_nan(),
                "unselected LPIPS must be NaN, got {}",
                view.lpips_approx
            );
            assert!(
                view.rmse.is_nan(),
                "unselected RMSE must be NaN, got {}",
                view.rmse
            );
        }
        // The aggregate of an unselected metric is NaN in turn, which is what
        // makes "not computed" impossible to mistake for a measurement.
        assert!(result.mean_ssim.is_nan());
        assert!(result.mean_lpips.is_nan());
        assert!(result.mean_mae.is_finite());
    }

    #[test]
    fn test_eval_suite_default_config_computes_every_metric() {
        // The default selects everything, so existing callers are unaffected.
        let result = eval_suite(&two_item_set(), &EvalConfig::default()).expect("ok");
        for view in &result.per_view {
            for (name, value) in [
                ("psnr", view.psnr),
                ("ssim", view.ssim),
                ("lpips", view.lpips_approx),
                ("mae", view.mae),
                ("rmse", view.rmse),
                ("ssim_ms", view.ssim_ms),
            ] {
                assert!(
                    !value.is_nan(),
                    "{name} must be computed under the default config, got NaN"
                );
            }
        }
    }

    #[test]
    fn test_selected_metrics_match_the_unfiltered_values() {
        // Gating must skip work, never change a selected metric's value.
        let items = two_item_set();
        let full = eval_suite(&items, &EvalConfig::default()).expect("ok");
        let cfg = EvalConfig {
            metrics: vec![EvalMetricKind::Ssim, EvalMetricKind::Rmse],
            ..EvalConfig::default()
        };
        let narrowed = eval_suite(&items, &cfg).expect("ok");
        for (a, b) in full.per_view.iter().zip(narrowed.per_view.iter()) {
            assert_eq!(a.view_id, b.view_id);
            assert!(
                (a.ssim - b.ssim).abs() < 1e-6,
                "SSIM changed: {} vs {}",
                a.ssim,
                b.ssim
            );
            assert!(
                (a.rmse - b.rmse).abs() < 1e-6,
                "RMSE changed: {} vs {}",
                a.rmse,
                b.rmse
            );
            assert!(
                (a.psnr - b.psnr).abs() < 1e-6,
                "PSNR changed: {} vs {}",
                a.psnr,
                b.psnr
            );
        }
    }

    #[test]
    fn test_ssim_ms_alone_still_reuses_scale_zero_correctly() {
        // Selecting MS-SSIM without plain SSIM must still produce the same
        // MS-SSIM value: the shared scale-0 components are computed for
        // either selection, but only SSIM's own field is left NaN.
        let items = two_item_set();
        let cfg = EvalConfig {
            metrics: vec![EvalMetricKind::SsimMs],
            ..EvalConfig::default()
        };
        let narrowed = eval_suite(&items, &cfg).expect("ok");
        let full = eval_suite(&items, &EvalConfig::default()).expect("ok");
        for (a, b) in full.per_view.iter().zip(narrowed.per_view.iter()) {
            assert!(b.ssim.is_nan(), "SSIM was not selected");
            assert!(
                (a.ssim_ms - b.ssim_ms).abs() < 1e-6,
                "MS-SSIM changed: {} vs {}",
                a.ssim_ms,
                b.ssim_ms
            );
        }
    }

    #[test]
    fn test_eval_suite_rejects_empty_metrics() {
        let cfg = EvalConfig {
            metrics: Vec::new(),
            ..EvalConfig::default()
        };
        let result = eval_suite(&two_item_set(), &cfg);
        assert!(
            matches!(
                &result,
                Err(EvalError::InvalidConfig(msg)) if msg.contains("metrics")
            ),
            "expected InvalidConfig mentioning 'metrics', got {:?}",
            result.map(|r| r.n_views)
        );
    }

    #[test]
    fn test_empty_test_set_outranks_empty_metrics() {
        // An empty item list is reported as EmptyTestSet even when the config
        // is also invalid, so the caller sees the more actionable error.
        let cfg = EvalConfig {
            metrics: Vec::new(),
            ..EvalConfig::default()
        };
        assert!(matches!(
            eval_suite(&[], &cfg),
            Err(EvalError::EmptyTestSet)
        ));
    }

    #[test]
    fn test_eval_single_view_with_metrics_psnr_survives_omission() {
        // PSNR is documented as unconditional; omitting it from the list must
        // not yield NaN, which `is_finite()` filters would misread as
        // "pixel-perfect" throughout the aggregates.
        let pred: Vec<f32> = (0..16 * 16 * 3).map(|i| (i % 7) as f32 / 7.0).collect();
        let gt: Vec<f32> = (0..16 * 16 * 3).map(|i| (i % 5) as f32 / 5.0).collect();
        let view = eval_single_view_with_metrics(
            &pred,
            &gt,
            16,
            16,
            "v",
            &[EvalMetricKind::Mae, EvalMetricKind::Rmse],
        )
        .expect("ok");
        let expected = eval_psnr(&pred, &gt).expect("psnr ok");
        assert!(
            (view.psnr - expected).abs() < 1e-6,
            "PSNR must be computed regardless of selection, got {}",
            view.psnr
        );
    }

    #[test]
    fn test_eval_single_view_matches_explicit_full_selection() {
        let pred: Vec<f32> = (0..24 * 24 * 3).map(|i| (i % 9) as f32 / 9.0).collect();
        let gt: Vec<f32> = (0..24 * 24 * 3).map(|i| (i % 4) as f32 / 4.0).collect();
        let implicit = eval_single_view(&pred, &gt, 24, 24, "v").expect("ok");
        let explicit = eval_single_view_with_metrics(
            &pred,
            &gt,
            24,
            24,
            "v",
            &[
                EvalMetricKind::Psnr,
                EvalMetricKind::Ssim,
                EvalMetricKind::LpipsApprox,
                EvalMetricKind::Mae,
                EvalMetricKind::Rmse,
                EvalMetricKind::SsimMs,
            ],
        )
        .expect("ok");
        assert!((implicit.psnr - explicit.psnr).abs() < 1e-6);
        assert!((implicit.ssim - explicit.ssim).abs() < 1e-6);
        assert!((implicit.ssim_ms - explicit.ssim_ms).abs() < 1e-6);
        assert!((implicit.lpips_approx - explicit.lpips_approx).abs() < 1e-6);
        assert!((implicit.mae - explicit.mae).abs() < 1e-6);
        assert!((implicit.rmse - explicit.rmse).abs() < 1e-6);
    }

    #[test]
    fn test_psnr_histogram_places_infinite_psnr_in_last_bin() {
        let cfg = EvalConfig::default();
        let identical = vec![0.5_f32; 16 * 16 * 3];
        let items = vec![
            EvalTestItem {
                view_id: "perfect".to_string(),
                pred: identical.clone(),
                gt: identical,
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "mid1".to_string(),
                pred: vec![0.0_f32; 16 * 16 * 3],
                gt: vec![0.1_f32; 16 * 16 * 3],
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "mid2".to_string(),
                pred: vec![0.0_f32; 16 * 16 * 3],
                gt: vec![0.2_f32; 16 * 16 * 3],
                width: 16,
                height: 16,
            },
        ];
        let result = eval_suite(&items, &cfg).expect("ok");
        let (_, counts) = eval_psnr_histogram(&result, 4);
        assert_eq!(
            counts.iter().sum::<usize>(),
            3,
            "counts must sum to n_views even with an infinite view"
        );
        assert!(
            counts[3] >= 1,
            "the perfect view must land in the last (best) bin, got counts={counts:?}"
        );
    }
}
