//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    EvalComparison, EvalConfig, EvalError, EvalSuiteResult, EvalTestItem, ViewEvalResult,
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
                    acc += kernel[ki] * image[pi];
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
pub fn eval_sobel(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let mut gray = vec![0.0_f32; n];
    for i in 0..n {
        let r = image[i * 3];
        let g = image[i * 3 + 1];
        let b = image[i * 3 + 2];
        gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
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
/// Internal: compute SSIM on a single-channel image of length `width × height`.
fn ssim_single_channel(chan: &[f32], ref_chan: &[f32], width: usize, height: usize) -> f32 {
    let k11 = eval_gaussian_kernel_11();
    let mu_x = eval_convolve(chan, width, height, &k11, 11);
    let mu_y = eval_convolve(ref_chan, width, height, &k11, 11);
    let n = width * height;
    let mut xx = vec![0.0_f32; n];
    let mut yy = vec![0.0_f32; n];
    let mut xy = vec![0.0_f32; n];
    for i in 0..n {
        xx[i] = chan[i] * chan[i];
        yy[i] = ref_chan[i] * ref_chan[i];
        xy[i] = chan[i] * ref_chan[i];
    }
    let mu_xx = eval_convolve(&xx, width, height, &k11, 11);
    let mu_yy = eval_convolve(&yy, width, height, &k11, 11);
    let mu_xy = eval_convolve(&xy, width, height, &k11, 11);
    const C1: f32 = 0.0001;
    const C2: f32 = 0.0009;
    let mut ssim_sum = 0.0_f64;
    for i in 0..n {
        let mux = mu_x[i];
        let muy = mu_y[i];
        let sig_x = mu_xx[i] - mux * mux;
        let sig_y = mu_yy[i] - muy * muy;
        let sig_xy = mu_xy[i] - mux * muy;
        let num = (2.0 * mux * muy + C1) * (2.0 * sig_xy + C2);
        let den = (mux * mux + muy * muy + C1) * (sig_x + sig_y + C2);
        ssim_sum += (num / den) as f64;
    }
    (ssim_sum / n as f64) as f32
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
/// Both `pred` and `gt` must have length `width * height * 3`.
pub fn eval_ssim(pred: &[f32], gt: &[f32], width: usize, height: usize) -> Result<f32, EvalError> {
    if pred.len() != gt.len() {
        return Err(EvalError::DimensionMismatch {
            pred: pred.len(),
            gt: gt.len(),
        });
    }
    let expected = width * height * 3;
    if pred.len() != expected {
        return Err(EvalError::MetricFailed(format!(
            "SSIM: buffer length {} does not match {}×{}×3={}",
            pred.len(),
            width,
            height,
            expected
        )));
    }
    if pred.is_empty() {
        return Err(EvalError::MetricFailed("SSIM: empty image".to_string()));
    }
    let n = width * height;
    let mut ssim_total = 0.0_f32;
    for c in 0..3 {
        let mut pred_c = vec![0.0_f32; n];
        let mut gt_c = vec![0.0_f32; n];
        for i in 0..n {
            pred_c[i] = pred[i * 3 + c];
            gt_c[i] = gt[i * 3 + c];
        }
        ssim_total += ssim_single_channel(&pred_c, &gt_c, width, height);
    }
    Ok(ssim_total / 3.0)
}
/// Multi-scale SSIM with 3 scales.
///
/// Weights before normalisation: `[0.0448, 0.2856, 0.3001]`.
/// Images are progressively downsampled by 2× between scales.
pub fn eval_ssim_ms(
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
            "MS-SSIM: buffer length {} does not match {}×{}×3={}",
            pred.len(),
            width,
            height,
            expected
        )));
    }
    if pred.is_empty() {
        return Err(EvalError::MetricFailed("MS-SSIM: empty image".to_string()));
    }
    const RAW_WEIGHTS: [f32; 3] = [0.0448, 0.2856, 0.3001];
    let weight_sum: f32 = RAW_WEIGHTS.iter().sum();
    let weights: [f32; 3] = [
        RAW_WEIGHTS[0] / weight_sum,
        RAW_WEIGHTS[1] / weight_sum,
        RAW_WEIGHTS[2] / weight_sum,
    ];
    let mut p_cur = pred.to_vec();
    let mut g_cur = gt.to_vec();
    let mut w_cur = width;
    let mut h_cur = height;
    let mut ms_ssim = 0.0_f32;
    for (scale, &weight) in weights.iter().enumerate() {
        let s = eval_ssim(&p_cur, &g_cur, w_cur, h_cur)?;
        ms_ssim += weight * s;
        if scale < 2 {
            let (pd, pw, ph) = eval_downsample_2x(&p_cur, w_cur, h_cur);
            let (gd, gw, gh) = eval_downsample_2x(&g_cur, w_cur, h_cur);
            if pw != gw || ph != gh {
                return Err(EvalError::MetricFailed(
                    "MS-SSIM: downsampled dimensions mismatch".to_string(),
                ));
            }
            p_cur = pd;
            g_cur = gd;
            w_cur = pw;
            h_cur = ph;
        }
    }
    Ok(ms_ssim)
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
/// Compute all metrics for a single view pair.
pub fn eval_single_view(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
    view_id: &str,
) -> Result<ViewEvalResult, EvalError> {
    if pred.len() != gt.len() {
        return Err(EvalError::DimensionMismatch {
            pred: pred.len(),
            gt: gt.len(),
        });
    }
    let psnr = eval_psnr(pred, gt)?;
    let ssim = eval_ssim(pred, gt, width, height)?;
    let lpips_approx = eval_lpips_approx(pred, gt, width, height)?;
    let mae = eval_mae(pred, gt)?;
    let rmse = eval_rmse(pred, gt)?;
    let ssim_ms = eval_ssim_ms(pred, gt, width, height)?;
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
pub fn eval_suite(
    items: &[EvalTestItem],
    config: &EvalConfig,
) -> Result<EvalSuiteResult, EvalError> {
    if items.is_empty() {
        return Err(EvalError::EmptyTestSet);
    }
    let mut per_view: Vec<ViewEvalResult> = Vec::with_capacity(items.len());
    for item in items {
        let result =
            eval_single_view(&item.pred, &item.gt, item.width, item.height, &item.view_id)?;
        per_view.push(result);
    }
    let n = per_view.len();
    let mean_psnr = aggregate_mean_psnr(&per_view);
    let mean_ssim: f32 = per_view.iter().map(|v| v.ssim).sum::<f32>() / n as f32;
    let mean_lpips: f32 = per_view.iter().map(|v| v.lpips_approx).sum::<f32>() / n as f32;
    let mean_mae: f32 = per_view.iter().map(|v| v.mae).sum::<f32>() / n as f32;
    let finite_psnrs: Vec<f32> = per_view
        .iter()
        .map(|v| {
            if v.psnr.is_infinite() {
                999.0_f32
            } else {
                v.psnr
            }
        })
        .collect();
    let min_psnr = finite_psnrs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_psnr = finite_psnrs
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let variance: f32 = if n > 1 {
        let mu = finite_psnrs.iter().sum::<f32>() / n as f32;
        finite_psnrs
            .iter()
            .map(|p| {
                let d = p - mu;
                d * d
            })
            .sum::<f32>()
            / (n - 1) as f32
    } else {
        0.0
    };
    let std_psnr = variance.sqrt();
    let n_worst = config.n_worst_views.min(n);
    let n_best = config.n_best_views.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        finite_psnrs[a]
            .partial_cmp(&finite_psnrs[b])
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
/// Compute mean PSNR, handling infinite values (identical images) correctly.
fn aggregate_mean_psnr(views: &[ViewEvalResult]) -> f32 {
    if views.is_empty() {
        return 0.0;
    }
    if views.iter().all(|v| v.psnr.is_infinite()) {
        return f32::INFINITY;
    }
    let sum: f32 = views
        .iter()
        .map(|v| {
            if v.psnr.is_infinite() {
                999.0_f32
            } else {
                v.psnr
            }
        })
        .sum();
    sum / views.len() as f32
}
/// Compare two evaluation suite results.
///
/// If the two results have different numbers of views, the per-view comparison
/// is skipped (only aggregate statistics are compared).
pub fn eval_compare(
    baseline: &EvalSuiteResult,
    candidate: &EvalSuiteResult,
) -> Result<EvalComparison, EvalError> {
    let norm_psnr = |v: f32| if v.is_infinite() { 999.0_f32 } else { v };
    let delta_psnr = norm_psnr(candidate.mean_psnr) - norm_psnr(baseline.mean_psnr);
    let delta_ssim = candidate.mean_ssim - baseline.mean_ssim;
    let delta_lpips = candidate.mean_lpips - baseline.mean_lpips;
    let (n_views_improved, n_views_degraded) = if baseline.n_views == candidate.n_views {
        let mut improved = 0usize;
        let mut degraded = 0usize;
        for (b, c) in baseline.per_view.iter().zip(candidate.per_view.iter()) {
            let bp = norm_psnr(b.psnr);
            let cp = norm_psnr(c.psnr);
            if cp > bp {
                improved += 1;
            } else if cp < bp {
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
/// Returns `(bin_edges, counts)` where `bin_edges` has `n_bins + 1` entries.
/// Infinite PSNR values are mapped to the maximum finite bin.
pub fn eval_psnr_histogram(results: &EvalSuiteResult, n_bins: usize) -> (Vec<f32>, Vec<usize>) {
    if n_bins == 0 || results.per_view.is_empty() {
        return (vec![], vec![]);
    }
    let psnrs: Vec<f32> = results
        .per_view
        .iter()
        .map(|v| {
            if v.psnr.is_infinite() {
                999.0_f32
            } else {
                v.psnr
            }
        })
        .collect();
    let min_val = psnrs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = psnrs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut edges = vec![0.0_f32; n_bins + 1];
    let range = max_val - min_val;
    for (i, edge_val) in edges.iter_mut().enumerate().take(n_bins + 1) {
        *edge_val = min_val + (i as f32 / n_bins as f32) * range;
    }
    let mut counts = vec![0usize; n_bins];
    for &p in &psnrs {
        let idx = if range == 0.0 {
            0
        } else {
            let raw = ((p - min_val) / range * n_bins as f32) as usize;
            raw.min(n_bins - 1)
        };
        counts[idx] += 1;
    }
    (edges, counts)
}
/// Compute percentiles [P5, P25, P50, P75, P95] of the PSNR distribution.
///
/// Infinite PSNR values are treated as 999.0 for ordering purposes.
pub fn eval_psnr_percentiles(results: &EvalSuiteResult) -> [f32; 5] {
    let mut psnrs: Vec<f32> = results
        .per_view
        .iter()
        .map(|v| {
            if v.psnr.is_infinite() {
                999.0_f32
            } else {
                v.psnr
            }
        })
        .collect();
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
/// Format a single view result as a human-readable string.
pub fn eval_format_view_result(result: &ViewEvalResult) -> String {
    let psnr_str = if result.psnr.is_infinite() {
        "∞".to_string()
    } else {
        format!("{:.2}", result.psnr)
    };
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
    let mean_psnr_str = if result.mean_psnr.is_infinite() {
        "∞".to_string()
    } else {
        format!("{:.2}", result.mean_psnr)
    };
    let mut lines = vec![
        "=== Evaluation Suite Results ===".to_string(),
        format!("Views evaluated : {}", result.n_views),
        format!(
            "Mean PSNR       : {} dB  (σ={:.2}, min={:.2}, max={:.2})",
            mean_psnr_str, result.std_psnr, result.min_psnr, result.max_psnr
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
