//! Image-space gradients of the photometric objective.
//!
//! [`crate::trainer::Trainer`] needs `∂L/∂rendered` — the derivative of exactly
//! the scalar loss [`crate::loss::LossComputer`] reports — to feed the
//! rasterizer's backward pass.  This module holds that math, kept apart from the
//! training loop so the forward loss (in [`crate::loss`]) and its adjoint stay
//! side by side and independently testable.
//!
//! The entry point is [`photometric_pixel_gradient`]; everything else is the
//! SSIM / MS-SSIM adjoint machinery it is built from:
//!
//! | forward term ([`crate::loss`]) | adjoint here |
//! |---|---|
//! | [`crate::loss::l1_loss`] | inline sub-gradient ([`sign_or_zero`]) |
//! | [`crate::loss::ssim_loss`] | [`ssim_pixel_gradient`] |
//! | [`crate::loss::ms_ssim_loss`] | [`ms_ssim_pixel_gradient`] |
//!
//! Every term is differentiated at exactly the normalisation the forward loss
//! uses — mean over pixels, mean over views, configured weight — so changing a
//! configured weight changes what is optimised, not only what is logged.

use crate::config::LossConfig;
use crate::loss::gaussian_kernel_1d;

// ---------------------------------------------------------------------------
// Constants — mirrored from `crate::loss` so forward and adjoint agree
// ---------------------------------------------------------------------------

/// SSIM stabiliser `(K₁·L)²` with `K₁ = 0.01`, `L = 1` — as in [`crate::loss`].
pub const SSIM_C1: f32 = 0.01 * 0.01;
/// SSIM stabiliser `(K₂·L)²` with `K₂ = 0.03`, `L = 1` — as in [`crate::loss`].
pub const SSIM_C2: f32 = 0.03 * 0.03;

/// Taps of the SSIM Gaussian window used by [`crate::loss::ssim_loss`].
pub const SSIM_KERNEL_TAPS: usize = 11;
/// Sigma of the SSIM Gaussian window used by [`crate::loss::ssim_loss`].
pub const SSIM_KERNEL_SIGMA: f32 = 1.5;
/// Taps of the smaller window [`crate::loss::ms_ssim_loss`] uses per scale.
pub const MS_SSIM_KERNEL_TAPS: usize = 7;
/// Sigma of the smaller window [`crate::loss::ms_ssim_loss`] uses per scale.
pub const MS_SSIM_KERNEL_SIGMA: f32 = 1.0;
/// Smallest dimension [`crate::loss::ms_ssim_loss`] still builds a scale for.
pub const MS_SSIM_MIN_DIM: usize = 7;
/// Maximum number of MS-SSIM scales (one per weight).
pub const MS_SSIM_MAX_SCALES: usize = 5;

// ---------------------------------------------------------------------------
// PhotometricSpec
// ---------------------------------------------------------------------------

/// Everything [`photometric_pixel_gradient`] needs besides the image pair.
///
/// Bundling the parameters keeps the differentiated objective *exactly* the one
/// a [`crate::loss::LossComputer`] instance reports: the caller passes that
/// computer's own [`LossConfig`], SSIM window and MS-SSIM scale weights rather
/// than rebuilding them from constants, so a computer built with
/// [`crate::loss::LossComputer::with_ms_ssim_weights`] cannot silently report
/// one objective while the optimiser descends another.
#[derive(Debug, Clone, Copy)]
pub struct PhotometricSpec<'a> {
    /// Loss-term weights of the objective being differentiated.
    pub config: &'a LossConfig,
    /// Separable Gaussian window of the single-scale SSIM term.
    pub ssim_kernel: &'a [f32],
    /// Per-scale MS-SSIM weights.
    pub ms_ssim_weights: &'a [f32; 5],
    /// Number of views the loss averages over (the `1/V` factor).
    pub num_views: usize,
}

impl<'a> PhotometricSpec<'a> {
    /// Build a spec straight from a [`crate::loss::LossComputer`].
    ///
    /// This is the form the trainer uses: it takes the window and the scale
    /// weights from the computer that produced the reported loss.
    pub fn from_loss_computer(computer: &'a crate::loss::LossComputer, num_views: usize) -> Self {
        Self {
            config: computer.config(),
            ssim_kernel: computer.ssim_kernel(),
            ms_ssim_weights: computer.ms_ssim_weights(),
            num_views,
        }
    }
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

/// Sub-gradient of `|x|`, taking `0` at the kink.
#[inline]
pub fn sign_or_zero(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Clamp a (possibly negative) index into `0..len`.  `len` must be non-zero.
#[inline]
fn clamp_index(idx: isize, len: usize) -> usize {
    idx.clamp(0, len as isize - 1) as usize
}

/// Extract one interleaved HWC channel as a dense plane, zero-filling any pixel
/// the source is too short for (the same tolerance the forward loss applies).
fn extract_channel(img: &[f32], n_pixels: usize, channel: usize) -> Vec<f32> {
    (0..n_pixels)
        .map(|p| img.get(p * 3 + channel).copied().unwrap_or(0.0))
        .collect()
}

/// Separable 2-D convolution with replicate-boundary padding.
///
/// Mirrors the private helper behind [`crate::loss::ssim_loss`] so forward loss
/// and backward gradient see the same window.
pub fn convolve_separable(src: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let n = width * height;
    let mut out = vec![0.0_f32; n];
    if n == 0 || src.len() < n || kernel.is_empty() {
        return out;
    }
    let half = (kernel.len() / 2) as isize;

    // Horizontal pass.
    let mut tmp = vec![0.0_f32; n];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let mut sum = 0.0_f32;
            for (i, &kv) in kernel.iter().enumerate() {
                let ix = clamp_index(x as isize + i as isize - half, width);
                sum += src[row + ix] * kv;
            }
            tmp[row + x] = sum;
        }
    }

    // Vertical pass.
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f32;
            for (i, &kv) in kernel.iter().enumerate() {
                let iy = clamp_index(y as isize + i as isize - half, height);
                sum += tmp[iy * width + x] * kv;
            }
            out[y * width + x] = sum;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// LocalStats
// ---------------------------------------------------------------------------

/// Windowed first- and second-order statistics of an image-pair channel.
///
/// All five maps use the same separable Gaussian window as the forward SSIM, so
/// the gradient is the exact adjoint away from the border (replicate padding is
/// only its own approximate transpose).
struct LocalStats {
    mu_x: Vec<f32>,
    mu_y: Vec<f32>,
    sigma_x2: Vec<f32>,
    sigma_y2: Vec<f32>,
    sigma_xy: Vec<f32>,
}

impl LocalStats {
    /// Compute the windowed statistics of one channel pair (`len = width·height`).
    fn compute(x: &[f32], y: &[f32], width: usize, height: usize, kernel: &[f32]) -> Self {
        let n = width * height;
        let xx: Vec<f32> = x.iter().take(n).map(|v| v * v).collect();
        let yy: Vec<f32> = y.iter().take(n).map(|v| v * v).collect();
        let xy: Vec<f32> = x.iter().zip(y.iter()).take(n).map(|(a, b)| a * b).collect();

        let mu_x = convolve_separable(x, width, height, kernel);
        let mu_y = convolve_separable(y, width, height, kernel);
        let mut sigma_x2 = convolve_separable(&xx, width, height, kernel);
        let mut sigma_y2 = convolve_separable(&yy, width, height, kernel);
        let mut sigma_xy = convolve_separable(&xy, width, height, kernel);
        for i in 0..n {
            sigma_x2[i] -= mu_x[i] * mu_x[i];
            sigma_y2[i] -= mu_y[i] * mu_y[i];
            sigma_xy[i] -= mu_x[i] * mu_y[i];
        }

        Self {
            mu_x,
            mu_y,
            sigma_x2,
            sigma_y2,
            sigma_xy,
        }
    }

    /// SSIM luminance term `l = (2μxμy + C₁)/(μx² + μy² + C₁)` at pixel `q`.
    #[inline]
    fn luminance(&self, q: usize) -> f32 {
        let (mu_x, mu_y) = (self.mu_x[q], self.mu_y[q]);
        (2.0 * mu_x * mu_y + SSIM_C1) / (mu_x * mu_x + mu_y * mu_y + SSIM_C1)
    }

    /// SSIM contrast-structure term `cs = (2σxy + C₂)/(σx² + σy² + C₂)` at `q`.
    #[inline]
    fn contrast_structure(&self, q: usize) -> f32 {
        (2.0 * self.sigma_xy[q] + SSIM_C2) / (self.sigma_x2[q] + self.sigma_y2[q] + SSIM_C2)
    }

    /// Accumulate `∂/∂x Σ_q [ wl(q)·l(q) + wcs(q)·cs(q) ]` into `out`.
    ///
    /// Every SSIM-family term in [`crate::loss`] is a weighted sum of the
    /// luminance and contrast-structure maps, so one adjoint pass serves all of
    /// them.  With `∂μx(q)/∂x(p) = G(p−q)`,
    /// `∂σx²(q)/∂x(p) = 2·G(p−q)·(x(p) − μx(q))` and
    /// `∂σxy(q)/∂x(p) = G(p−q)·(y(p) − μy(q))`, the sum over windows collapses
    /// into five convolutions with the same (symmetric) window.
    fn accumulate_gradient(&self, plane: &ChannelPlane<'_>, weights: &mut AdjointWeights<'_>) {
        let ChannelPlane {
            x,
            y,
            width,
            height,
            kernel,
        } = *plane;
        let n = width * height;
        let mut au = vec![0.0_f32; n];
        let mut av = vec![0.0_f32; n];
        let mut avmu = vec![0.0_f32; n];
        let mut az = vec![0.0_f32; n];
        let mut azmu = vec![0.0_f32; n];

        for q in 0..n {
            let mu_x = self.mu_x[q];
            let mu_y = self.mu_y[q];
            let a1 = 2.0 * mu_x * mu_y + SSIM_C1;
            let b1 = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
            let a2 = 2.0 * self.sigma_xy[q] + SSIM_C2;
            let b2 = self.sigma_x2[q] + self.sigma_y2[q] + SSIM_C2;

            // ∂l/∂μx = 2(μy·B₁ − μx·A₁)/B₁²
            au[q] = weights.wl[q] * 2.0 * (mu_y * b1 - mu_x * a1) / (b1 * b1);
            // ∂cs/∂σxy = 2/B₂ and ∂cs/∂σx² = −2·A₂/B₂²
            let v = weights.wcs[q] * 2.0 / b2;
            let z = weights.wcs[q] * -2.0 * a2 / (b2 * b2);
            av[q] = v;
            avmu[q] = v * mu_y;
            az[q] = z;
            azmu[q] = z * mu_x;
        }

        let gu = convolve_separable(&au, width, height, kernel);
        let gv = convolve_separable(&av, width, height, kernel);
        let gvmu = convolve_separable(&avmu, width, height, kernel);
        let gz = convolve_separable(&az, width, height, kernel);
        let gzmu = convolve_separable(&azmu, width, height, kernel);

        for p in 0..n {
            weights.out[p] += gu[p] + y[p] * gv[p] - gvmu[p] + x[p] * gz[p] - gzmu[p];
        }
    }
}

/// One channel plane of an image pair plus the window used on it.
#[derive(Clone, Copy)]
struct ChannelPlane<'a> {
    x: &'a [f32],
    y: &'a [f32],
    width: usize,
    height: usize,
    kernel: &'a [f32],
}

/// Per-pixel upstream weights on the luminance / contrast-structure maps, and
/// the buffer [`LocalStats::accumulate_gradient`] accumulates into.
struct AdjointWeights<'a> {
    wl: &'a [f32],
    wcs: &'a [f32],
    out: &'a mut [f32],
}

// ---------------------------------------------------------------------------
// SSIM / MS-SSIM adjoints
// ---------------------------------------------------------------------------

/// `∂(1 − SSIM)/∂pred` for one HWC RGB image pair.
///
/// The forward term is `1 − (1/3)·Σ_c mean_q S_c(q)` with `S = l·cs`, so
/// `∂(mean S)/∂l = cs/N` and `∂(mean S)/∂cs = l/N`.
pub fn ssim_pixel_gradient(
    pred: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) -> Vec<f32> {
    let n = width * height;
    let mut grad = vec![0.0_f32; n * 3];
    // `crate::loss::ssim_loss` returns a constant 0.0 for undersized buffers.
    if n == 0 || pred.len() < n * 3 || target.len() < n * 3 {
        return grad;
    }

    let outer = -1.0 / (3.0 * n as f32);
    let mut wl = vec![0.0_f32; n];
    let mut wcs = vec![0.0_f32; n];
    let mut chan = vec![0.0_f32; n];

    for c in 0..3 {
        let x = extract_channel(pred, n, c);
        let y = extract_channel(target, n, c);
        let stats = LocalStats::compute(&x, &y, width, height, kernel);

        for q in 0..n {
            wl[q] = outer * stats.contrast_structure(q);
            wcs[q] = outer * stats.luminance(q);
        }

        chan.fill(0.0);
        stats.accumulate_gradient(
            &ChannelPlane {
                x: &x,
                y: &y,
                width,
                height,
                kernel,
            },
            &mut AdjointWeights {
                wl: &wl,
                wcs: &wcs,
                out: &mut chan,
            },
        );
        for (p, &g) in chan.iter().enumerate() {
            grad[p * 3 + c] = g;
        }
    }

    grad
}

/// `∂(1 − MS-SSIM)/∂pred` for one HWC RGB image pair.
///
/// [`crate::loss::ms_ssim_loss`] combines *scalar* per-scale means,
/// `P = Πⱼ cs̄ⱼ^wⱼ · l̄_M^w_M`, so `∂(1−P)/∂cs̄ⱼ = −P·wⱼ/cs̄ⱼ` (likewise for the
/// coarsest luminance), followed by the adjoint of the box-downsampling chain.
/// Outside the `[0, 1]` clamp the forward term is constant → zero gradient.
pub fn ms_ssim_pixel_gradient(
    pred: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    weights: &[f32; 5],
) -> Vec<f32> {
    let mut grad = vec![0.0_f32; width * height * 3];
    // Mirror the guards of the forward term: inside them it is a constant.
    if width < 16 || height < 16 {
        return grad;
    }
    if pred.len() < width * height * 3 || target.len() < width * height * 3 {
        return grad;
    }

    let kernel = gaussian_kernel_1d(MS_SSIM_KERNEL_TAPS, MS_SSIM_KERNEL_SIGMA);

    // Pyramid dimensions, exactly as the forward term derives them.
    let mut dims: Vec<(usize, usize)> = Vec::with_capacity(MS_SSIM_MAX_SCALES);
    let mut w = width;
    let mut h = height;
    for _ in 0..MS_SSIM_MAX_SCALES {
        if w < MS_SSIM_MIN_DIM || h < MS_SSIM_MIN_DIM {
            break;
        }
        dims.push((w, h));
        w /= 2;
        h /= 2;
    }
    let num_scales = dims.len();
    if num_scales == 0 {
        return grad;
    }
    let last = num_scales - 1;

    // Build the pyramid.
    let mut preds: Vec<Vec<f32>> = Vec::with_capacity(num_scales);
    let mut tgts: Vec<Vec<f32>> = Vec::with_capacity(num_scales);
    preds.push(pred[..width * height * 3].to_vec());
    tgts.push(target[..width * height * 3].to_vec());
    for idx in 1..num_scales {
        let (pw, ph) = dims[idx - 1];
        let down_pred = downsample_2x(&preds[idx - 1], pw, ph);
        let down_tgt = downsample_2x(&tgts[idx - 1], pw, ph);
        preds.push(down_pred);
        tgts.push(down_tgt);
    }

    // Per-scale scalar means, then the product the forward term reports.
    let mut l_means = Vec::with_capacity(num_scales);
    let mut cs_means = Vec::with_capacity(num_scales);
    for idx in 0..num_scales {
        let (sw, sh) = dims[idx];
        let (l, cs) = ssim_component_means(&preds[idx], &tgts[idx], sw, sh, &kernel);
        l_means.push(l);
        cs_means.push(cs);
    }

    let mut product = 1.0_f32;
    for idx in 0..num_scales {
        product *= cs_means[idx].max(0.0).powf(weights[idx]);
    }
    product *= l_means[last].max(0.0).powf(weights[last]);

    // `1 − clamp(P, 0, 1)`: on or outside the clamp the term is constant.
    if !product.is_finite() || product <= 0.0 || product >= 1.0 {
        return grad;
    }

    // Walk from the coarsest scale back up, folding in each scale's
    // contribution and applying the downsampling adjoint between scales.
    let (mut acc_w, mut acc_h) = dims[last];
    let mut acc = vec![0.0_f32; acc_w * acc_h * 3];
    for idx in (0..num_scales).rev() {
        let (sw, sh) = dims[idx];
        let n = sw * sh;

        let d_cs = if weights[idx] > 0.0 && cs_means[idx] > 0.0 {
            -product * weights[idx] / cs_means[idx]
        } else {
            0.0
        };
        let d_l = if idx == last && weights[last] > 0.0 && l_means[last] > 0.0 {
            -product * weights[last] / l_means[last]
        } else {
            0.0
        };

        if d_cs != 0.0 || d_l != 0.0 {
            // `ssim_components` averages over pixels *and* the three channels.
            let inv_count = 1.0 / (3 * n) as f32;
            let wl = vec![d_l * inv_count; n];
            let wcs = vec![d_cs * inv_count; n];
            let mut chan = vec![0.0_f32; n];
            for c in 0..3 {
                let x = extract_channel(&preds[idx], n, c);
                let y = extract_channel(&tgts[idx], n, c);
                let stats = LocalStats::compute(&x, &y, sw, sh, &kernel);
                chan.fill(0.0);
                stats.accumulate_gradient(
                    &ChannelPlane {
                        x: &x,
                        y: &y,
                        width: sw,
                        height: sh,
                        kernel: &kernel,
                    },
                    &mut AdjointWeights {
                        wl: &wl,
                        wcs: &wcs,
                        out: &mut chan,
                    },
                );
                for (p, &g) in chan.iter().enumerate() {
                    acc[p * 3 + c] += g;
                }
            }
        }

        if idx > 0 {
            let (fw, fh) = dims[idx - 1];
            acc = upsample_adjoint_2x(&acc, acc_w, acc_h, fw, fh);
            acc_w = fw;
            acc_h = fh;
        }
    }

    for (dst, src) in grad.iter_mut().zip(acc.iter()) {
        *dst += src;
    }

    grad
}

/// Mean SSIM luminance and contrast-structure over all pixels *and* channels —
/// the two scalars [`crate::loss::ms_ssim_loss`] combines per scale.
pub fn ssim_component_means(
    pred: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) -> (f32, f32) {
    let n = width * height;
    if n == 0 {
        return (0.0, 0.0);
    }

    let mut l_sum = 0.0_f32;
    let mut cs_sum = 0.0_f32;
    for c in 0..3 {
        let x = extract_channel(pred, n, c);
        let y = extract_channel(target, n, c);
        let stats = LocalStats::compute(&x, &y, width, height, kernel);
        for q in 0..n {
            l_sum += stats.luminance(q);
            cs_sum += stats.contrast_structure(q);
        }
    }

    let count = (3 * n) as f32;
    (l_sum / count, cs_sum / count)
}

/// 2× box downsample of an HWC RGB image — identical to the forward term's.
pub fn downsample_2x(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let new_w = width / 2;
    let new_h = height / 2;
    if new_w == 0 || new_h == 0 {
        return Vec::new();
    }

    let mut out = vec![0.0_f32; new_w * new_h * 3];
    for y in 0..new_h {
        for x in 0..new_w {
            for c in 0..3 {
                let mut sum = 0.0_f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let idx = ((y * 2 + dy) * width + (x * 2 + dx)) * 3 + c;
                        if idx < image.len() {
                            sum += image[idx];
                        }
                    }
                }
                out[(y * new_w + x) * 3 + c] = sum / 4.0;
            }
        }
    }

    out
}

/// Adjoint of [`downsample_2x`]: spread each coarse gradient over the 2×2 block
/// it averaged, carrying the same `1/4` factor.
pub fn upsample_adjoint_2x(
    grad_coarse: &[f32],
    coarse_w: usize,
    coarse_h: usize,
    fine_w: usize,
    fine_h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; fine_w * fine_h * 3];
    for y in 0..coarse_h {
        for x in 0..coarse_w {
            for c in 0..3 {
                let g = grad_coarse
                    .get((y * coarse_w + x) * 3 + c)
                    .copied()
                    .unwrap_or(0.0)
                    * 0.25;
                if g == 0.0 {
                    continue;
                }
                for dy in 0..2 {
                    for dx in 0..2 {
                        let fy = y * 2 + dy;
                        let fx = x * 2 + dx;
                        if fy < fine_h && fx < fine_w {
                            out[(fy * fine_w + fx) * 3 + c] += g;
                        }
                    }
                }
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Image-space gradient `∂L/∂rendered` of the configured photometric loss for
/// one view, as a flat HWC RGB buffer of length `width·height·3`.
///
/// Every term is differentiated at exactly the normalisation
/// [`crate::loss::LossComputer::compute`] uses — mean over pixels, mean over
/// views ([`PhotometricSpec::num_views`]), configured weight — and a zero-weight
/// term is skipped.  LPIPS is absent on purpose:
/// [`crate::loss::LossComputer::compute`] reports a hard `0.0` for it, so it is
/// not part of the objective the trainer optimises.
pub fn photometric_pixel_gradient(
    spec: &PhotometricSpec<'_>,
    rendered: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let cfg = spec.config;
    let npx = width * height;
    let mut grad = vec![0.0_f32; npx * 3];
    if npx == 0 || rendered.is_empty() || target.is_empty() || spec.num_views == 0 {
        return grad;
    }
    let view_norm = 1.0 / spec.num_views as f32;

    // ---- L1: mean absolute error over every element of the view ------------
    if cfg.w_l1 > 0.0 {
        // `crate::loss::l1_loss` divides by `pred.len()`.
        let k = cfg.w_l1 * view_norm / rendered.len() as f32;
        for ((g, r), t) in grad.iter_mut().zip(rendered.iter()).zip(target.iter()) {
            *g += k * sign_or_zero(r - t);
        }
    }

    // ---- SSIM dissimilarity (1 − SSIM) -------------------------------------
    if cfg.w_ssim > 0.0 {
        let g = ssim_pixel_gradient(rendered, target, width, height, spec.ssim_kernel);
        let k = cfg.w_ssim * view_norm;
        for (dst, src) in grad.iter_mut().zip(g.iter()) {
            *dst += k * src;
        }
    }

    // ---- MS-SSIM dissimilarity (1 − MS-SSIM) -------------------------------
    if cfg.w_ms_ssim > 0.0 {
        let g = ms_ssim_pixel_gradient(rendered, target, width, height, spec.ms_ssim_weights);
        let k = cfg.w_ms_ssim * view_norm;
        for (dst, src) in grad.iter_mut().zip(g.iter()) {
            *dst += k * src;
        }
    }

    grad
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::{l1_loss, ms_ssim_loss, ssim_loss, LossComputer};

    /// Deterministic, non-trivial image pair in `[0.1, 0.9]`.
    fn make_pair(width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
        let n = width * height * 3;
        let mut pred = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            let f = i as f32;
            pred.push(0.5 + 0.4 * (f * 0.137).sin());
            target.push(0.5 + 0.4 * (f * 0.091).cos());
        }
        (pred, target)
    }

    fn loss_cfg(w_l1: f32, w_ssim: f32, w_ms_ssim: f32) -> LossConfig {
        LossConfig {
            w_l1,
            w_ssim,
            w_ms_ssim,
            w_lpips: 0.0,
            w_position_reg: 0.0,
            w_scale_reg: 0.0,
            w_opacity_reg: 0.0,
            w_normal: 0.0,
            w_gradient_penalty: 0.0,
            gradient_penalty_threshold: 100.0,
            w_scale_reg_max_scale: crate::loss::MAX_REASONABLE_WORLD_SCALE,
        }
    }

    /// `photometric_pixel_gradient` with the stock SSIM window and MS-SSIM
    /// scale weights.
    fn pixel_grad(c: &LossConfig, p: &[f32], t: &[f32], w: usize, h: usize, v: usize) -> Vec<f32> {
        let kernel = gaussian_kernel_1d(SSIM_KERNEL_TAPS, SSIM_KERNEL_SIGMA);
        let spec = PhotometricSpec {
            config: c,
            ssim_kernel: &kernel,
            ms_ssim_weights: &LossComputer::DEFAULT_MS_SSIM_WEIGHTS,
            num_views: v,
        };
        photometric_pixel_gradient(&spec, p, t, w, h)
    }

    /// The scalar photometric objective `LossComputer::compute` reports for one
    /// view (its regularisation terms are constant w.r.t. the pixels).
    fn scalar_loss(cfg: &LossConfig, pred: &[f32], tgt: &[f32], w: usize, h: usize) -> f32 {
        let kernel = gaussian_kernel_1d(SSIM_KERNEL_TAPS, SSIM_KERNEL_SIGMA);
        cfg.w_l1 * l1_loss(pred, tgt)
            + cfg.w_ssim * ssim_loss(pred, tgt, w, h, &kernel)
            + cfg.w_ms_ssim * ms_ssim_loss(pred, tgt, w, h, &LossComputer::DEFAULT_MS_SSIM_WEIGHTS)
    }

    /// Central-difference derivative of [`scalar_loss`] w.r.t. `p[i]`.
    fn numeric_grad(c: &LossConfig, p: &[f32], t: &[f32], w: usize, h: usize, i: usize) -> f32 {
        let eps = 0.02_f32;
        let mut plus = p.to_vec();
        let mut minus = p.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        (scalar_loss(c, &plus, t, w, h) - scalar_loss(c, &minus, t, w, h)) / (2.0 * eps)
    }

    fn relative_error(analytic: f32, numeric: f32) -> f32 {
        let denom = analytic.abs().max(numeric.abs()).max(1e-6);
        (analytic - numeric).abs() / denom
    }

    #[test]
    fn l1_gradient_is_view_normalised_sign() {
        let (w, h) = (4, 4);
        let (pred, target) = make_pair(w, h);
        let cfg = loss_cfg(0.8, 0.0, 0.0);
        let grad = pixel_grad(&cfg, &pred, &target, w, h, 2);

        let k = 0.8 / 2.0 / pred.len() as f32;
        for i in 0..pred.len() {
            let expected = k * sign_or_zero(pred[i] - target[i]);
            assert!((grad[i] - expected).abs() < 1e-9, "L1 gradient at {i}");
        }
    }

    #[test]
    fn ssim_gradient_matches_finite_differences() {
        let (w, h) = (16, 16);
        let (pred, target) = make_pair(w, h);
        let cfg = loss_cfg(0.0, 1.0, 0.0);
        let grad = pixel_grad(&cfg, &pred, &target, w, h, 1);

        // Well-interior pixels only: replicate padding makes the window adjoint
        // approximate within `kernel_taps / 2` of the border.
        for &i in &[409_usize, 354, 317, 453, 255, 510] {
            let numeric = numeric_grad(&cfg, &pred, &target, w, h, i);
            assert!(
                relative_error(grad[i], numeric) < 0.1,
                "SSIM gradient at {i}: analytic {} vs numeric {numeric}",
                grad[i]
            );
        }
    }

    #[test]
    fn ms_ssim_gradient_matches_finite_differences() {
        let (w, h) = (64, 64);
        let (pred, target) = make_pair(w, h);
        let cfg = loss_cfg(0.0, 0.0, 1.0);
        let grad = pixel_grad(&cfg, &pred, &target, w, h, 1);

        for &i in &[6240_usize, 6241, 5484] {
            let numeric = numeric_grad(&cfg, &pred, &target, w, h, i);
            assert!(
                relative_error(grad[i], numeric) < 0.25,
                "MS-SSIM gradient at {i}: analytic {} vs numeric {numeric}",
                grad[i]
            );
        }

        // At the optimum (identical images) the clamped product pins the term.
        let flat = pixel_grad(&cfg, &pred, &pred, w, h, 1);
        assert!(flat.iter().all(|g| *g == 0.0));
    }

    #[test]
    fn configured_weights_reach_the_gradient() {
        // Regression: the image-space gradient used to be a hardcoded MSE that
        // ignored every configured loss weight.
        let (w, h) = (16, 16);
        let (pred, target) = make_pair(w, h);
        let g_l1 = pixel_grad(&loss_cfg(1.0, 0.0, 0.0), &pred, &target, w, h, 1);
        let g_ssim = pixel_grad(&loss_cfg(0.0, 1.0, 0.0), &pred, &target, w, h, 1);
        let g_double = pixel_grad(&loss_cfg(2.0, 0.0, 0.0), &pred, &target, w, h, 1);

        assert!(g_l1.iter().any(|g| g.abs() > 0.0));
        assert!(g_ssim.iter().any(|g| g.abs() > 0.0));
        // Two different objectives must not yield the same descent direction,
        // and doubling a weight must double that term's contribution.
        let zipped = g_l1.iter().zip(g_ssim.iter());
        assert!(zipped.map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max) > 1e-9);
        for (a, b) in g_double.iter().zip(g_l1.iter()) {
            assert!((a - 2.0 * b).abs() < 1e-9);
        }
    }

    #[test]
    fn downsample_and_adjoint_are_transposes() {
        // <D·x, y> == <x, Dᵀ·y> for the 2× box filter.
        let (w, h) = (4, 4);
        let x: Vec<f32> = (0..w * h * 3).map(|i| i as f32 * 0.01).collect();
        let y: Vec<f32> = (0..(w / 2) * (h / 2) * 3)
            .map(|i| 1.0 - i as f32 * 0.02)
            .collect();

        let dx = downsample_2x(&x, w, h);
        let dty = upsample_adjoint_2x(&y, w / 2, h / 2, w, h);
        let lhs: f32 = dx.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let rhs: f32 = x.iter().zip(dty.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-5, "lhs {lhs} rhs {rhs}");
    }

    #[test]
    fn spec_from_loss_computer_uses_custom_ms_ssim_weights() {
        // Regression (F140): the trainer used to hardcode
        // `DEFAULT_MS_SSIM_WEIGHTS`, so a computer built with custom weights
        // reported one objective while the optimiser descended another.
        let custom = [0.5_f32, 0.5, 0.0, 0.0, 0.0];
        let computer = LossComputer::with_ms_ssim_weights(loss_cfg(0.0, 0.0, 1.0), custom);
        let spec = PhotometricSpec::from_loss_computer(&computer, 1);
        assert_eq!(spec.ms_ssim_weights, &custom);
        assert_eq!(spec.ssim_kernel.len(), SSIM_KERNEL_TAPS);

        let (w, h) = (32, 32);
        let (pred, target) = make_pair(w, h);
        let custom_grad = photometric_pixel_gradient(&spec, &pred, &target, w, h);
        let default_grad = pixel_grad(&loss_cfg(0.0, 0.0, 1.0), &pred, &target, w, h, 1);
        let max_delta = custom_grad
            .iter()
            .zip(default_grad.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_delta > 1e-9,
            "custom MS-SSIM weights must change the descent direction"
        );
    }
}
