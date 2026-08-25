//! Pure tensor / image helpers behind [`crate::diffusion_target`].
//!
//! Everything here is a total function of its arguments — resampling, tensor
//! packing/unpacking, camera-pose encoding, depth-based view warping and the
//! DDPM `ᾱ` schedule.  None of it touches the pipeline, the VAE or any
//! generator state, which is exactly why it lives apart from the orchestration
//! in the parent module: it is independently testable and keeps that file
//! within the file-size budget.

use candle_core::{Device, Tensor};
use nalgebra as na;

use oxigaf_diffusion::DiffusionError;
use oxigaf_flame::Camera;

use super::{CLIP_INPUT_SIZE, CLIP_MEAN, CLIP_STD};

/// Convert HWC images to a batched tensor [N, C, H, W].
pub(crate) fn images_to_tensor(
    images: &[Vec<f32>],
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor, DiffusionError> {
    let n = images.len();
    let h = height as usize;
    let w = width as usize;

    // Create NCHW tensor
    let mut data = vec![0.0_f32; n * 3 * h * w];

    for (idx, img) in images.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                let hwc_idx = (y * w + x) * 3;
                let r = img.get(hwc_idx).copied().unwrap_or(0.0);
                let g = img.get(hwc_idx + 1).copied().unwrap_or(0.0);
                let b = img.get(hwc_idx + 2).copied().unwrap_or(0.0);

                let base = idx * 3 * h * w;
                let channel_stride = h * w;
                data[base + y * w + x] = r;
                data[base + channel_stride + y * w + x] = g;
                data[base + 2 * channel_stride + y * w + x] = b;
            }
        }
    }

    Tensor::from_vec(data, (n, 3, h, w), device)
        .map_err(|e| DiffusionError::Inference(format!("images_to_tensor: {e}")))
}

/// Convert a tensor [C, H, W] to HWC Vec<f32>.
pub(crate) fn tensor_to_hwc_image(
    tensor: &Tensor,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, DiffusionError> {
    let h = height as usize;
    let w = width as usize;

    // Flatten and convert to Vec<f32>
    let data: Vec<f32> = tensor
        .flatten_all()
        .and_then(|t| t.to_vec1())
        .map_err(|e| DiffusionError::Inference(format!("tensor_to_hwc: {e}")))?;

    if data.len() < 3 * h * w {
        return Err(DiffusionError::Inference(format!(
            "tensor_to_hwc: data length {} < expected {}",
            data.len(),
            3 * h * w
        )));
    }

    // CHW to HWC
    let mut hwc = vec![0.0_f32; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let hwc_idx = (y * w + x) * 3;
            let channel_stride = h * w;
            let pixel_offset = y * w + x;
            hwc[hwc_idx] = data.get(pixel_offset).copied().unwrap_or(0.0);
            hwc[hwc_idx + 1] = data
                .get(channel_stride + pixel_offset)
                .copied()
                .unwrap_or(0.0);
            hwc[hwc_idx + 2] = data
                .get(2 * channel_stride + pixel_offset)
                .copied()
                .unwrap_or(0.0);
        }
    }

    Ok(hwc)
}

/// Convert cameras to a batched pose tensor `[num_views, 12]` (flattened 4x3
/// extrinsics).
///
/// Cameras are tiled cyclically when fewer than `num_views` are supplied and
/// dropped when more are, for the same reason the normal latents are (see
/// [`normal_maps_to_tensor`]): the pipeline denoises exactly
/// `DiffusionConfig::num_views` latents and every per-view conditioning tensor
/// has to line up with that batch dimension.
pub(crate) fn cameras_to_tensor(
    cameras: &[Camera],
    num_views: usize,
    device: &Device,
) -> Result<Tensor, DiffusionError> {
    if cameras.is_empty() || num_views == 0 {
        return Err(DiffusionError::Inference(format!(
            "cameras_to_tensor: {} camera(s), {num_views} view(s)",
            cameras.len()
        )));
    }

    let mut data = vec![0.0_f32; num_views * 12];

    for i in 0..num_views {
        let cam = &cameras[i % cameras.len()];
        // Flatten rotation (3x3) and translation (3)
        // Row-major: r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz
        for r in 0..3 {
            for c in 0..3 {
                data[i * 12 + r * 3 + c] = cam.rotation[(r, c)];
            }
        }
        data[i * 12 + 9] = cam.translation.x;
        data[i * 12 + 10] = cam.translation.y;
        data[i * 12 + 11] = cam.translation.z;
    }

    Tensor::from_vec(data, (num_views, 12), device)
        .map_err(|e| DiffusionError::Inference(format!("cameras_to_tensor: {e}")))
}

/// Half-pixel-centred bilinear sampling weights along one axis.
///
/// Maps destination index `dst` of a `dst_len`-long axis onto the source axis
/// with `src = (dst + 0.5) · src_len / dst_len − 0.5`, which covers the *whole*
/// source extent for both up- and down-scaling.  Returns `(lo, hi, frac)` so
/// the resampled value is `v[lo] + (v[hi] − v[lo]) · frac`.
pub(crate) fn bilinear_axis(dst: usize, dst_len: usize, src_len: usize) -> (usize, usize, f32) {
    if src_len == 0 || dst_len == 0 {
        return (0, 0, 0.0);
    }
    let scale = src_len as f32 / dst_len as f32;
    let pos = ((dst as f32 + 0.5) * scale - 0.5).clamp(0.0, (src_len - 1) as f32);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(src_len - 1);
    (lo, hi, pos - lo as f32)
}

/// Bilinearly resample an interleaved HWC RGB image.
pub(crate) fn resample_hwc(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; dst_w * dst_h * 3];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }

    for y in 0..dst_h {
        let (y0, y1, wy) = bilinear_axis(y, dst_h, src_h);
        for x in 0..dst_w {
            let (x0, x1, wx) = bilinear_axis(x, dst_w, src_w);
            for c in 0..3 {
                let p00 = src.get((y0 * src_w + x0) * 3 + c).copied().unwrap_or(0.0);
                let p01 = src.get((y0 * src_w + x1) * 3 + c).copied().unwrap_or(0.0);
                let p10 = src.get((y1 * src_w + x0) * 3 + c).copied().unwrap_or(0.0);
                let p11 = src.get((y1 * src_w + x1) * 3 + c).copied().unwrap_or(0.0);
                let top = p00 + (p01 - p00) * wx;
                let bottom = p10 + (p11 - p10) * wx;
                if let Some(slot) = out.get_mut((y * dst_w + x) * 3 + c) {
                    *slot = top + (bottom - top) * wy;
                }
            }
        }
    }

    out
}

/// Prepare the CLIP reference image from the first rendered view.
///
/// `images` is the `(V, 3, H, W)` render batch in `[0, 1]`.  The first view is
/// bilinearly resampled to 224×224 — covering the *whole* frame; the previous
/// integer-ratio box filter cropped the right/bottom edges and, for inputs
/// smaller than twice the target, degenerated into a plain top-left crop — and
/// then normalised with CLIP's own channel mean/std, which is what the ViT
/// image encoder behind the IP-Adapter tokens was trained on.
pub(crate) fn prepare_reference_image(images: &Tensor) -> Result<Tensor, DiffusionError> {
    let first = images
        .narrow(0, 0, 1)
        .map_err(|e| DiffusionError::Inference(format!("narrow: {e}")))?;

    let (_b, c, h, w) = first
        .dims4()
        .map_err(|e| DiffusionError::Inference(format!("dims4: {e}")))?;
    if c == 0 || h == 0 || w == 0 {
        return Err(DiffusionError::Inference(format!(
            "reference image has an empty dimension: ({c}, {h}, {w})"
        )));
    }

    let data: Vec<f32> = first
        .flatten_all()
        .and_then(|t| t.to_vec1())
        .map_err(|e| DiffusionError::Inference(format!("flatten: {e}")))?;

    let size = CLIP_INPUT_SIZE;
    // The axis weights are shared by every channel, so compute them once.
    let rows: Vec<(usize, usize, f32)> = (0..size).map(|y| bilinear_axis(y, size, h)).collect();
    let cols: Vec<(usize, usize, f32)> = (0..size).map(|x| bilinear_axis(x, size, w)).collect();

    let mut resized = vec![0.0_f32; c * size * size];
    for ch in 0..c {
        let src_plane = ch * h * w;
        let dst_plane = ch * size * size;
        let mean = CLIP_MEAN.get(ch).copied().unwrap_or(0.0);
        let std_dev = CLIP_STD.get(ch).copied().unwrap_or(1.0);
        for (y, &(y0, y1, wy)) in rows.iter().enumerate() {
            for (x, &(x0, x1, wx)) in cols.iter().enumerate() {
                let p00 = data.get(src_plane + y0 * w + x0).copied().unwrap_or(0.0);
                let p01 = data.get(src_plane + y0 * w + x1).copied().unwrap_or(0.0);
                let p10 = data.get(src_plane + y1 * w + x0).copied().unwrap_or(0.0);
                let p11 = data.get(src_plane + y1 * w + x1).copied().unwrap_or(0.0);
                let top = p00 + (p01 - p00) * wx;
                let bottom = p10 + (p11 - p10) * wx;
                let value = top + (bottom - top) * wy;
                if let Some(slot) = resized.get_mut(dst_plane + y * size + x) {
                    *slot = (value - mean) / std_dev;
                }
            }
        }
    }

    Tensor::from_vec(resized, (1, c, size, size), first.device())
        .map_err(|e| DiffusionError::Inference(format!("reference image: {e}")))
}

/// Build the `(num_views, 3, size, size)` pixel tensor the VAE encoder expects
/// from per-view HWC normal maps in `[0, 1]`.
///
/// Each map is bilinearly resampled to `size × size` and rescaled to the
/// `[-1, 1]` range the encoder was trained on.  Views are tiled cyclically when
/// fewer maps than views are supplied and dropped when more are: the pipeline
/// always denoises exactly `num_views` latents and concatenates these onto them
/// channel-wise, so the batch dimensions have to agree.
pub(crate) fn normal_maps_to_tensor(
    maps: &[Vec<f32>],
    num_views: usize,
    src_width: usize,
    src_height: usize,
    size: usize,
    device: &Device,
) -> Result<Tensor, DiffusionError> {
    if maps.is_empty() || num_views == 0 || size == 0 || src_width == 0 || src_height == 0 {
        return Err(DiffusionError::Inference(format!(
            "normal maps: {} map(s), {num_views} view(s), source {src_width}x{src_height}, \
             target {size}x{size}",
            maps.len()
        )));
    }

    let plane = size * size;
    let mut data = vec![0.0_f32; num_views * 3 * plane];

    for view in 0..num_views {
        let map = &maps[view % maps.len()];
        let resized = resample_hwc(map, src_width, src_height, size, size);
        let base = view * 3 * plane;
        for pixel in 0..plane {
            for c in 0..3 {
                let value = resized.get(pixel * 3 + c).copied().unwrap_or(0.0);
                if let Some(slot) = data.get_mut(base + c * plane + pixel) {
                    // [0, 1] → [-1, 1]
                    *slot = value.mul_add(2.0, -1.0);
                }
            }
        }
    }

    Tensor::from_vec(data, (num_views, 3, size, size), device)
        .map_err(|e| DiffusionError::Inference(format!("normal_maps_to_tensor: {e}")))
}

/// A source view forward-warped into a target view.
pub(crate) struct WarpedView {
    /// Warped RGB pixels in HWC order; entries whose `mask` is `false` are unset.
    pub(crate) pixels: Vec<f32>,
    /// `true` where at least one source pixel projected onto the target pixel.
    pub(crate) mask: Vec<bool>,
}

/// Warp a source view to a target view using depth.
///
/// This is a forward scatter, so it leaves holes wherever no source pixel lands;
/// `WarpedView::mask` records which target pixels were actually written so the
/// caller can skip the rest.  Competing sources are resolved with a per-target
/// depth buffer — nearest wins — instead of last-write-wins.
pub(crate) fn warp_view(
    src_view: &[f32],
    src_cam: &Camera,
    tgt_cam: &Camera,
    src_depth: &[f32],
    width: usize,
    height: usize,
) -> WarpedView {
    let mut warped = WarpedView {
        pixels: vec![0.0_f32; width * height * 3],
        mask: vec![false; width * height],
    };
    let mut depth_buffer = vec![f32::INFINITY; width * height];

    // For each pixel in source view
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let depth = src_depth.get(idx).copied().unwrap_or(0.0);

            if depth <= 0.0 || !depth.is_finite() {
                continue;
            }

            // Unproject to 3D
            let px = (x as f32 - src_cam.cx) / src_cam.focal_x;
            let py = (y as f32 - src_cam.cy) / src_cam.focal_y;
            let point_cam = na::Vector3::new(px * depth, py * depth, depth);

            // Transform to world space
            let r_inv = src_cam.rotation.transpose();
            let point_world = r_inv * (point_cam - src_cam.translation);

            // Project to target camera
            let point_tgt_cam = tgt_cam.rotation * point_world + tgt_cam.translation;

            if point_tgt_cam.z <= 0.0 || !point_tgt_cam.z.is_finite() {
                continue;
            }

            let tx = (point_tgt_cam.x / point_tgt_cam.z) * tgt_cam.focal_x + tgt_cam.cx;
            let ty = (point_tgt_cam.y / point_tgt_cam.z) * tgt_cam.focal_y + tgt_cam.cy;
            if !tx.is_finite() || !ty.is_finite() {
                continue;
            }

            let tx_i = tx.round() as i32;
            let ty_i = ty.round() as i32;

            if tx_i >= 0 && tx_i < width as i32 && ty_i >= 0 && ty_i < height as i32 {
                let tgt_idx = (ty_i as usize) * width + (tx_i as usize);

                // Z-buffer: keep the nearest source sample for this target pixel.
                let Some(slot) = depth_buffer.get_mut(tgt_idx) else {
                    continue;
                };
                if point_tgt_cam.z >= *slot {
                    continue;
                }
                *slot = point_tgt_cam.z;

                let src_hwc = idx * 3;
                let tgt_hwc = tgt_idx * 3;
                if let Some(dst) = warped.pixels.get_mut(tgt_hwc..tgt_hwc + 3) {
                    dst[0] = src_view.get(src_hwc).copied().unwrap_or(0.0);
                    dst[1] = src_view.get(src_hwc + 1).copied().unwrap_or(0.0);
                    dst[2] = src_view.get(src_hwc + 2).copied().unwrap_or(0.0);
                }
                if let Some(valid) = warped.mask.get_mut(tgt_idx) {
                    *valid = true;
                }
            }
        }
    }

    warped
}

/// SDS timestep weighting factor `w(t) = max(1 − ᾱ(t), 0.001)`.
///
/// Higher timesteps (more noise) get higher weights.  `max_timestep` is the
/// length of the DDPM *training* schedule — pass [`super::DDPM_TRAIN_TIMESTEPS`]
/// unless the caller deliberately distils from a different schedule.
///
/// This is the single definition of the variance-preserving weighting: both
/// `SdsLoss::weight` (for [`super::SdsWeighting::SigmaBased`]) and
/// [`super::DiffusionTargetGenerator::compute_sds_gradient`] call it, so the
/// reported loss and the applied gradient carry identical factors.  The `0.001` floor
/// matters at the low timesteps annealing ends on, where `1 − ᾱ(t)` approaches
/// zero and an unfloored weight would silently switch distillation off while
/// still reporting a loss.
pub fn sds_timestep_weight(timestep: u32, max_timestep: u32) -> f32 {
    let alpha = ddpm_alpha_cumprod(timestep, max_timestep);
    let sigma_sq = 1.0 - alpha;

    // w(t) = sigma(t)^2 for variance-preserving weighting
    sigma_sq.max(0.001)
}

/// Approximate DDPM `alpha_cumprod` for a given timestep.
///
/// `max_timestep` is the length of the training schedule.  It is clamped to at
/// least 2 because the beta interpolation divides by `max_timestep - 1`, which
/// would otherwise yield `inf`/`NaN` and propagate straight into the SDS
/// weighting.  `timestep` is likewise clamped to the last schedule entry, so a
/// caller passing `t == max_timestep` stops at `beta_end` instead of
/// extrapolating past it.
pub(crate) fn ddpm_alpha_cumprod(timestep: u32, max_timestep: u32) -> f32 {
    // Scaled linear beta schedule (SD 2.1 style)
    let beta_start = 0.00085_f32.sqrt();
    let beta_end = 0.012_f32.sqrt();

    let last = max_timestep.max(2) - 1;
    let denom = last as f32;

    let mut alpha_cumprod = 1.0_f32;
    for t in 0..=timestep.min(last) {
        let beta = beta_start + (beta_end - beta_start) * (t as f32) / denom;
        let beta = beta * beta;
        let alpha = 1.0 - beta;
        alpha_cumprod *= alpha;
    }

    alpha_cumprod
}
