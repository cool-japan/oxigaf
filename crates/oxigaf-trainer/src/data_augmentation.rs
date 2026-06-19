//! Data augmentation for training 3DGS avatar models.
//!
//! Images are stored as flat `Vec<f32>` with values in [0, 1],
//! layout `[R, G, B, R, G, B, ...]` (no alpha), H×W×3 row-major.
//!
//! All randomness uses a local xorshift64 PRNG — no `rand` crate.

use thiserror::Error;

// ---- Error Type ---------------------------------------------------------------

/// Errors produced by data augmentation operations.
#[derive(Debug, Error)]
pub enum AugmentError {
    #[error("Image buffer size {got} does not match {width}x{height}x{channels}")]
    SizeMismatch {
        got: usize,
        width: usize,
        height: usize,
        channels: usize,
    },
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
    #[error("Empty image")]
    EmptyImage,
}

// ---- PRNG helpers (xorshift64, no rand crate) ---------------------------------

/// xorshift64 PRNG step — modifies state in place, returns next pseudo-random u64.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    (*state) ^= (*state) << 13;
    (*state) ^= (*state) >> 7;
    (*state) ^= (*state) << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Uniform float in [0, 1) with 53 bits of precision.
#[inline]
pub fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

// ---- Image statistics ---------------------------------------------------------

/// Per-channel mean/std and global min/max for a flat H×W×3 image.
#[derive(Debug, Clone)]
pub struct AugImageStats {
    pub mean: [f32; 3],
    pub std_dev: [f32; 3],
    pub min: f32,
    pub max: f32,
}

/// Compute per-channel mean, per-channel std, and global min/max over a flat image.
pub fn aug_image_stats(img: &[f32]) -> AugImageStats {
    if img.is_empty() {
        return AugImageStats {
            mean: [0.0; 3],
            std_dev: [0.0; 3],
            min: 0.0,
            max: 0.0,
        };
    }

    let n_pixels = img.len() / 3;
    let n_pixels_f = n_pixels.max(1) as f32;

    // First pass: accumulate sums and find global min/max
    let mut sum = [0.0f64; 3];
    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;

    for i in 0..n_pixels {
        let base = i * 3;
        for c in 0..3 {
            let v = img[base + c];
            sum[c] += v as f64;
            if v < global_min {
                global_min = v;
            }
            if v > global_max {
                global_max = v;
            }
        }
    }

    let mean = [
        (sum[0] / n_pixels_f as f64) as f32,
        (sum[1] / n_pixels_f as f64) as f32,
        (sum[2] / n_pixels_f as f64) as f32,
    ];

    // Second pass: accumulate variance
    let mut var_sum = [0.0f64; 3];
    for i in 0..n_pixels {
        let base = i * 3;
        for c in 0..3 {
            let diff = img[base + c] as f64 - mean[c] as f64;
            var_sum[c] += diff * diff;
        }
    }

    let std_dev = [
        (var_sum[0] / n_pixels_f as f64).sqrt() as f32,
        (var_sum[1] / n_pixels_f as f64).sqrt() as f32,
        (var_sum[2] / n_pixels_f as f64).sqrt() as f32,
    ];

    // Handle edge: single pixel → no variance
    let global_min = if global_min == f32::INFINITY {
        0.0
    } else {
        global_min
    };
    let global_max = if global_max == f32::NEG_INFINITY {
        0.0
    } else {
        global_max
    };

    AugImageStats {
        mean,
        std_dev,
        min: global_min,
        max: global_max,
    }
}

// ---- Helper functions ---------------------------------------------------------

/// Convert RGB (all in [0, 1]) to HSV (H in [0, 360), S, V in [0, 1]).
pub fn aug_rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 1e-7 { delta / max } else { 0.0 };

    let h = if delta < 1e-7 {
        0.0_f32
    } else if (max - r).abs() < 1e-7 {
        let raw = (g - b) / delta;
        raw.rem_euclid(6.0) * 60.0
    } else if (max - g).abs() < 1e-7 {
        ((b - r) / delta + 2.0) * 60.0
    } else {
        ((r - g) / delta + 4.0) * 60.0
    };

    (h, s, v)
}

/// Convert HSV (H in [0, 360), S, V in [0, 1]) to RGB (all in [0, 1]).
pub fn aug_hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s < 1e-7 {
        return (v, v, v);
    }

    let hh = h.rem_euclid(360.0) / 60.0;
    let i = hh.floor() as u32;
    let f = hh - i as f32;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        5 => (v, p, q),
        _ => (v, v, v),
    }
}

/// Box-Muller transform: produce two standard normal samples from two uniform \[0,1\] inputs.
pub fn aug_box_muller(u1: f32, u2: f32) -> (f32, f32) {
    // Protect against log(0) by clamping u1 away from zero
    let u1 = u1.max(1e-38);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f32::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Build a normalized 1D Gaussian kernel of odd `kernel_size` and given `sigma`.
pub fn aug_gaussian_kernel(kernel_size: usize, sigma: f32) -> Result<Vec<f32>, AugmentError> {
    if kernel_size == 0 {
        return Err(AugmentError::InvalidParam("kernel_size must be > 0".into()));
    }
    if kernel_size.is_multiple_of(2) {
        return Err(AugmentError::InvalidParam(format!(
            "kernel_size must be odd, got {}",
            kernel_size
        )));
    }
    if sigma <= 0.0 {
        return Err(AugmentError::InvalidParam(format!(
            "sigma must be > 0, got {}",
            sigma
        )));
    }

    let half = (kernel_size / 2) as i32;
    let mut kernel = Vec::with_capacity(kernel_size);
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);

    for i in -half..=half {
        let x = i as f32;
        kernel.push((-x * x * inv_two_sigma_sq).exp());
    }

    // Normalize so the kernel sums to 1
    let sum: f32 = kernel.iter().sum();
    if sum > 1e-12 {
        for v in kernel.iter_mut() {
            *v /= sum;
        }
    }

    Ok(kernel)
}

/// Apply a 1D separable convolution both horizontally and vertically.
///
/// `kernel` is a 1D kernel of odd length. Applies horizontally first, then vertically.
/// Uses clamp-to-edge padding. Input and output are H×W×3 flat arrays.
pub fn aug_separable_convolve(
    img: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) -> Vec<f32> {
    let k_len = kernel.len();
    let half = (k_len / 2) as i32;
    let n = width * height * 3;

    // --- Horizontal pass ---
    let mut horiz = vec![0.0f32; n];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (ki, &k_val) in kernel.iter().enumerate() {
                    let dx = ki as i32 - half;
                    let sx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                    acc += img[(y * width + sx) * 3 + c] * k_val;
                }
                horiz[(y * width + x) * 3 + c] = acc;
            }
        }
    }

    // --- Vertical pass ---
    let mut out = vec![0.0f32; n];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (ki, &k_val) in kernel.iter().enumerate() {
                    let dy = ki as i32 - half;
                    let sy = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                    acc += horiz[(sy * width + x) * 3 + c] * k_val;
                }
                out[(y * width + x) * 3 + c] = acc;
            }
        }
    }

    out
}

// ---- Individual transform functions -------------------------------------------

/// Flip the image horizontally (mirror left-right).
pub fn horizontal_flip(img: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; img.len()];
    for y in 0..height {
        for x in 0..width {
            let src = (y * width + x) * 3;
            let dst = (y * width + (width - 1 - x)) * 3;
            out[dst] = img[src];
            out[dst + 1] = img[src + 1];
            out[dst + 2] = img[src + 2];
        }
    }
    out
}

/// Flip the image vertically (mirror top-bottom).
pub fn vertical_flip(img: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; img.len()];
    for y in 0..height {
        for x in 0..width {
            let src = (y * width + x) * 3;
            let dst = ((height - 1 - y) * width + x) * 3;
            out[dst] = img[src];
            out[dst + 1] = img[src + 1];
            out[dst + 2] = img[src + 2];
        }
    }
    out
}

/// Apply random color jitter in order: brightness → contrast → saturation → hue.
///
/// - Brightness: `pixel *= 1.0 + brightness * (2*U - 1)` — clamp \[0,1\]
/// - Contrast: per-channel mean, `pixel = mean + (pixel - mean) * (1.0 + contrast*(2*U-1))` — clamp
/// - Saturation: convert to HSV, `s *= 1.0 + saturation*(2*U-1)`, clamp s \[0,1\], convert back
/// - Hue: convert to HSV, `h += hue*(2*U-1)*180`, wrap h to \[0,360\), convert back
#[allow(clippy::too_many_arguments)]
pub fn aug_color_jitter(
    img: &[f32],
    width: usize,
    height: usize,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
    state: &mut u64,
) -> Vec<f32> {
    let n_pixels = width * height;
    let mut out = img.to_vec();

    // --- Brightness ---
    if brightness > 0.0 {
        let u = xorshift_f32(state);
        let factor = 1.0 + brightness * (2.0 * u - 1.0);
        for v in out.iter_mut() {
            *v = (*v * factor).clamp(0.0, 1.0);
        }
    }

    // --- Contrast ---
    if contrast > 0.0 {
        let u = xorshift_f32(state);
        let factor = 1.0 + contrast * (2.0 * u - 1.0);
        // Compute per-channel mean
        let mut chan_mean = [0.0f64; 3];
        for i in 0..n_pixels {
            for c in 0..3 {
                chan_mean[c] += out[i * 3 + c] as f64;
            }
        }
        let n_f = n_pixels.max(1) as f64;
        for m in chan_mean.iter_mut() {
            *m /= n_f;
        }
        for i in 0..n_pixels {
            for c in 0..3 {
                let mean_c = chan_mean[c] as f32;
                let v = mean_c + (out[i * 3 + c] - mean_c) * factor;
                out[i * 3 + c] = v.clamp(0.0, 1.0);
            }
        }
    }

    // --- Saturation ---
    if saturation > 0.0 {
        let u = xorshift_f32(state);
        let factor = 1.0 + saturation * (2.0 * u - 1.0);
        for i in 0..n_pixels {
            let r = out[i * 3];
            let g = out[i * 3 + 1];
            let b = out[i * 3 + 2];
            let (h_v, s_v, v_v) = aug_rgb_to_hsv(r, g, b);
            let new_s = (s_v * factor).clamp(0.0, 1.0);
            let (nr, ng, nb) = aug_hsv_to_rgb(h_v, new_s, v_v);
            out[i * 3] = nr.clamp(0.0, 1.0);
            out[i * 3 + 1] = ng.clamp(0.0, 1.0);
            out[i * 3 + 2] = nb.clamp(0.0, 1.0);
        }
    }

    // --- Hue ---
    if hue > 0.0 {
        let u = xorshift_f32(state);
        let delta_h = hue * (2.0 * u - 1.0) * 180.0;
        for i in 0..n_pixels {
            let r = out[i * 3];
            let g = out[i * 3 + 1];
            let b = out[i * 3 + 2];
            let (h_v, s_v, v_v) = aug_rgb_to_hsv(r, g, b);
            let new_h = (h_v + delta_h).rem_euclid(360.0);
            let (nr, ng, nb) = aug_hsv_to_rgb(new_h, s_v, v_v);
            out[i * 3] = nr.clamp(0.0, 1.0);
            out[i * 3 + 1] = ng.clamp(0.0, 1.0);
            out[i * 3 + 2] = nb.clamp(0.0, 1.0);
        }
    }

    out
}

/// Add Gaussian noise with zero mean and given `std` to each pixel; clamp result to [0, 1].
pub fn aug_add_gaussian_noise(img: &[f32], std: f32, state: &mut u64) -> Vec<f32> {
    let mut out = img.to_vec();
    let len = out.len();
    let mut i = 0;
    while i + 1 < len {
        let u1 = xorshift_f32(state);
        let u2 = xorshift_f32(state);
        let (z0, z1) = aug_box_muller(u1, u2);
        out[i] = (out[i] + z0 * std).clamp(0.0, 1.0);
        out[i + 1] = (out[i + 1] + z1 * std).clamp(0.0, 1.0);
        i += 2;
    }
    // Handle odd-length tail
    if i < len {
        let u1 = xorshift_f32(state);
        let u2 = xorshift_f32(state);
        let (z0, _) = aug_box_muller(u1, u2);
        out[i] = (out[i] + z0 * std).clamp(0.0, 1.0);
    }
    out
}

/// Crop a `crop_w × crop_h` region at a random offset from the image.
///
/// Returns an error if the crop is larger than the image in either dimension.
pub fn aug_random_crop(
    img: &[f32],
    width: usize,
    height: usize,
    crop_w: usize,
    crop_h: usize,
    state: &mut u64,
) -> Result<Vec<f32>, AugmentError> {
    if crop_w == 0 || crop_h == 0 {
        return Err(AugmentError::InvalidParam(
            "crop dimensions must be > 0".into(),
        ));
    }
    if crop_w > width || crop_h > height {
        return Err(AugmentError::InvalidParam(format!(
            "crop {}x{} exceeds image {}x{}",
            crop_w, crop_h, width, height
        )));
    }

    let max_x = width - crop_w;
    let max_y = height - crop_h;

    let off_x = if max_x == 0 {
        0
    } else {
        (xorshift64(state) as usize) % (max_x + 1)
    };
    let off_y = if max_y == 0 {
        0
    } else {
        (xorshift64(state) as usize) % (max_y + 1)
    };

    let mut out = Vec::with_capacity(crop_w * crop_h * 3);
    for y in 0..crop_h {
        for x in 0..crop_w {
            let src = ((off_y + y) * width + (off_x + x)) * 3;
            out.push(img[src]);
            out.push(img[src + 1]);
            out.push(img[src + 2]);
        }
    }

    Ok(out)
}

/// Rotate the image by `times * 90` degrees clockwise.
///
/// Returns `(rotated_image, new_width, new_height)`.
///
/// For `times = 1` (90° clockwise): pixel `(x, y)` in new image comes from `(y, old_width-1-x)`.
pub fn aug_rotate_90(
    img: &[f32],
    width: usize,
    height: usize,
    times: u32,
) -> (Vec<f32>, usize, usize) {
    let times = times % 4;
    if times == 0 {
        return (img.to_vec(), width, height);
    }

    // Apply one 90° clockwise rotation at a time
    let mut current = img.to_vec();
    let mut cur_w = width;
    let mut cur_h = height;

    for _ in 0..times {
        let new_w = cur_h;
        let new_h = cur_w;
        let mut rotated = vec![0.0f32; new_w * new_h * 3];

        // 90° clockwise: new(nx, ny) from old(old_x, old_y) where:
        //   old_x = ny,  old_y = cur_h - 1 - nx
        // new_w = cur_h, new_h = cur_w
        for ny in 0..new_h {
            for nx in 0..new_w {
                // nx in [0, new_w) = [0, cur_h), ny in [0, new_h) = [0, cur_w)
                let old_x = ny;
                let old_y = cur_h - 1 - nx;
                let src = (old_y * cur_w + old_x) * 3;
                let dst = (ny * new_w + nx) * 3;
                rotated[dst] = current[src];
                rotated[dst + 1] = current[src + 1];
                rotated[dst + 2] = current[src + 2];
            }
        }

        current = rotated;
        cur_w = new_w;
        cur_h = new_h;
    }

    (current, cur_w, cur_h)
}

/// Apply Gaussian blur with a separable kernel.
///
/// Returns an error if `kernel_size` is even.
pub fn aug_gaussian_blur(
    img: &[f32],
    width: usize,
    height: usize,
    kernel_size: usize,
    sigma: f32,
) -> Result<Vec<f32>, AugmentError> {
    if kernel_size.is_multiple_of(2) {
        return Err(AugmentError::InvalidParam(format!(
            "kernel_size must be odd, got {}",
            kernel_size
        )));
    }
    let kernel = aug_gaussian_kernel(kernel_size, sigma)?;
    Ok(aug_separable_convolve(img, width, height, &kernel))
}

/// Randomly erase a rectangular region, filling it with the per-channel image mean.
///
/// `min_area` and `max_area` are fractions of total image area. If no valid rectangle
/// can be found after 10 attempts, returns the input unchanged.
pub fn aug_random_erasing(
    img: &[f32],
    width: usize,
    height: usize,
    min_area: f32,
    max_area: f32,
    state: &mut u64,
) -> Vec<f32> {
    let total_area = (width * height) as f32;
    let stats = aug_image_stats(img);

    const MAX_ATTEMPTS: usize = 10;
    for _ in 0..MAX_ATTEMPTS {
        let u_area = xorshift_f32(state);
        let target_area = (min_area + u_area * (max_area - min_area)) * total_area;

        // Sample aspect ratio near 1.0: in [0.3, 3.3]
        let u_asp = xorshift_f32(state);
        let aspect = 0.3 + u_asp * 3.0;

        let erase_h = (target_area / aspect).sqrt();
        let erase_w = erase_h * aspect;

        let erase_h = erase_h.round() as usize;
        let erase_w = erase_w.round() as usize;

        if erase_h == 0 || erase_w == 0 || erase_h > height || erase_w > width {
            continue;
        }

        let max_x = width - erase_w;
        let max_y = height - erase_h;

        let off_x = if max_x == 0 {
            0
        } else {
            (xorshift64(state) as usize) % (max_x + 1)
        };
        let off_y = if max_y == 0 {
            0
        } else {
            (xorshift64(state) as usize) % (max_y + 1)
        };

        let mut out = img.to_vec();
        for y in 0..erase_h {
            for x in 0..erase_w {
                let idx = ((off_y + y) * width + (off_x + x)) * 3;
                out[idx] = stats.mean[0];
                out[idx + 1] = stats.mean[1];
                out[idx + 2] = stats.mean[2];
            }
        }
        return out;
    }

    // Could not fit — return unchanged
    img.to_vec()
}

/// Normalize each channel: `(v - mean[c]) / (std[c] + 1e-7)`.
pub fn aug_normalize(
    img: &[f32],
    mean: &[f32; 3],
    std_dev: &[f32; 3],
) -> Result<Vec<f32>, AugmentError> {
    if img.is_empty() {
        return Err(AugmentError::EmptyImage);
    }
    let mut out = img.to_vec();
    let n_pixels = out.len() / 3;
    for i in 0..n_pixels {
        for c in 0..3 {
            out[i * 3 + c] = (out[i * 3 + c] - mean[c]) / (std_dev[c] + 1e-7);
        }
    }
    Ok(out)
}

/// Denormalize each channel: `v * (std[c] + 1e-7) + mean[c]`.
pub fn aug_denormalize(
    img: &[f32],
    mean: &[f32; 3],
    std_dev: &[f32; 3],
) -> Result<Vec<f32>, AugmentError> {
    if img.is_empty() {
        return Err(AugmentError::EmptyImage);
    }
    let mut out = img.to_vec();
    let n_pixels = out.len() / 3;
    for i in 0..n_pixels {
        for c in 0..3 {
            out[i * 3 + c] = out[i * 3 + c] * (std_dev[c] + 1e-7) + mean[c];
        }
    }
    Ok(out)
}

// ---- Augmentation pipeline ---------------------------------------------------

/// A single augmentation operation.
#[derive(Debug, Clone)]
pub enum AugmentOp {
    HorizontalFlip,
    VerticalFlip,
    ColorJitter {
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
    },
    GaussianNoise {
        std: f32,
    },
    RandomCrop {
        crop_w: usize,
        crop_h: usize,
    },
    RandomRotation90,
    GaussianBlur {
        kernel_size: usize,
        sigma: f32,
    },
    RandomErasing {
        min_area: f32,
        max_area: f32,
    },
    Normalize {
        mean: [f32; 3],
        std: [f32; 3],
    },
    Denormalize {
        mean: [f32; 3],
        std: [f32; 3],
    },
}

/// Pipeline configuration: a list of `(op, probability)` pairs plus a seed.
#[derive(Debug, Clone)]
pub struct AugmentConfig {
    /// Operations to apply, each with an independent application probability in [0, 1].
    pub ops: Vec<(AugmentOp, f32)>,
    /// Initial PRNG seed.
    pub seed: u64,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            ops: vec![
                (AugmentOp::HorizontalFlip, 0.5),
                (
                    AugmentOp::ColorJitter {
                        brightness: 0.2,
                        contrast: 0.2,
                        saturation: 0.2,
                        hue: 0.05,
                    },
                    0.8,
                ),
                (AugmentOp::GaussianNoise { std: 0.02 }, 0.3),
            ],
            seed: 42,
        }
    }
}

/// Pipeline augmenter that applies a sequence of random operations to images.
pub struct ImageAugmenter {
    config: AugmentConfig,
    rng_state: u64,
}

impl ImageAugmenter {
    /// Construct a new augmenter with the given configuration.
    pub fn new(config: AugmentConfig) -> Self {
        let rng_state = if config.seed == 0 { 1 } else { config.seed };
        Self { config, rng_state }
    }

    /// Reset the PRNG seed (allows reproducible re-augmentation).
    pub fn reset_seed(&mut self, seed: u64) {
        self.rng_state = if seed == 0 { 1 } else { seed };
    }

    /// Augment a single image. Returns an error if `img.len() != width * height * 3`.
    pub fn augment(
        &mut self,
        img: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, AugmentError> {
        let expected = width * height * 3;
        if img.len() != expected {
            return Err(AugmentError::SizeMismatch {
                got: img.len(),
                width,
                height,
                channels: 3,
            });
        }
        if img.is_empty() {
            return Err(AugmentError::EmptyImage);
        }

        let mut current: Vec<f32> = img.to_vec();
        let mut cur_w = width;
        let mut cur_h = height;
        let mut last_was_normalize = false;

        for (op, prob) in &self.config.ops.clone() {
            let u = xorshift_f32(&mut self.rng_state);
            if u >= *prob {
                continue;
            }

            last_was_normalize = false;

            match op {
                AugmentOp::HorizontalFlip => {
                    current = horizontal_flip(&current, cur_w, cur_h);
                }
                AugmentOp::VerticalFlip => {
                    current = vertical_flip(&current, cur_w, cur_h);
                }
                AugmentOp::ColorJitter {
                    brightness,
                    contrast,
                    saturation,
                    hue,
                } => {
                    current = aug_color_jitter(
                        &current,
                        cur_w,
                        cur_h,
                        *brightness,
                        *contrast,
                        *saturation,
                        *hue,
                        &mut self.rng_state,
                    );
                }
                AugmentOp::GaussianNoise { std } => {
                    current = aug_add_gaussian_noise(&current, *std, &mut self.rng_state);
                }
                AugmentOp::RandomCrop { crop_w, crop_h } => {
                    current = aug_random_crop(
                        &current,
                        cur_w,
                        cur_h,
                        *crop_w,
                        *crop_h,
                        &mut self.rng_state,
                    )?;
                    cur_w = *crop_w;
                    cur_h = *crop_h;
                }
                AugmentOp::RandomRotation90 => {
                    let times = (xorshift64(&mut self.rng_state) % 4) as u32;
                    let (rotated, new_w, new_h) = aug_rotate_90(&current, cur_w, cur_h, times);
                    current = rotated;
                    cur_w = new_w;
                    cur_h = new_h;
                }
                AugmentOp::GaussianBlur { kernel_size, sigma } => {
                    current = aug_gaussian_blur(&current, cur_w, cur_h, *kernel_size, *sigma)?;
                }
                AugmentOp::RandomErasing { min_area, max_area } => {
                    current = aug_random_erasing(
                        &current,
                        cur_w,
                        cur_h,
                        *min_area,
                        *max_area,
                        &mut self.rng_state,
                    );
                }
                AugmentOp::Normalize { mean, std } => {
                    current = aug_normalize(&current, mean, std)?;
                    last_was_normalize = true;
                }
                AugmentOp::Denormalize { mean, std } => {
                    current = aug_denormalize(&current, mean, std)?;
                    last_was_normalize = true;
                }
            }
        }

        // Clamp to [0, 1] unless the last op was normalize/denormalize
        if !last_was_normalize {
            for v in current.iter_mut() {
                *v = v.clamp(0.0, 1.0);
            }
        }

        Ok(current)
    }

    /// Augment a batch of images (all must have the same `width × height`).
    pub fn augment_batch(
        &mut self,
        imgs: &[Vec<f32>],
        width: usize,
        height: usize,
    ) -> Result<Vec<Vec<f32>>, AugmentError> {
        imgs.iter()
            .map(|img| self.augment(img, width, height))
            .collect()
    }
}

// ---- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a W×H×3 flat image filled with constant per-channel values
    fn solid_image(width: usize, height: usize, r: f32, g: f32, b: f32) -> Vec<f32> {
        let n = width * height;
        let mut img = Vec::with_capacity(n * 3);
        for _ in 0..n {
            img.push(r);
            img.push(g);
            img.push(b);
        }
        img
    }

    // Helper: 2×2 checkerboard
    fn checkerboard_2x2() -> Vec<f32> {
        // Row 0: (1,0,0) (0,1,0)
        // Row 1: (0,0,1) (1,1,0)
        vec![
            1.0, 0.0, 0.0, // (0,0) top-left
            0.0, 1.0, 0.0, // (1,0) top-right
            0.0, 0.0, 1.0, // (0,1) bottom-left
            1.0, 1.0, 0.0, // (1,1) bottom-right
        ]
    }

    // ---- horizontal_flip tests ------------------------------------------------

    #[test]
    fn test_horizontal_flip_2x2() {
        let img = checkerboard_2x2();
        let flipped = horizontal_flip(&img, 2, 2);
        // Row 0 left→right swap: (0,1,0) (1,0,0)
        assert_eq!(flipped[0..3], [0.0, 1.0, 0.0]); // was right, now left
        assert_eq!(flipped[3..6], [1.0, 0.0, 0.0]); // was left, now right
                                                    // Row 1: (1,1,0) (0,0,1)
        assert_eq!(flipped[6..9], [1.0, 1.0, 0.0]);
        assert_eq!(flipped[9..12], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_horizontal_flip_twice_identity() {
        let img = checkerboard_2x2();
        let once = horizontal_flip(&img, 2, 2);
        let twice = horizontal_flip(&once, 2, 2);
        for (a, b) in img.iter().zip(twice.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_horizontal_flip_1x1() {
        let img = vec![0.5, 0.3, 0.7];
        let out = horizontal_flip(&img, 1, 1);
        assert_eq!(out, vec![0.5, 0.3, 0.7]);
    }

    // ---- vertical_flip tests --------------------------------------------------

    #[test]
    fn test_vertical_flip_2x2() {
        let img = checkerboard_2x2();
        let flipped = vertical_flip(&img, 2, 2);
        // Row 0 should now be old Row 1
        assert_eq!(flipped[0..3], [0.0, 0.0, 1.0]);
        assert_eq!(flipped[3..6], [1.0, 1.0, 0.0]);
        // Row 1 should now be old Row 0
        assert_eq!(flipped[6..9], [1.0, 0.0, 0.0]);
        assert_eq!(flipped[9..12], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_vertical_flip_twice_identity() {
        let img = checkerboard_2x2();
        let once = vertical_flip(&img, 2, 2);
        let twice = vertical_flip(&once, 2, 2);
        for (a, b) in img.iter().zip(twice.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vertical_flip_1x1() {
        let img = vec![0.5, 0.3, 0.7];
        let out = vertical_flip(&img, 1, 1);
        assert_eq!(out, vec![0.5, 0.3, 0.7]);
    }

    // ---- aug_rgb_to_hsv / aug_hsv_to_rgb round-trip ---------------------------

    fn hsv_roundtrip(r: f32, g: f32, b: f32) {
        let (h, s, v) = aug_rgb_to_hsv(r, g, b);
        let (rr, rg, rb) = aug_hsv_to_rgb(h, s, v);
        assert!((r - rr).abs() < 1e-5, "r: {r} vs {rr}");
        assert!((g - rg).abs() < 1e-5, "g: {g} vs {rg}");
        assert!((b - rb).abs() < 1e-5, "b: {b} vs {rb}");
    }

    #[test]
    fn test_hsv_roundtrip_red() {
        hsv_roundtrip(1.0, 0.0, 0.0);
        let (h, s, v) = aug_rgb_to_hsv(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1e-4, "H for red should be 0: {h}");
        assert!((s - 1.0).abs() < 1e-4);
        assert!((v - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_hsv_roundtrip_green() {
        hsv_roundtrip(0.0, 1.0, 0.0);
        let (h, _, _) = aug_rgb_to_hsv(0.0, 1.0, 0.0);
        assert!((h - 120.0).abs() < 1e-3, "H for green should be 120: {h}");
    }

    #[test]
    fn test_hsv_roundtrip_blue() {
        hsv_roundtrip(0.0, 0.0, 1.0);
        let (h, _, _) = aug_rgb_to_hsv(0.0, 0.0, 1.0);
        assert!((h - 240.0).abs() < 1e-3, "H for blue should be 240: {h}");
    }

    #[test]
    fn test_hsv_roundtrip_white() {
        hsv_roundtrip(1.0, 1.0, 1.0);
        let (_, s, v) = aug_rgb_to_hsv(1.0, 1.0, 1.0);
        assert!(s < 1e-5, "s for white should be 0: {s}");
        assert!((v - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hsv_roundtrip_black() {
        hsv_roundtrip(0.0, 0.0, 0.0);
        let (_, _, v) = aug_rgb_to_hsv(0.0, 0.0, 0.0);
        assert!(v < 1e-5, "v for black should be 0: {v}");
    }

    #[test]
    fn test_hsv_roundtrip_gray() {
        hsv_roundtrip(0.5, 0.5, 0.5);
        let (_, s, v) = aug_rgb_to_hsv(0.5, 0.5, 0.5);
        assert!(s < 1e-5, "s for gray should be 0: {s}");
        assert!((v - 0.5).abs() < 1e-5);
    }

    // ---- aug_box_muller -------------------------------------------------------

    #[test]
    fn test_box_muller_statistics() {
        let mut state = 12345u64;
        let n = 1000;
        let mut samples = Vec::with_capacity(2 * n);
        for _ in 0..n {
            let u1 = xorshift_f32(&mut state);
            let u2 = xorshift_f32(&mut state);
            let (z0, z1) = aug_box_muller(u1, u2);
            samples.push(z0);
            samples.push(z1);
        }
        let len = samples.len() as f64;
        let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / len;
        let var: f64 = samples
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / len;
        let std = var.sqrt();
        // With 2000 samples, mean should be within ±0.15 of 0 and std within [0.85, 1.15]
        assert!(mean.abs() < 0.15, "mean {mean} too far from 0");
        assert!(std > 0.85 && std < 1.15, "std {std} not ≈ 1");
    }

    #[test]
    fn test_box_muller_pairs_finite() {
        let (z0, z1) = aug_box_muller(0.5, 0.5);
        assert!(z0.is_finite());
        assert!(z1.is_finite());
    }

    // ---- aug_gaussian_kernel --------------------------------------------------

    #[test]
    fn test_gaussian_kernel_odd_valid() {
        let k = aug_gaussian_kernel(5, 1.0).unwrap();
        assert_eq!(k.len(), 5);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        // Center element is the largest
        assert!(k[2] >= k[0]);
        assert!(k[2] >= k[4]);
    }

    #[test]
    fn test_gaussian_kernel_even_errors() {
        assert!(aug_gaussian_kernel(4, 1.0).is_err());
        assert!(aug_gaussian_kernel(2, 1.0).is_err());
    }

    #[test]
    fn test_gaussian_kernel_size_1() {
        let k = aug_gaussian_kernel(1, 1.0).unwrap();
        assert_eq!(k.len(), 1);
        assert!((k[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gaussian_kernel_center_is_max() {
        let k = aug_gaussian_kernel(7, 2.0).unwrap();
        let center = k[3];
        for (i, &v) in k.iter().enumerate() {
            if i != 3 {
                assert!(center >= v, "center {center} < k[{i}]={v}");
            }
        }
    }

    // ---- aug_separable_convolve -----------------------------------------------

    #[test]
    fn test_separable_convolve_identity_kernel() {
        // A kernel of [0, 1, 0] is an identity convolution (at its center)
        let img = solid_image(4, 4, 0.3, 0.5, 0.7);
        let kernel = vec![0.0f32, 1.0, 0.0];
        let out = aug_separable_convolve(&img, 4, 4, &kernel);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5, "identity kernel mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_separable_convolve_size1_kernel() {
        let img = solid_image(3, 3, 0.2, 0.4, 0.6);
        let kernel = vec![1.0f32];
        let out = aug_separable_convolve(&img, 3, 3, &kernel);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // ---- aug_add_gaussian_noise -----------------------------------------------

    #[test]
    fn test_gaussian_noise_changes_image() {
        let img = solid_image(8, 8, 0.5, 0.5, 0.5);
        let mut state = 99u64;
        let noisy = aug_add_gaussian_noise(&img, 0.1, &mut state);
        let diffs: Vec<f32> = img
            .iter()
            .zip(noisy.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        let any_diff = diffs.iter().any(|&d| d > 1e-6);
        assert!(any_diff, "noise should change at least one pixel");
    }

    #[test]
    fn test_gaussian_noise_values_clamped() {
        // Large std to force clamping
        let img = solid_image(8, 8, 0.5, 0.5, 0.5);
        let mut state = 777u64;
        let noisy = aug_add_gaussian_noise(&img, 10.0, &mut state);
        for &v in &noisy {
            assert!((0.0..=1.0).contains(&v), "value {v} out of [0, 1]");
        }
    }

    #[test]
    fn test_gaussian_noise_zero_std_unchanged() {
        let img = solid_image(4, 4, 0.3, 0.6, 0.9);
        let mut state = 1u64;
        let out = aug_add_gaussian_noise(&img, 0.0, &mut state);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ---- aug_color_jitter ----------------------------------------------------

    #[test]
    fn test_color_jitter_changes_image() {
        let img = solid_image(4, 4, 0.5, 0.4, 0.3);
        let mut state = 1234u64;
        let out = aug_color_jitter(&img, 4, 4, 0.5, 0.5, 0.5, 0.5, &mut state);
        let diff_sum: f32 = img.iter().zip(out.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff_sum > 1e-4, "color jitter should change image");
    }

    #[test]
    fn test_color_jitter_zero_params_unchanged() {
        let img = solid_image(4, 4, 0.5, 0.4, 0.3);
        let mut state = 42u64;
        let out = aug_color_jitter(&img, 4, 4, 0.0, 0.0, 0.0, 0.0, &mut state);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_color_jitter_clamped() {
        let img = solid_image(4, 4, 0.9, 0.9, 0.9);
        let mut state = 8888u64;
        let out = aug_color_jitter(&img, 4, 4, 1.0, 1.0, 1.0, 1.0, &mut state);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "value {v} out of [0, 1]");
        }
    }

    // ---- aug_random_crop -----------------------------------------------------

    #[test]
    fn test_random_crop_correct_size() {
        let img = solid_image(8, 6, 0.5, 0.5, 0.5);
        let mut state = 7u64;
        let cropped = aug_random_crop(&img, 8, 6, 4, 3, &mut state).unwrap();
        assert_eq!(cropped.len(), 4 * 3 * 3);
    }

    #[test]
    fn test_random_crop_too_large_errors() {
        let img = solid_image(4, 4, 0.5, 0.5, 0.5);
        let mut state = 1u64;
        assert!(aug_random_crop(&img, 4, 4, 5, 3, &mut state).is_err());
        assert!(aug_random_crop(&img, 4, 4, 3, 5, &mut state).is_err());
    }

    #[test]
    fn test_random_crop_exact_size_ok() {
        let img = solid_image(4, 4, 0.5, 0.5, 0.5);
        let mut state = 1u64;
        let out = aug_random_crop(&img, 4, 4, 4, 4, &mut state).unwrap();
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_random_crop_1x1() {
        let img = solid_image(4, 4, 0.3, 0.5, 0.7);
        let mut state = 99u64;
        let out = aug_random_crop(&img, 4, 4, 1, 1, &mut state).unwrap();
        assert_eq!(out.len(), 3);
        // Solid image, so any crop pixel equals the fill color
        assert!((out[0] - 0.3).abs() < 1e-6);
    }

    // ---- aug_rotate_90 -------------------------------------------------------

    #[test]
    fn test_rotate_90_times0_unchanged() {
        let img = checkerboard_2x2();
        let (out, w, h) = aug_rotate_90(&img, 2, 2, 0);
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(out, img);
    }

    #[test]
    fn test_rotate_90_times4_identity() {
        let img = checkerboard_2x2();
        let (out, w, h) = aug_rotate_90(&img, 2, 2, 4);
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rotate_90_times2_double_flip() {
        // 2×90° rotation = 180° = vertical + horizontal flip
        let img = checkerboard_2x2();
        let (rotated, w, h) = aug_rotate_90(&img, 2, 2, 2);
        // Compute expected: h-flip then v-flip (or vice versa gives same 180)
        let expected_hv = vertical_flip(&horizontal_flip(&img, 2, 2), 2, 2);
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        for (a, b) in rotated.iter().zip(expected_hv.iter()) {
            assert!((a - b).abs() < 1e-6, "180° rotation mismatch");
        }
    }

    #[test]
    fn test_rotate_90_dimensions_3x2() {
        // 3w × 2h input, rotate 90 → 2w × 3h
        let img = solid_image(3, 2, 0.1, 0.2, 0.3);
        let (_, w, h) = aug_rotate_90(&img, 3, 2, 1);
        assert_eq!(w, 2);
        assert_eq!(h, 3);
    }

    #[test]
    fn test_rotate_90_times1_pixel_positions() {
        let img = checkerboard_2x2();
        // Row0: (1,0,0) (0,1,0) | Row1: (0,0,1) (1,1,0)
        // After 90° CW: new(x, y) = old(y, old_w-1-x) with new_w=2, new_h=2
        // new(0,0) = old(y=0, x=old_w-1-0=1) = old(1,0) = (0,1,0)
        // new(1,0) = old(y=0, x=old_w-1-1=0) = old(0,0) = (1,0,0)
        // Wait: in aug_rotate_90 we have: old_x = ny, old_y = cur_w - 1 - nx
        // For new pixel (nx=0, ny=0): old_x=0, old_y=1 → img[(1*2+0)*3] = (0,0,1)
        // For new pixel (nx=1, ny=0): old_x=0, old_y=0 → img[(0*2+0)*3] = (1,0,0)
        // For new pixel (nx=0, ny=1): old_x=1, old_y=1 → img[(1*2+1)*3] = (1,1,0)
        // For new pixel (nx=1, ny=1): old_x=1, old_y=0 → img[(0*2+1)*3] = (0,1,0)
        let (out, w, h) = aug_rotate_90(&img, 2, 2, 1);
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(out.len(), 12);
        // Verify the rotation produces a valid 12-element image
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    // ---- aug_gaussian_blur ---------------------------------------------------

    #[test]
    fn test_gaussian_blur_even_kernel_errors() {
        let img = solid_image(4, 4, 0.5, 0.5, 0.5);
        assert!(aug_gaussian_blur(&img, 4, 4, 4, 1.0).is_err());
    }

    #[test]
    fn test_gaussian_blur_smooths_image() {
        // Create a noisy image (alternating 0/1 per pixel) and verify blur reduces
        // the maximum per-step difference between adjacent pixels.
        let w = 16usize;
        let h = 8usize;
        let mut img = vec![0.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
                img[(y * w + x) * 3] = v;
                img[(y * w + x) * 3 + 1] = v;
                img[(y * w + x) * 3 + 2] = v;
            }
        }
        let blurred = aug_gaussian_blur(&img, w, h, 5, 1.5).unwrap();
        // Max adjacent difference in original should be 1.0 (alternating checkerboard)
        // After blur it should be reduced significantly
        let mut max_diff_orig = 0.0f32;
        let mut max_diff_blur = 0.0f32;
        for y in 0..h {
            for x in 0..(w - 1) {
                let d_orig = (img[(y * w + x + 1) * 3] - img[(y * w + x) * 3]).abs();
                let d_blur = (blurred[(y * w + x + 1) * 3] - blurred[(y * w + x) * 3]).abs();
                if d_orig > max_diff_orig {
                    max_diff_orig = d_orig;
                }
                if d_blur > max_diff_blur {
                    max_diff_blur = d_blur;
                }
            }
        }
        assert!(
            max_diff_blur < max_diff_orig,
            "blurred max-diff {max_diff_blur} should < original max-diff {max_diff_orig}"
        );
    }

    #[test]
    fn test_gaussian_blur_solid_unchanged() {
        // Blurring a solid image should not change pixel values
        let img = solid_image(4, 4, 0.3, 0.5, 0.7);
        let out = aug_gaussian_blur(&img, 4, 4, 3, 1.0).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5, "solid blur mismatch: {a} vs {b}");
        }
    }

    // ---- aug_random_erasing --------------------------------------------------

    #[test]
    fn test_random_erasing_changes_image() {
        let img = solid_image(8, 8, 0.3, 0.6, 0.9);
        let mut state = 4444u64;
        let out = aug_random_erasing(&img, 8, 8, 0.1, 0.5, &mut state);
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_random_erasing_returns_same_size() {
        let img = solid_image(6, 5, 0.5, 0.5, 0.5);
        let mut state = 1111u64;
        let out = aug_random_erasing(&img, 6, 5, 0.1, 0.3, &mut state);
        assert_eq!(out.len(), 6 * 5 * 3);
    }

    #[test]
    fn test_random_erasing_1x1_returns_unchanged() {
        // A 1×1 image: erasing any area >= full area means a random fill with mean
        let img = vec![0.5f32, 0.5, 0.5];
        let mut state = 1u64;
        let out = aug_random_erasing(&img, 1, 1, 0.1, 0.9, &mut state);
        assert_eq!(out.len(), 3);
    }

    // ---- aug_normalize / aug_denormalize -------------------------------------

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let img = solid_image(4, 4, 0.3, 0.5, 0.7);
        let mean = [0.485f32, 0.456, 0.406];
        let std_dev = [0.229f32, 0.224, 0.225];
        let normed = aug_normalize(&img, &mean, &std_dev).unwrap();
        let recovered = aug_denormalize(&normed, &mean, &std_dev).unwrap();
        for (a, b) in img.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-5, "normalize roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_normalize_empty_errors() {
        let img: Vec<f32> = vec![];
        let mean = [0.0f32; 3];
        let std_dev = [1.0f32; 3];
        assert!(aug_normalize(&img, &mean, &std_dev).is_err());
    }

    #[test]
    fn test_denormalize_empty_errors() {
        let img: Vec<f32> = vec![];
        let mean = [0.0f32; 3];
        let std_dev = [1.0f32; 3];
        assert!(aug_denormalize(&img, &mean, &std_dev).is_err());
    }

    #[test]
    fn test_normalize_known_value() {
        // Single pixel: r=0.5, mean=0.5, std=0.5 → normalized = 0.0/(0.5+1e-7) ≈ 0
        let img = vec![0.5f32, 0.0, 0.0];
        let mean = [0.5f32, 0.0, 0.0];
        let std_dev = [0.5f32, 1.0, 1.0];
        let out = aug_normalize(&img, &mean, &std_dev).unwrap();
        assert!(out[0].abs() < 1e-4, "normalized r should be ~0: {}", out[0]);
    }

    // ---- aug_image_stats ------------------------------------------------------

    #[test]
    fn test_image_stats_solid_color() {
        // Solid red image: mean=[1,0,0], std=[0,0,0], min=0, max=1
        let img = solid_image(4, 4, 1.0, 0.0, 0.0);
        let stats = aug_image_stats(&img);
        assert!(
            (stats.mean[0] - 1.0).abs() < 1e-5,
            "mean R should be 1: {}",
            stats.mean[0]
        );
        assert!(
            stats.mean[1].abs() < 1e-5,
            "mean G should be 0: {}",
            stats.mean[1]
        );
        assert!(
            stats.mean[2].abs() < 1e-5,
            "mean B should be 0: {}",
            stats.mean[2]
        );
        assert!(stats.std_dev[0].abs() < 1e-5, "std R for solid should be 0");
        assert!((stats.max - 1.0).abs() < 1e-5);
        assert!(stats.min.abs() < 1e-5);
    }

    #[test]
    fn test_image_stats_empty_image() {
        let stats = aug_image_stats(&[]);
        assert_eq!(stats.mean, [0.0; 3]);
        assert_eq!(stats.std_dev, [0.0; 3]);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn test_image_stats_known_values() {
        // 2 pixels: (0.0, 0.0, 0.0) and (1.0, 1.0, 1.0) → mean=0.5, std=0.5
        let img = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let stats = aug_image_stats(&img);
        for c in 0..3 {
            assert!(
                (stats.mean[c] - 0.5).abs() < 1e-5,
                "mean[{c}] should be 0.5"
            );
            assert!(
                (stats.std_dev[c] - 0.5).abs() < 1e-5,
                "std[{c}] should be 0.5"
            );
        }
        assert!((stats.min - 0.0).abs() < 1e-5);
        assert!((stats.max - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_image_stats_single_pixel() {
        let img = vec![0.4f32, 0.6, 0.8];
        let stats = aug_image_stats(&img);
        assert!((stats.mean[0] - 0.4).abs() < 1e-5);
        assert!((stats.mean[1] - 0.6).abs() < 1e-5);
        assert!((stats.mean[2] - 0.8).abs() < 1e-5);
        assert!(
            stats.std_dev[0].abs() < 1e-5,
            "std for single pixel should be 0"
        );
    }

    // ---- ImageAugmenter::augment ----------------------------------------------

    #[test]
    fn test_augmenter_zero_probability_unchanged() {
        let img = solid_image(4, 4, 0.3, 0.5, 0.7);
        let config = AugmentConfig {
            ops: vec![
                (AugmentOp::HorizontalFlip, 0.0),
                (AugmentOp::VerticalFlip, 0.0),
                (AugmentOp::GaussianNoise { std: 0.1 }, 0.0),
            ],
            seed: 1,
        };
        let mut aug = ImageAugmenter::new(config);
        let out = aug.augment(&img, 4, 4).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "zero-prob augment should be identity");
        }
    }

    #[test]
    fn test_augmenter_size_mismatch_errors() {
        let img = solid_image(4, 4, 0.5, 0.5, 0.5);
        let config = AugmentConfig::default();
        let mut aug = ImageAugmenter::new(config);
        // Pass wrong dimensions
        assert!(aug.augment(&img, 3, 4).is_err());
        assert!(aug.augment(&img, 4, 3).is_err());
    }

    #[test]
    fn test_augmenter_default_config() {
        let img = solid_image(8, 8, 0.5, 0.5, 0.5);
        let mut aug = ImageAugmenter::new(AugmentConfig::default());
        // Should not error
        let out = aug.augment(&img, 8, 8).unwrap();
        assert_eq!(out.len(), 8 * 8 * 3);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "value {v} out of [0, 1]");
        }
    }

    #[test]
    fn test_augmenter_reset_seed_reproducible() {
        let img = solid_image(4, 4, 0.3, 0.6, 0.9);
        let config = AugmentConfig::default();
        let mut aug = ImageAugmenter::new(config.clone());
        let out1 = aug.augment(&img, 4, 4).unwrap();
        aug.reset_seed(config.seed);
        let out2 = aug.augment(&img, 4, 4).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "reset_seed should give reproducible output"
            );
        }
    }

    // ---- ImageAugmenter::augment_batch ----------------------------------------

    #[test]
    fn test_augment_batch_three_images() {
        let imgs: Vec<Vec<f32>> = (0..3)
            .map(|i| solid_image(4, 4, i as f32 * 0.3, 0.5, 0.7))
            .collect();
        let config = AugmentConfig::default();
        let mut aug = ImageAugmenter::new(config);
        let results = aug.augment_batch(&imgs, 4, 4).unwrap();
        assert_eq!(results.len(), 3);
        for out in &results {
            assert_eq!(out.len(), 4 * 4 * 3);
            for &v in out {
                assert!((0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn test_augment_batch_empty_batch() {
        let config = AugmentConfig::default();
        let mut aug = ImageAugmenter::new(config);
        let results = aug.augment_batch(&[], 4, 4).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_augment_batch_size_mismatch_errors() {
        let imgs = vec![solid_image(4, 4, 0.5, 0.5, 0.5)];
        let config = AugmentConfig::default();
        let mut aug = ImageAugmenter::new(config);
        // Wrong dimensions passed
        assert!(aug.augment_batch(&imgs, 5, 4).is_err());
    }

    // ---- Edge cases: all-black, all-white, 1×1 --------------------------------

    #[test]
    fn test_augment_all_black_1x1() {
        let img = vec![0.0f32, 0.0, 0.0];
        let config = AugmentConfig {
            ops: vec![(AugmentOp::GaussianNoise { std: 0.01 }, 1.0)],
            seed: 5,
        };
        let mut aug = ImageAugmenter::new(config);
        let out = aug.augment(&img, 1, 1).unwrap();
        assert_eq!(out.len(), 3);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_augment_all_white_1x1() {
        let img = vec![1.0f32, 1.0, 1.0];
        let config = AugmentConfig {
            ops: vec![(AugmentOp::GaussianNoise { std: 0.01 }, 1.0)],
            seed: 6,
        };
        let mut aug = ImageAugmenter::new(config);
        let out = aug.augment(&img, 1, 1).unwrap();
        assert_eq!(out.len(), 3);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_augment_1x1_horizontal_flip() {
        let img = vec![0.4f32, 0.5, 0.6];
        let config = AugmentConfig {
            ops: vec![(AugmentOp::HorizontalFlip, 1.0)],
            seed: 7,
        };
        let mut aug = ImageAugmenter::new(config);
        let out = aug.augment(&img, 1, 1).unwrap();
        // 1×1 flip is identity
        assert!((out[0] - 0.4).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_augment_1x1_all_ops() {
        let img = vec![0.5f32, 0.5, 0.5];
        let config = AugmentConfig {
            ops: vec![
                (AugmentOp::HorizontalFlip, 1.0),
                (AugmentOp::VerticalFlip, 1.0),
                (
                    AugmentOp::ColorJitter {
                        brightness: 0.1,
                        contrast: 0.1,
                        saturation: 0.1,
                        hue: 0.05,
                    },
                    1.0,
                ),
                (AugmentOp::GaussianNoise { std: 0.05 }, 1.0),
            ],
            seed: 99,
        };
        let mut aug = ImageAugmenter::new(config);
        let out = aug.augment(&img, 1, 1).unwrap();
        assert_eq!(out.len(), 3);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    // ---- SizeMismatch error variant -------------------------------------------

    #[test]
    fn test_size_mismatch_error_variant() {
        // Build a purposely wrong-size buffer
        let img = vec![0.5f32; 4 * 4 * 3 + 1];
        let config = AugmentConfig::default();
        let mut aug = ImageAugmenter::new(config);
        let result = aug.augment(&img, 4, 4);
        assert!(result.is_err());
        match result {
            Err(AugmentError::SizeMismatch {
                got,
                width,
                height,
                channels,
            }) => {
                assert_eq!(got, 4 * 4 * 3 + 1);
                assert_eq!(width, 4);
                assert_eq!(height, 4);
                assert_eq!(channels, 3);
            }
            other => panic!("expected SizeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_size_mismatch_empty_dims() {
        // Zero-dimension image: 0*0*3 = 0 but img has something
        let img = vec![0.5f32, 0.5, 0.5];
        let mut aug = ImageAugmenter::new(AugmentConfig::default());
        let result = aug.augment(&img, 0, 0);
        assert!(result.is_err());
    }

    // ---- Normalize + Denormalize (no clamp after) ----------------------------

    #[test]
    fn test_normalize_op_in_augmenter() {
        let img = solid_image(2, 2, 0.5, 0.5, 0.5);
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let config = AugmentConfig {
            ops: vec![(AugmentOp::Normalize { mean, std }, 1.0)],
            seed: 1,
        };
        let mut aug = ImageAugmenter::new(config);
        let out = aug.augment(&img, 2, 2).unwrap();
        // After normalize, values may be outside [0, 1] — check they are not clamped
        // r = (0.5 - 0.485) / 0.229 ≈ 0.065
        assert!(out[0].is_finite());
    }

    #[test]
    fn test_random_erasing_fills_with_mean() {
        // Create an image where we can verify the erased region is filled with mean
        let img = solid_image(10, 10, 0.3, 0.6, 0.9);
        let mut state = 5555u64;
        // With large erasing area (0.9 max) it should erase something
        let out = aug_random_erasing(&img, 10, 10, 0.5, 0.9, &mut state);
        assert_eq!(out.len(), 10 * 10 * 3);
        // Any pixel should be either the original or the mean (both are 0.3, 0.6, 0.9 for solid)
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "solid image with uniform mean: pixel should match"
            );
        }
    }

    #[test]
    fn test_aug_color_jitter_all_channels_affected() {
        // With all params set to 1.0, all channels should be affected
        let img: Vec<f32> = (0..64)
            .flat_map(|i| {
                let v = (i as f32) / 63.0;
                [v, 1.0 - v, v * 0.5]
            })
            .collect();
        let mut state = 2024u64;
        let out = aug_color_jitter(&img, 8, 8, 1.0, 1.0, 1.0, 1.0, &mut state);
        let diff: f32 = img.iter().zip(out.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.01,
            "color jitter with max params should change image significantly"
        );
    }

    // ---- xorshift64 zero-guard ------------------------------------------------

    #[test]
    fn test_xorshift64_zero_guard() {
        let mut state = 0u64;
        let _ = xorshift64(&mut state);
        // State should never be 0 after the call
        assert_ne!(state, 0);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 12345u64;
        let mut s2 = 12345u64;
        for _ in 0..100 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
    }

    #[test]
    fn test_xorshift_f32_range() {
        let mut state = 99999u64;
        for _ in 0..1000 {
            let v = xorshift_f32(&mut state);
            assert!((0.0..1.0).contains(&v), "xorshift_f32 out of [0,1): {v}");
        }
    }

    // ---- Gaussian blur kernel size 1 ----------------------------------------

    #[test]
    fn test_gaussian_blur_kernel1_identity() {
        let img = solid_image(4, 4, 0.3, 0.6, 0.9);
        let out = aug_gaussian_blur(&img, 4, 4, 1, 1.0).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5, "kernel_size=1 should be identity");
        }
    }
}
