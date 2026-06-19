//! # Latent Blend
//!
//! Latent space blending operations for the diffusion model. Enables smooth
//! interpolation between identity embeddings, expression states, and noise
//! trajectories entirely in latent space — no full forward/backward passes
//! required.
//!
//! ## Capabilities
//!
//! - **Basic blending**: lerp, slerp, weighted multi-blend, scaled addition
//! - **Spatial blending**: mask-based compositing with circular/gradient/morphological masks
//! - **Statistical harmonization**: mean/variance matching across latents
//! - **Frequency blending**: low/high-frequency content separation and recombination
//! - **Trajectory blending**: step-wise interpolation over noise schedules
//!
//! All standalone functions carry the `lb_` prefix to avoid name collisions
//! with other modules (e.g., `lerp` / `slerp` from `latent_interp`,
//! `fm_interpolate` from `flow_matching`).

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Private PRNG helpers (not re-exported — module-local only)
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

#[allow(dead_code)]
fn xorshift_f32(state: &mut u64) -> f32 {
    xorshift64(state) as f32 / u64::MAX as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by latent blend operations.
#[derive(Debug, Error, PartialEq)]
pub enum LatentBlendError {
    /// Two latents that must match in shape do not.
    #[error("dimension mismatch: {a} vs {b}")]
    DimensionMismatch { a: usize, b: usize },

    /// Operation received no elements.
    #[error("empty input")]
    EmptyInput,

    /// A blend weight is outside `[0, 1]`.
    #[error("invalid weight: must be in [0, 1], got {0}")]
    InvalidWeight(f32),

    /// An operation needs more latents than were supplied.
    #[error("not enough latents: need at least {needed}, got {got}")]
    NotEnoughLatents { needed: usize, got: usize },

    /// A configuration parameter is logically invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// A spatial mask has a different number of elements than the latent's spatial size.
    #[error("spatial mismatch: latent has {n} elements, mask has {m}")]
    SpatialMismatch { n: usize, m: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// LatentTensor
// ─────────────────────────────────────────────────────────────────────────────

/// A diffusion latent tensor stored as a flat `Vec<f32>`.
///
/// Layout: channel-first, i.e. element `(c, h, w)` lives at index
/// `c * height * width + h * width + w`.
#[derive(Debug, Clone)]
pub struct LatentTensor {
    /// Raw element data (length == `channels * height * width`).
    pub data: Vec<f32>,
    /// Number of channels.
    pub channels: usize,
    /// Spatial height.
    pub height: usize,
    /// Spatial width.
    pub width: usize,
}

impl LatentTensor {
    /// Construct a zero-filled latent of the given shape.
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            data: vec![0.0_f32; channels * height * width],
            channels,
            height,
            width,
        }
    }

    /// Construct from existing data, validating that the length matches the shape.
    pub fn from_data(
        data: Vec<f32>,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Self, LatentBlendError> {
        let expected = channels * height * width;
        if data.len() != expected {
            return Err(LatentBlendError::DimensionMismatch {
                a: data.len(),
                b: expected,
            });
        }
        Ok(Self {
            data,
            channels,
            height,
            width,
        })
    }

    /// Total number of elements (`channels * height * width`).
    #[inline]
    pub fn n_elements(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// Number of spatial elements per channel (`height * width`).
    #[inline]
    pub fn spatial_size(&self) -> usize {
        self.height * self.width
    }

    /// Global mean across all elements.
    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.data.iter().sum();
        sum / self.data.len() as f32
    }

    /// Global variance across all elements.
    pub fn variance(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let m = self.mean();
        let sum_sq: f32 = self.data.iter().map(|&x| (x - m) * (x - m)).sum();
        sum_sq / self.data.len() as f32
    }

    /// Global standard deviation.
    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Minimum element value.
    pub fn min(&self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Maximum element value.
    pub fn max(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Compute the flat index for `(c, h, w)`.
    #[inline]
    fn index(&self, c: usize, h: usize, w: usize) -> Result<usize, LatentBlendError> {
        if c >= self.channels || h >= self.height || w >= self.width {
            return Err(LatentBlendError::InvalidConfig(format!(
                "index ({c},{h},{w}) out of bounds for shape ({},{},{})",
                self.channels, self.height, self.width
            )));
        }
        Ok(c * self.height * self.width + h * self.width + w)
    }

    /// Get the value at `(c, h, w)`.
    pub fn get(&self, c: usize, h: usize, w: usize) -> Result<f32, LatentBlendError> {
        let idx = self.index(c, h, w)?;
        Ok(self.data[idx])
    }

    /// Set the value at `(c, h, w)`.
    pub fn set(&mut self, c: usize, h: usize, w: usize, v: f32) -> Result<(), LatentBlendError> {
        let idx = self.index(c, h, w)?;
        self.data[idx] = v;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Check that two latents have identical shapes; return the shared size.
fn check_same_shape(a: &LatentTensor, b: &LatentTensor) -> Result<usize, LatentBlendError> {
    let na = a.n_elements();
    let nb = b.n_elements();
    if na != nb || a.channels != b.channels || a.height != b.height || a.width != b.width {
        return Err(LatentBlendError::DimensionMismatch { a: na, b: nb });
    }
    Ok(na)
}

/// Check a blend weight is in `[0, 1]`.
fn check_weight(t: f32) -> Result<(), LatentBlendError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(LatentBlendError::InvalidWeight(t));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic blending
// ─────────────────────────────────────────────────────────────────────────────

/// Linear interpolation: `result[i] = (1-t)*a[i] + t*b[i]`.
pub fn lb_lerp(
    a: &LatentTensor,
    b: &LatentTensor,
    t: f32,
) -> Result<LatentTensor, LatentBlendError> {
    check_weight(t)?;
    let n = check_same_shape(a, b)?;
    let mut data = Vec::with_capacity(n);
    let one_minus_t = 1.0 - t;
    for i in 0..n {
        data.push(one_minus_t * a.data[i] + t * b.data[i]);
    }
    Ok(LatentTensor {
        data,
        channels: a.channels,
        height: a.height,
        width: a.width,
    })
}

/// Spherical linear interpolation (slerp) treating each latent as a vector on a
/// hypersphere. Falls back to `lb_lerp` when the angle between the two vectors
/// is smaller than `1e-6` rad.
pub fn lb_slerp(
    a: &LatentTensor,
    b: &LatentTensor,
    t: f32,
) -> Result<LatentTensor, LatentBlendError> {
    check_weight(t)?;
    let n = check_same_shape(a, b)?;

    // Compute dot product and norms.
    let dot: f32 = a.data.iter().zip(b.data.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.data.iter().map(|&x| x * x).sum::<f32>().sqrt();

    // If either vector is (near-)zero, fall back to lerp.
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return lb_lerp(a, b, t);
    }

    let cos_omega = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    let omega = cos_omega.acos();

    if omega.abs() < 1e-6 {
        return lb_lerp(a, b, t);
    }

    let sin_omega = omega.sin();
    let scale_a = ((1.0 - t) * omega).sin() / sin_omega;
    let scale_b = (t * omega).sin() / sin_omega;

    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(scale_a * a.data[i] + scale_b * b.data[i]);
    }
    Ok(LatentTensor {
        data,
        channels: a.channels,
        height: a.height,
        width: a.width,
    })
}

/// Weighted average of multiple latents. Weights are normalised internally so
/// they do not have to sum to 1.
///
/// Returns an error if:
/// - `latents` or `weights` is empty.
/// - `latents.len() != weights.len()`.
/// - Any weight is negative (values outside `[0, 1]` are not checked — only
///   negative values are meaningless for a convex blend).
pub fn lb_blend_multi(
    latents: &[LatentTensor],
    weights: &[f32],
) -> Result<LatentTensor, LatentBlendError> {
    if latents.is_empty() {
        return Err(LatentBlendError::EmptyInput);
    }
    if weights.len() != latents.len() {
        return Err(LatentBlendError::DimensionMismatch {
            a: weights.len(),
            b: latents.len(),
        });
    }

    // Validate weights (must all be non-negative).
    for &w in weights {
        if w < 0.0 {
            return Err(LatentBlendError::InvalidWeight(w));
        }
    }

    let weight_sum: f32 = weights.iter().sum();
    if weight_sum < 1e-12 {
        return Err(LatentBlendError::InvalidConfig(
            "sum of weights is effectively zero".into(),
        ));
    }

    let n = latents[0].n_elements();
    // Verify all shapes match.
    for l in latents.iter().skip(1) {
        if l.n_elements() != n
            || l.channels != latents[0].channels
            || l.height != latents[0].height
            || l.width != latents[0].width
        {
            return Err(LatentBlendError::DimensionMismatch {
                a: n,
                b: l.n_elements(),
            });
        }
    }

    let inv_sum = 1.0 / weight_sum;
    let mut data = vec![0.0_f32; n];
    for (latent, &w) in latents.iter().zip(weights.iter()) {
        let nw = w * inv_sum;
        for (d, &ld) in latent.data.iter().enumerate() {
            data[d] += nw * ld;
        }
    }

    Ok(LatentTensor {
        data,
        channels: latents[0].channels,
        height: latents[0].height,
        width: latents[0].width,
    })
}

/// Scaled addition: `result = a + scale * b`.
///
/// `scale` is unconstrained (can be negative or larger than 1).
pub fn lb_add_scaled(
    a: &LatentTensor,
    b: &LatentTensor,
    scale: f32,
) -> Result<LatentTensor, LatentBlendError> {
    let n = check_same_shape(a, b)?;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(a.data[i] + scale * b.data[i]);
    }
    Ok(LatentTensor {
        data,
        channels: a.channels,
        height: a.height,
        width: a.width,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial mask blending
// ─────────────────────────────────────────────────────────────────────────────

/// Blend two latents using a spatial mask.
///
/// `mask`: flat slice of length `height * width`, values in `[0, 1]`.
/// `mask[h * width + w] == 1` → use `a`; `== 0` → use `b`.
/// Each spatial position gets the same mask value across all channels.
pub fn lb_mask_blend(
    a: &LatentTensor,
    b: &LatentTensor,
    mask: &[f32],
) -> Result<LatentTensor, LatentBlendError> {
    check_same_shape(a, b)?;
    let spatial = a.spatial_size();
    if mask.len() != spatial {
        return Err(LatentBlendError::SpatialMismatch {
            n: spatial,
            m: mask.len(),
        });
    }

    let n = a.n_elements();
    let mut data = Vec::with_capacity(n);
    for c in 0..a.channels {
        let base = c * spatial;
        for (s, &m_val) in mask.iter().enumerate() {
            let m = m_val.clamp(0.0, 1.0);
            data.push(m * a.data[base + s] + (1.0 - m) * b.data[base + s]);
        }
    }
    Ok(LatentTensor {
        data,
        channels: a.channels,
        height: a.height,
        width: a.width,
    })
}

/// Create a soft circular mask centred at `(cx, cy)` with the given radius.
///
/// Values: `1.0` well inside, `0.0` well outside, smooth falloff over `feather`
/// pixels at the boundary (using a smoothstep-style transition).
pub fn lb_circular_mask(
    height: usize,
    width: usize,
    cx: f32,
    cy: f32,
    radius: f32,
    feather: f32,
) -> Vec<f32> {
    let mut mask = Vec::with_capacity(height * width);
    let feather = feather.max(1e-6);
    for h in 0..height {
        for w in 0..width {
            let dx = w as f32 - cx;
            let dy = h as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            // Smoothstep: 1 inside (radius - feather/2), 0 outside (radius + feather/2).
            let inner = radius - feather * 0.5;
            let outer = radius + feather * 0.5;
            let val = if dist <= inner {
                1.0
            } else if dist >= outer {
                0.0
            } else {
                // Smoothstep mapping [inner, outer] → [1, 0].
                let frac = (dist - inner) / (outer - inner); // 0..1
                1.0 - frac * frac * (3.0 - 2.0 * frac)
            };
            mask.push(val);
        }
    }
    mask
}

/// Create a horizontal gradient mask transitioning linearly from `left_value`
/// at `x = 0` to `right_value` at `x = width - 1`.
pub fn lb_gradient_mask(
    height: usize,
    width: usize,
    left_value: f32,
    right_value: f32,
) -> Vec<f32> {
    let mut mask = Vec::with_capacity(height * width);
    for _h in 0..height {
        for w in 0..width {
            let t = if width <= 1 {
                0.0
            } else {
                w as f32 / (width - 1) as f32
            };
            mask.push((1.0 - t) * left_value + t * right_value);
        }
    }
    mask
}

/// Morphological dilation with a square kernel of half-size `radius`.
///
/// Any spatial position becomes `max` over a `(2*radius+1) × (2*radius+1)` window.
pub fn lb_dilate_mask(mask: &[f32], height: usize, width: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; height * width];
    let r = radius as isize;
    for h in 0..height {
        for w in 0..width {
            let mut best: f32 = 0.0;
            let h_start = (h as isize - r).max(0) as usize;
            let h_end = (h as isize + r).min(height as isize - 1) as usize;
            let w_start = (w as isize - r).max(0) as usize;
            let w_end = (w as isize + r).min(width as isize - 1) as usize;
            for nh in h_start..=h_end {
                for nw in w_start..=w_end {
                    let v = mask[nh * width + nw];
                    if v > best {
                        best = v;
                    }
                }
            }
            out[h * width + w] = best;
        }
    }
    out
}

/// Smooth a mask with a box blur of the given pixel radius.
///
/// Uses two-pass (horizontal then vertical) separable box blur for efficiency.
pub fn lb_smooth_mask(mask: &[f32], height: usize, width: usize, radius: usize) -> Vec<f32> {
    if radius == 0 {
        return mask.to_vec();
    }
    let r = radius as isize;

    // Horizontal pass.
    let mut tmp = vec![0.0_f32; height * width];
    for h in 0..height {
        for w in 0..width {
            let w_start = (w as isize - r).max(0) as usize;
            let w_end = (w as isize + r).min(width as isize - 1) as usize;
            let count = (w_end - w_start + 1) as f32;
            let sum: f32 = (w_start..=w_end).map(|ww| mask[h * width + ww]).sum();
            tmp[h * width + w] = sum / count;
        }
    }

    // Vertical pass.
    let mut out = vec![0.0_f32; height * width];
    for h in 0..height {
        for w in 0..width {
            let h_start = (h as isize - r).max(0) as usize;
            let h_end = (h as isize + r).min(height as isize - 1) as usize;
            let count = (h_end - h_start + 1) as f32;
            let sum: f32 = (h_start..=h_end).map(|hh| tmp[hh * width + w]).sum();
            out[h * width + w] = sum / count;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistical harmonization
// ─────────────────────────────────────────────────────────────────────────────

/// Match the per-channel mean and variance of `source` to those of `target`.
///
/// ```text
/// result_c = (source_c - mean(source_c)) / std(source_c) * std(target_c) + mean(target_c)
/// ```
///
/// If a channel's standard deviation is effectively zero, the channel is kept
/// constant at the target mean.
pub fn lb_harmonize_statistics(
    source: &LatentTensor,
    target: &LatentTensor,
) -> Result<LatentTensor, LatentBlendError> {
    check_same_shape(source, target)?;
    let spatial = source.spatial_size();
    let mut data = source.data.clone();

    for c in 0..source.channels {
        let base = c * spatial;
        let slice = &source.data[base..base + spatial];

        let src_mean: f32 = slice.iter().sum::<f32>() / spatial as f32;
        let src_var: f32 = slice
            .iter()
            .map(|&x| (x - src_mean) * (x - src_mean))
            .sum::<f32>()
            / spatial as f32;
        let src_std = src_var.sqrt().max(1e-8);

        let tgt_slice = &target.data[base..base + spatial];
        let tgt_mean: f32 = tgt_slice.iter().sum::<f32>() / spatial as f32;
        let tgt_var: f32 = tgt_slice
            .iter()
            .map(|&x| (x - tgt_mean) * (x - tgt_mean))
            .sum::<f32>()
            / spatial as f32;
        let tgt_std = tgt_var.sqrt();

        for s in 0..spatial {
            let normalized = (source.data[base + s] - src_mean) / src_std;
            data[base + s] = normalized * tgt_std + tgt_mean;
        }
    }

    Ok(LatentTensor {
        data,
        channels: source.channels,
        height: source.height,
        width: source.width,
    })
}

/// Normalize a latent to zero mean and unit variance, computed per channel.
pub fn lb_normalize(latent: &LatentTensor) -> Result<LatentTensor, LatentBlendError> {
    if latent.data.is_empty() {
        return Err(LatentBlendError::EmptyInput);
    }
    let spatial = latent.spatial_size();
    let mut data = latent.data.clone();

    for c in 0..latent.channels {
        let base = c * spatial;
        let slice = &latent.data[base..base + spatial];
        let mean: f32 = slice.iter().sum::<f32>() / spatial as f32;
        let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / spatial as f32;
        let std = var.sqrt().max(1e-8);
        for s in 0..spatial {
            data[base + s] = (latent.data[base + s] - mean) / std;
        }
    }

    Ok(LatentTensor {
        data,
        channels: latent.channels,
        height: latent.height,
        width: latent.width,
    })
}

/// Denormalize: apply per-channel mean and std to a normalized latent.
///
/// `mean` and `std` must each have length `channels`.
pub fn lb_denormalize(
    normalized: &LatentTensor,
    mean: &[f32],
    std: &[f32],
) -> Result<LatentTensor, LatentBlendError> {
    if mean.len() != normalized.channels {
        return Err(LatentBlendError::DimensionMismatch {
            a: mean.len(),
            b: normalized.channels,
        });
    }
    if std.len() != normalized.channels {
        return Err(LatentBlendError::DimensionMismatch {
            a: std.len(),
            b: normalized.channels,
        });
    }
    let spatial = normalized.spatial_size();
    let mut data = normalized.data.clone();
    for c in 0..normalized.channels {
        let base = c * spatial;
        for s in 0..spatial {
            data[base + s] = normalized.data[base + s] * std[c] + mean[c];
        }
    }
    Ok(LatentTensor {
        data,
        channels: normalized.channels,
        height: normalized.height,
        width: normalized.width,
    })
}

/// Compute per-channel mean and standard deviation.
///
/// Returns `(means, stds)` each of length `channels`.
pub fn lb_channel_stats(latent: &LatentTensor) -> (Vec<f32>, Vec<f32>) {
    let spatial = latent.spatial_size();
    let mut means = Vec::with_capacity(latent.channels);
    let mut stds = Vec::with_capacity(latent.channels);

    for c in 0..latent.channels {
        let base = c * spatial;
        let slice = &latent.data[base..base + spatial];
        let mean: f32 = slice.iter().sum::<f32>() / spatial as f32;
        let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / spatial as f32;
        means.push(mean);
        stds.push(var.sqrt());
    }
    (means, stds)
}

// ─────────────────────────────────────────────────────────────────────────────
// Frequency-domain blending (spatial frequency separation)
// ─────────────────────────────────────────────────────────────────────────────

/// Separate a latent into low-frequency and high-frequency components.
///
/// `low` = box-blurred version (radius `radius`); `high` = original − low.
pub fn lb_frequency_separate(
    latent: &LatentTensor,
    radius: usize,
) -> Result<(LatentTensor, LatentTensor), LatentBlendError> {
    if latent.data.is_empty() {
        return Err(LatentBlendError::EmptyInput);
    }

    let spatial = latent.spatial_size();
    let channels = latent.channels;
    let height = latent.height;
    let width = latent.width;
    let n = latent.n_elements();

    let mut low_data = Vec::with_capacity(n);
    let mut high_data = Vec::with_capacity(n);

    for c in 0..channels {
        let base = c * spatial;
        let channel_slice = &latent.data[base..base + spatial];
        // Apply box blur on the channel slice.
        let blurred = lb_smooth_mask(channel_slice, height, width, radius);
        for (s, &blurred_val) in blurred.iter().enumerate() {
            low_data.push(blurred_val);
            high_data.push(latent.data[base + s] - blurred_val);
        }
    }

    let low = LatentTensor {
        data: low_data,
        channels,
        height,
        width,
    };
    let high = LatentTensor {
        data: high_data,
        channels,
        height,
        width,
    };
    Ok((low, high))
}

/// Blend two latents with independent low/high-frequency weights.
///
/// - `low_weight_for_a = 1.0`: take all low-freq content from `a`.
/// - `low_weight_for_a = 0.0`: take all low-freq content from `b`.
/// - Same semantics for `high_weight_for_a`.
pub fn lb_frequency_blend(
    a: &LatentTensor,
    b: &LatentTensor,
    low_weight_for_a: f32,
    high_weight_for_a: f32,
    blur_radius: usize,
) -> Result<LatentTensor, LatentBlendError> {
    check_same_shape(a, b)?;

    let (low_a, high_a) = lb_frequency_separate(a, blur_radius)?;
    let (low_b, high_b) = lb_frequency_separate(b, blur_radius)?;

    let low_blend = lb_lerp(&low_b, &low_a, low_weight_for_a)?;
    let high_blend = lb_lerp(&high_b, &high_a, high_weight_for_a)?;

    lb_add_scaled(&low_blend, &high_blend, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Latent trajectory
// ─────────────────────────────────────────────────────────────────────────────

/// A single step in a latent trajectory, pairing a noise level `sigma` with a
/// latent tensor.
#[derive(Debug, Clone)]
pub struct LatentTrajectoryStep {
    /// Noise level at this step.
    pub sigma: f32,
    /// Latent tensor at this noise level.
    pub latent: LatentTensor,
}

/// A trajectory through latent space, indexed by noise level `sigma`.
#[derive(Debug, Clone, Default)]
pub struct LatentTrajectory {
    /// Steps in ascending or arbitrary sigma order.
    pub steps: Vec<LatentTrajectoryStep>,
}

impl LatentTrajectory {
    /// Create an empty trajectory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a step.
    pub fn push(&mut self, sigma: f32, latent: LatentTensor) {
        self.steps.push(LatentTrajectoryStep { sigma, latent });
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// True when there are no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Look up a latent by exact sigma match (floating-point equality; use only
    /// when the value was recorded directly).
    pub fn at_sigma(&self, sigma: f32) -> Option<&LatentTensor> {
        self.steps
            .iter()
            .find(|s| s.sigma == sigma)
            .map(|s| &s.latent)
    }

    /// Interpolate the latent at an arbitrary `sigma` by finding the two
    /// bracketing steps and linearly blending between them.
    ///
    /// - If `sigma` is before all steps, returns the first step's latent.
    /// - If `sigma` is after all steps, returns the last step's latent.
    /// - Returns an error when there are fewer than 2 steps.
    pub fn interpolate_at(&self, sigma: f32) -> Result<LatentTensor, LatentBlendError> {
        if self.steps.is_empty() {
            return Err(LatentBlendError::EmptyInput);
        }
        if self.steps.len() == 1 {
            return Ok(self.steps[0].latent.clone());
        }

        // Find the two steps that bracket `sigma` (assumes steps are sorted
        // by sigma; if not, we do a linear scan for the closest pair).
        let n = self.steps.len();

        // Clamp to range.
        let first_sigma = self.steps[0].sigma;
        let last_sigma = self.steps[n - 1].sigma;
        let increasing = last_sigma >= first_sigma;

        if (increasing && sigma <= first_sigma) || (!increasing && sigma >= first_sigma) {
            return Ok(self.steps[0].latent.clone());
        }
        if (increasing && sigma >= last_sigma) || (!increasing && sigma <= last_sigma) {
            return Ok(self.steps[n - 1].latent.clone());
        }

        // Find bracketing pair.
        let (low_step, high_step) = if increasing {
            let idx = self
                .steps
                .iter()
                .position(|s| s.sigma >= sigma)
                .unwrap_or(n - 1);
            let idx = idx.max(1);
            (&self.steps[idx - 1], &self.steps[idx])
        } else {
            // Decreasing: find last step where sigma <= query sigma.
            let idx = self
                .steps
                .iter()
                .position(|s| s.sigma <= sigma)
                .unwrap_or(n - 1);
            let idx = idx.max(1);
            (&self.steps[idx - 1], &self.steps[idx])
        };

        let range = high_step.sigma - low_step.sigma;
        let t = if range.abs() < 1e-9 {
            0.0
        } else {
            (sigma - low_step.sigma) / range
        };
        let t = t.clamp(0.0, 1.0);

        lb_lerp(&low_step.latent, &high_step.latent, t)
    }
}

/// Blend two trajectories step-by-step using lerp.
///
/// Both trajectories must have the same number of steps and matching shapes at
/// each step.
pub fn lb_blend_trajectories(
    a: &LatentTrajectory,
    b: &LatentTrajectory,
    t: f32,
) -> Result<LatentTrajectory, LatentBlendError> {
    check_weight(t)?;
    if a.steps.len() != b.steps.len() {
        return Err(LatentBlendError::DimensionMismatch {
            a: a.steps.len(),
            b: b.steps.len(),
        });
    }
    let mut result = LatentTrajectory::new();
    for (step_a, step_b) in a.steps.iter().zip(b.steps.iter()) {
        let blended = lb_lerp(&step_a.latent, &step_b.latent, t)?;
        let sigma = (1.0 - t) * step_a.sigma + t * step_b.sigma;
        result.push(sigma, blended);
    }
    Ok(result)
}

/// Compute the total path length of a trajectory (sum of L2 distances between
/// consecutive steps).
pub fn lb_trajectory_length(trajectory: &LatentTrajectory) -> f32 {
    let n = trajectory.steps.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for i in 1..n {
        let dist = lb_l2_distance(&trajectory.steps[i - 1].latent, &trajectory.steps[i].latent)
            .unwrap_or(0.0);
        total += dist;
    }
    total
}

/// Compute step-wise velocities (L2 distance between each consecutive pair).
///
/// Returns a `Vec` of length `steps.len() - 1`.
pub fn lb_trajectory_velocity(trajectory: &LatentTrajectory) -> Vec<f32> {
    let n = trajectory.steps.len();
    if n < 2 {
        return Vec::new();
    }
    (1..n)
        .map(|i| {
            lb_l2_distance(&trajectory.steps[i - 1].latent, &trajectory.steps[i].latent)
                .unwrap_or(0.0)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics and formatting
// ─────────────────────────────────────────────────────────────────────────────

/// Blend quality statistics for a pair of latents and their blend.
#[derive(Debug, Clone)]
pub struct LatentBlendStats {
    /// Mean L2 distance between the two source latents.
    pub mean_distance: f32,
    /// Maximum per-channel mean absolute difference.
    pub max_channel_diff: f32,
    /// Heuristic quality: `1 - var(blend) / avg(var(a), var(b))`.
    pub blend_quality: f32,
}

/// Compute blend quality statistics.
pub fn lb_compute_stats(
    a: &LatentTensor,
    b: &LatentTensor,
    blend: &LatentTensor,
) -> LatentBlendStats {
    let mean_distance = lb_l2_distance(a, b).unwrap_or(0.0);

    // Per-channel max mean abs difference.
    let spatial = a.spatial_size().max(1);
    let channels = a.channels;
    let mut max_channel_diff: f32 = 0.0;
    for c in 0..channels {
        let base = c * spatial;
        let diff: f32 = (0..spatial)
            .map(|s| (a.data[base + s] - b.data[base + s]).abs())
            .sum::<f32>()
            / spatial as f32;
        if diff > max_channel_diff {
            max_channel_diff = diff;
        }
    }

    // Blend quality heuristic.
    let var_a = a.variance();
    let var_b = b.variance();
    let avg_var = (var_a + var_b) * 0.5;
    let var_blend = blend.variance();
    let blend_quality = if avg_var < 1e-12 {
        1.0
    } else {
        (1.0 - var_blend / avg_var).clamp(0.0, 1.0)
    };

    LatentBlendStats {
        mean_distance,
        max_channel_diff,
        blend_quality,
    }
}

/// Format blend stats as a human-readable string.
pub fn lb_format_stats(stats: &LatentBlendStats) -> String {
    format!(
        "LatentBlendStats {{ mean_distance: {:.6}, max_channel_diff: {:.6}, blend_quality: {:.6} }}",
        stats.mean_distance, stats.max_channel_diff, stats.blend_quality
    )
}

/// Compute the L2 (Euclidean) distance between two latents.
///
/// Returns an error when the shapes differ.
pub fn lb_l2_distance(a: &LatentTensor, b: &LatentTensor) -> Result<f32, LatentBlendError> {
    let n = check_same_shape(a, b)?;
    let sum_sq: f32 = (0..n).map(|i| (a.data[i] - b.data[i]).powi(2)).sum();
    Ok(sum_sq.sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ──────────────────────────────────────────────────────────────

    fn make_latent(channels: usize, h: usize, w: usize, fill: f32) -> LatentTensor {
        LatentTensor {
            data: vec![fill; channels * h * w],
            channels,
            height: h,
            width: w,
        }
    }

    fn make_latent_range(channels: usize, h: usize, w: usize) -> LatentTensor {
        let n = channels * h * w;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        LatentTensor {
            data,
            channels,
            height: h,
            width: w,
        }
    }

    // ── LatentTensor ─────────────────────────────────────────────────────────

    #[test]
    fn test_latent_tensor_new_zeros() {
        let lt = LatentTensor::new(2, 4, 4);
        assert_eq!(lt.n_elements(), 32);
        assert!(lt.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_latent_tensor_new_dims() {
        let lt = LatentTensor::new(3, 8, 16);
        assert_eq!(lt.channels, 3);
        assert_eq!(lt.height, 8);
        assert_eq!(lt.width, 16);
        assert_eq!(lt.spatial_size(), 128);
    }

    #[test]
    fn test_from_data_ok() {
        let data = vec![1.0_f32; 8];
        let lt = LatentTensor::from_data(data, 2, 2, 2).unwrap();
        assert_eq!(lt.n_elements(), 8);
    }

    #[test]
    fn test_from_data_wrong_size_error() {
        let data = vec![1.0_f32; 5];
        let err = LatentTensor::from_data(data, 2, 2, 2).unwrap_err();
        assert!(matches!(err, LatentBlendError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_get_set_correct_indexing() {
        let mut lt = LatentTensor::new(2, 3, 4);
        lt.set(1, 2, 3, 42.0).unwrap();
        assert_eq!(lt.get(1, 2, 3).unwrap(), 42.0);
    }

    #[test]
    fn test_get_out_of_bounds_error() {
        let lt = LatentTensor::new(2, 3, 4);
        let err = lt.get(2, 0, 0).unwrap_err(); // channel 2 OOB
        assert!(matches!(err, LatentBlendError::InvalidConfig(_)));
    }

    #[test]
    fn test_set_out_of_bounds_error() {
        let mut lt = LatentTensor::new(2, 3, 4);
        let err = lt.set(0, 3, 0, 1.0).unwrap_err(); // height 3 OOB
        assert!(matches!(err, LatentBlendError::InvalidConfig(_)));
    }

    #[test]
    fn test_mean_known_values() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let lt = LatentTensor::from_data(data, 1, 2, 2).unwrap();
        assert!((lt.mean() - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_variance_known_values() {
        // Variance of [1,2,3,4] = 1.25
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let lt = LatentTensor::from_data(data, 1, 2, 2).unwrap();
        assert!((lt.variance() - 1.25).abs() < 1e-5);
    }

    #[test]
    fn test_std_known_values() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let lt = LatentTensor::from_data(data, 1, 2, 2).unwrap();
        assert!((lt.std() - 1.25_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_min_max() {
        let data = vec![-3.0_f32, 0.0, 5.0, 2.0];
        let lt = LatentTensor::from_data(data, 1, 2, 2).unwrap();
        assert_eq!(lt.min(), -3.0);
        assert_eq!(lt.max(), 5.0);
    }

    // ── lb_lerp ──────────────────────────────────────────────────────────────

    #[test]
    fn test_lerp_t0_returns_a() {
        let a = make_latent(2, 4, 4, 1.0);
        let b = make_latent(2, 4, 4, 3.0);
        let result = lb_lerp(&a, &b, 0.0).unwrap();
        assert!(result.data.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_lerp_t1_returns_b() {
        let a = make_latent(2, 4, 4, 1.0);
        let b = make_latent(2, 4, 4, 3.0);
        let result = lb_lerp(&a, &b, 1.0).unwrap();
        assert!(result.data.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_lerp_t_half_midpoint() {
        let a = make_latent(1, 2, 2, 0.0);
        let b = make_latent(1, 2, 2, 4.0);
        let result = lb_lerp(&a, &b, 0.5).unwrap();
        assert!(result.data.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_lerp_dimension_mismatch_error() {
        let a = make_latent(2, 4, 4, 0.0);
        let b = make_latent(2, 4, 8, 0.0);
        assert!(matches!(
            lb_lerp(&a, &b, 0.5),
            Err(LatentBlendError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_lerp_invalid_weight_error() {
        let a = make_latent(1, 2, 2, 0.0);
        let b = make_latent(1, 2, 2, 1.0);
        assert!(matches!(
            lb_lerp(&a, &b, 1.5),
            Err(LatentBlendError::InvalidWeight(_))
        ));
    }

    // ── lb_slerp ─────────────────────────────────────────────────────────────

    #[test]
    fn test_slerp_t0_approx_a() {
        let a = make_latent(1, 2, 2, 1.0);
        let b = make_latent(1, 2, 2, 0.5);
        let result = lb_slerp(&a, &b, 0.0).unwrap();
        for (r, orig) in result.data.iter().zip(a.data.iter()) {
            assert!(
                (r - orig).abs() < 1e-4,
                "slerp t=0 should return a, got {r}"
            );
        }
    }

    #[test]
    fn test_slerp_t1_approx_b() {
        let a = make_latent(1, 2, 2, 1.0);
        let b = make_latent(1, 2, 2, 0.5);
        let result = lb_slerp(&a, &b, 1.0).unwrap();
        for (r, orig) in result.data.iter().zip(b.data.iter()) {
            assert!(
                (r - orig).abs() < 1e-4,
                "slerp t=1 should return b, got {r}"
            );
        }
    }

    #[test]
    fn test_slerp_identical_no_nan() {
        let a = make_latent(2, 4, 4, 2.0);
        let b = a.clone();
        let result = lb_slerp(&a, &b, 0.5).unwrap();
        assert!(
            result.data.iter().all(|x| x.is_finite()),
            "slerp produced NaN"
        );
    }

    #[test]
    fn test_slerp_zero_vector_fallback_no_nan() {
        let a = make_latent(1, 2, 2, 0.0);
        let b = make_latent(1, 2, 2, 1.0);
        let result = lb_slerp(&a, &b, 0.5).unwrap();
        assert!(result.data.iter().all(|x| x.is_finite()));
    }

    // ── lb_blend_multi ───────────────────────────────────────────────────────

    #[test]
    fn test_blend_multi_single_weight_1() {
        let a = make_latent(1, 2, 2, 5.0);
        let result = lb_blend_multi(std::slice::from_ref(&a), &[1.0]).unwrap();
        assert!(result.data.iter().all(|&x| (x - 5.0).abs() < 1e-6));
    }

    #[test]
    fn test_blend_multi_equal_weights_mean() {
        let a = make_latent(1, 2, 2, 0.0);
        let b = make_latent(1, 2, 2, 4.0);
        let result = lb_blend_multi(&[a, b], &[1.0, 1.0]).unwrap();
        assert!(result.data.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_blend_multi_empty_error() {
        let err = lb_blend_multi(&[], &[]).unwrap_err();
        assert!(matches!(err, LatentBlendError::EmptyInput));
    }

    #[test]
    fn test_blend_multi_mismatched_count_error() {
        let a = make_latent(1, 2, 2, 0.0);
        let err = lb_blend_multi(&[a], &[0.5, 0.5]).unwrap_err();
        assert!(matches!(err, LatentBlendError::DimensionMismatch { .. }));
    }

    // ── lb_add_scaled ────────────────────────────────────────────────────────

    #[test]
    fn test_add_scaled_scale_zero_returns_a() {
        let a = make_latent(1, 2, 2, 3.0);
        let b = make_latent(1, 2, 2, 7.0);
        let result = lb_add_scaled(&a, &b, 0.0).unwrap();
        assert!(result.data.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_add_scaled_b_zeros_returns_a() {
        let a = make_latent(1, 2, 2, 3.0);
        let b = make_latent(1, 2, 2, 0.0);
        let result = lb_add_scaled(&a, &b, 1.0).unwrap();
        assert!(result.data.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    // ── lb_mask_blend ────────────────────────────────────────────────────────

    #[test]
    fn test_mask_blend_all_ones_returns_a() {
        let a = make_latent(2, 4, 4, 1.0);
        let b = make_latent(2, 4, 4, 0.0);
        let mask = vec![1.0_f32; 16];
        let result = lb_mask_blend(&a, &b, &mask).unwrap();
        assert!(result.data.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_mask_blend_all_zeros_returns_b() {
        let a = make_latent(2, 4, 4, 1.0);
        let b = make_latent(2, 4, 4, 5.0);
        let mask = vec![0.0_f32; 16];
        let result = lb_mask_blend(&a, &b, &mask).unwrap();
        assert!(result.data.iter().all(|&x| (x - 5.0).abs() < 1e-6));
    }

    #[test]
    fn test_mask_blend_wrong_size_error() {
        let a = make_latent(2, 4, 4, 1.0);
        let b = make_latent(2, 4, 4, 0.0);
        let mask = vec![1.0_f32; 10]; // wrong: should be 16
        let err = lb_mask_blend(&a, &b, &mask).unwrap_err();
        assert!(matches!(err, LatentBlendError::SpatialMismatch { .. }));
    }

    // ── lb_circular_mask ─────────────────────────────────────────────────────

    #[test]
    fn test_circular_mask_length() {
        let h = 32;
        let w = 32;
        let mask = lb_circular_mask(h, w, 16.0, 16.0, 10.0, 2.0);
        assert_eq!(mask.len(), h * w);
    }

    #[test]
    fn test_circular_mask_center_is_one() {
        let mask = lb_circular_mask(64, 64, 32.0, 32.0, 20.0, 4.0);
        let center = mask[32 * 64 + 32];
        assert!(
            (center - 1.0).abs() < 1e-6,
            "center should be 1.0, got {center}"
        );
    }

    #[test]
    fn test_circular_mask_far_corner_is_zero() {
        let mask = lb_circular_mask(64, 64, 32.0, 32.0, 10.0, 2.0);
        let corner = mask[0]; // top-left
        assert!(corner < 0.01, "corner should be near 0.0, got {corner}");
    }

    // ── lb_gradient_mask ─────────────────────────────────────────────────────

    #[test]
    fn test_gradient_mask_first_pixel_left_value() {
        let mask = lb_gradient_mask(4, 8, 0.0, 1.0);
        assert!((mask[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_mask_last_pixel_right_value() {
        let mask = lb_gradient_mask(4, 8, 0.0, 1.0);
        let last = mask[4 * 8 - 1]; // bottom-right corner
        assert!((last - 1.0).abs() < 1e-6, "last pixel = {last}");
    }

    // ── lb_dilate_mask ───────────────────────────────────────────────────────

    #[test]
    fn test_dilate_mask_single_nonzero_spreads() {
        let h = 5;
        let w = 5;
        let mut mask = vec![0.0_f32; h * w];
        mask[2 * w + 2] = 1.0; // centre pixel
        let dilated = lb_dilate_mask(&mask, h, w, 1);
        // All 3x3 neighbours should be 1.0.
        for dh in 1..=3 {
            for dw in 1..=3 {
                assert_eq!(dilated[dh * w + dw], 1.0);
            }
        }
        // Corner should still be 0.
        assert_eq!(dilated[0], 0.0);
    }

    // ── lb_smooth_mask ───────────────────────────────────────────────────────

    #[test]
    fn test_smooth_mask_constant_unchanged() {
        let mask = vec![0.5_f32; 16];
        let result = lb_smooth_mask(&mask, 4, 4, 2);
        for &v in &result {
            assert!(
                (v - 0.5).abs() < 1e-5,
                "constant mask should be unchanged, got {v}"
            );
        }
    }

    // ── lb_harmonize_statistics ──────────────────────────────────────────────

    #[test]
    fn test_harmonize_statistics_mean_matches_target() {
        let source = make_latent(1, 4, 4, 10.0);
        let mut target_data = vec![0.0_f32; 16];
        for (i, v) in target_data.iter_mut().enumerate() {
            *v = i as f32 * 0.1;
        }
        let target = LatentTensor::from_data(target_data, 1, 4, 4).unwrap();
        let result = lb_harmonize_statistics(&source, &target).unwrap();
        let result_mean = result.mean();
        let target_mean = target.mean();
        assert!(
            (result_mean - target_mean).abs() < 1e-4,
            "after harmonize, source mean {result_mean} should ≈ target mean {target_mean}"
        );
    }

    #[test]
    fn test_harmonize_statistics_dimension_mismatch_error() {
        let a = make_latent(2, 4, 4, 1.0);
        let b = make_latent(2, 4, 8, 0.0);
        assert!(matches!(
            lb_harmonize_statistics(&a, &b),
            Err(LatentBlendError::DimensionMismatch { .. })
        ));
    }

    // ── lb_normalize ─────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_mean_near_zero() {
        let lt = make_latent_range(2, 4, 4);
        let norm = lb_normalize(&lt).unwrap();
        let (means, _) = lb_channel_stats(&norm);
        for m in &means {
            assert!(m.abs() < 1e-4, "channel mean after normalize: {m}");
        }
    }

    #[test]
    fn test_normalize_std_near_one() {
        let lt = make_latent_range(2, 4, 4);
        let norm = lb_normalize(&lt).unwrap();
        let (_, stds) = lb_channel_stats(&norm);
        for s in &stds {
            assert!((s - 1.0).abs() < 1e-4, "channel std after normalize: {s}");
        }
    }

    // ── lb_denormalize ───────────────────────────────────────────────────────

    #[test]
    fn test_denormalize_round_trip() {
        let lt = make_latent_range(2, 4, 4);
        let (means, stds) = lb_channel_stats(&lt);
        let norm = lb_normalize(&lt).unwrap();
        let restored = lb_denormalize(&norm, &means, &stds).unwrap();
        for (orig, res) in lt.data.iter().zip(restored.data.iter()) {
            assert!(
                (orig - res).abs() < 1e-4,
                "round-trip failed: {orig} vs {res}"
            );
        }
    }

    // ── lb_channel_stats ─────────────────────────────────────────────────────

    #[test]
    fn test_channel_stats_correct_mean() {
        // Channel 0 = all 1.0, channel 1 = all 3.0.
        let mut data = vec![1.0_f32; 8];
        for v in data.iter_mut().skip(4) {
            *v = 3.0;
        }
        let lt = LatentTensor::from_data(data, 2, 2, 2).unwrap();
        let (means, _) = lb_channel_stats(&lt);
        assert!((means[0] - 1.0).abs() < 1e-6);
        assert!((means[1] - 3.0).abs() < 1e-6);
    }

    // ── lb_frequency_separate ────────────────────────────────────────────────

    #[test]
    fn test_frequency_separate_low_plus_high_equals_original() {
        let lt = make_latent_range(2, 8, 8);
        let (low, high) = lb_frequency_separate(&lt, 2).unwrap();
        for i in 0..lt.n_elements() {
            let reconstructed = low.data[i] + high.data[i];
            assert!(
                (reconstructed - lt.data[i]).abs() < 1e-4,
                "low+high != original at {i}: {reconstructed} vs {}",
                lt.data[i]
            );
        }
    }

    // ── lb_frequency_blend ───────────────────────────────────────────────────

    #[test]
    fn test_frequency_blend_all_from_a() {
        let a = make_latent(1, 8, 8, 2.0);
        let b = make_latent(1, 8, 8, 0.0);
        let result = lb_frequency_blend(&a, &b, 1.0, 1.0, 2).unwrap();
        // With all-constant inputs, low of a = 2.0, high of a ≈ 0, result ≈ 2.0.
        let expected = 2.0_f32;
        for &v in &result.data {
            assert!((v - expected).abs() < 1e-4, "expected ~{expected}, got {v}");
        }
    }

    #[test]
    fn test_frequency_blend_all_from_b() {
        let a = make_latent(1, 8, 8, 0.0);
        let b = make_latent(1, 8, 8, 5.0);
        let result = lb_frequency_blend(&a, &b, 0.0, 0.0, 2).unwrap();
        let expected = 5.0_f32;
        for &v in &result.data {
            assert!((v - expected).abs() < 1e-4, "expected ~{expected}, got {v}");
        }
    }

    // ── LatentTrajectory ─────────────────────────────────────────────────────

    #[test]
    fn test_trajectory_push_len_is_empty() {
        let mut traj = LatentTrajectory::new();
        assert!(traj.is_empty());
        assert_eq!(traj.len(), 0);
        traj.push(1.0, make_latent(1, 2, 2, 0.0));
        assert!(!traj.is_empty());
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_trajectory_at_sigma_found() {
        let mut traj = LatentTrajectory::new();
        traj.push(0.5, make_latent(1, 2, 2, 3.0));
        let lt = traj.at_sigma(0.5).unwrap();
        assert!(lt.data.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_trajectory_at_sigma_not_found() {
        let mut traj = LatentTrajectory::new();
        traj.push(0.5, make_latent(1, 2, 2, 0.0));
        assert!(traj.at_sigma(0.9).is_none());
    }

    #[test]
    fn test_trajectory_interpolate_between_steps() {
        let mut traj = LatentTrajectory::new();
        traj.push(0.0, make_latent(1, 2, 2, 0.0));
        traj.push(1.0, make_latent(1, 2, 2, 4.0));
        let interp = traj.interpolate_at(0.5).unwrap();
        // Midpoint between 0 and 1 (sigma) maps t=0.5 → value = 2.0.
        assert!(
            interp.data.iter().all(|&x| (x - 2.0).abs() < 1e-4),
            "expected 2.0, got {:?}",
            &interp.data[..4]
        );
    }

    #[test]
    fn test_trajectory_interpolate_empty_error() {
        let traj = LatentTrajectory::new();
        assert!(matches!(
            traj.interpolate_at(0.5),
            Err(LatentBlendError::EmptyInput)
        ));
    }

    // ── lb_blend_trajectories ────────────────────────────────────────────────

    #[test]
    fn test_blend_trajectories_same_length() {
        let mut a = LatentTrajectory::new();
        let mut b = LatentTrajectory::new();
        for i in 0..5 {
            a.push(i as f32 * 0.25, make_latent(1, 2, 2, i as f32));
            b.push(i as f32 * 0.25, make_latent(1, 2, 2, i as f32 * 2.0));
        }
        let result = lb_blend_trajectories(&a, &b, 0.5).unwrap();
        assert_eq!(result.len(), 5);
    }

    // ── lb_trajectory_length ─────────────────────────────────────────────────

    #[test]
    fn test_trajectory_length_empty_is_zero() {
        let traj = LatentTrajectory::new();
        assert_eq!(lb_trajectory_length(&traj), 0.0);
    }

    #[test]
    fn test_trajectory_length_single_step_is_zero() {
        let mut traj = LatentTrajectory::new();
        traj.push(0.0, make_latent(1, 2, 2, 1.0));
        assert_eq!(lb_trajectory_length(&traj), 0.0);
    }

    #[test]
    fn test_trajectory_length_known_pair() {
        // Two latents each of shape 1×2×2 filled with 0 and 1.
        // L2 distance = sqrt(4) = 2.0.
        let mut traj = LatentTrajectory::new();
        traj.push(0.0, make_latent(1, 2, 2, 0.0));
        traj.push(1.0, make_latent(1, 2, 2, 1.0));
        let length = lb_trajectory_length(&traj);
        assert!((length - 2.0).abs() < 1e-5, "expected 2.0, got {length}");
    }

    // ── lb_trajectory_velocity ───────────────────────────────────────────────

    #[test]
    fn test_trajectory_velocity_length() {
        let mut traj = LatentTrajectory::new();
        for i in 0..4 {
            traj.push(i as f32, make_latent(1, 2, 2, i as f32));
        }
        let vel = lb_trajectory_velocity(&traj);
        assert_eq!(vel.len(), 3); // n-1 = 3
    }

    // ── lb_compute_stats / lb_l2_distance ────────────────────────────────────

    #[test]
    fn test_l2_distance_identical_is_zero() {
        let a = make_latent(2, 4, 4, 1.0);
        let b = a.clone();
        assert!((lb_l2_distance(&a, &b).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_known_pair() {
        // [0,0,0,3] vs [4,0,0,0] → sqrt(16+9) = sqrt(25) = 5
        let a = LatentTensor::from_data(vec![0.0, 0.0, 0.0, 3.0], 1, 2, 2).unwrap();
        let b = LatentTensor::from_data(vec![4.0, 0.0, 0.0, 0.0], 1, 2, 2).unwrap();
        assert!((lb_l2_distance(&a, &b).unwrap() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_fields() {
        let a = make_latent(2, 4, 4, 0.0);
        let b = make_latent(2, 4, 4, 1.0);
        let blend = lb_lerp(&a, &b, 0.5).unwrap();
        let stats = lb_compute_stats(&a, &b, &blend);
        // mean_distance should be positive (sqrt of sum of 32 unit diffs).
        assert!(stats.mean_distance > 0.0);
        // blend_quality in [0, 1].
        assert!(stats.blend_quality >= 0.0 && stats.blend_quality <= 1.0);
    }

    // ── lb_format_stats ──────────────────────────────────────────────────────

    #[test]
    fn test_format_stats_non_empty() {
        let stats = LatentBlendStats {
            mean_distance: 1.234,
            max_channel_diff: 0.5,
            blend_quality: 0.9,
        };
        let s = lb_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("mean_distance"));
    }

    // ── Additional edge-case tests ────────────────────────────────────────────

    #[test]
    fn test_lerp_single_element() {
        let a = LatentTensor::from_data(vec![2.0_f32], 1, 1, 1).unwrap();
        let b = LatentTensor::from_data(vec![6.0_f32], 1, 1, 1).unwrap();
        let result = lb_lerp(&a, &b, 0.25).unwrap();
        assert!((result.data[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_slerp_orthogonal_vectors() {
        // a = [1,0], b = [0,1] — 90° angle, slerp at t=0.5 should be unit diagonal.
        let a = LatentTensor::from_data(vec![1.0_f32, 0.0, 0.0, 0.0], 1, 2, 2).unwrap();
        let b = LatentTensor::from_data(vec![0.0_f32, 0.0, 1.0, 0.0], 1, 2, 2).unwrap();
        let result = lb_slerp(&a, &b, 0.5).unwrap();
        assert!(
            result.data.iter().all(|x| x.is_finite()),
            "slerp orthogonal NaN"
        );
    }

    #[test]
    fn test_blend_multi_three_equal_weights() {
        let a = make_latent(1, 2, 2, 0.0);
        let b = make_latent(1, 2, 2, 3.0);
        let c = make_latent(1, 2, 2, 6.0);
        let result = lb_blend_multi(&[a, b, c], &[1.0, 1.0, 1.0]).unwrap();
        assert!(result.data.iter().all(|&x| (x - 3.0).abs() < 1e-5));
    }

    #[test]
    fn test_denormalize_wrong_mean_length_error() {
        let lt = make_latent(2, 4, 4, 1.0);
        let err = lb_denormalize(&lt, &[0.0], &[1.0, 1.0]).unwrap_err();
        assert!(matches!(err, LatentBlendError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_add_scaled_negative_scale() {
        let a = make_latent(1, 2, 2, 5.0);
        let b = make_latent(1, 2, 2, 2.0);
        let result = lb_add_scaled(&a, &b, -1.0).unwrap();
        assert!(result.data.iter().all(|&x| (x - 3.0).abs() < 1e-5));
    }

    #[test]
    fn test_blend_trajectories_t0_equals_a() {
        let mut a = LatentTrajectory::new();
        let mut b = LatentTrajectory::new();
        a.push(0.0, make_latent(1, 2, 2, 1.0));
        b.push(0.0, make_latent(1, 2, 2, 9.0));
        let result = lb_blend_trajectories(&a, &b, 0.0).unwrap();
        assert!(result.steps[0]
            .latent
            .data
            .iter()
            .all(|&x| (x - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_trajectory_velocity_empty() {
        let traj = LatentTrajectory::new();
        let vel = lb_trajectory_velocity(&traj);
        assert!(vel.is_empty());
    }

    #[test]
    fn test_gradient_mask_monotone() {
        let mask = lb_gradient_mask(1, 8, 0.0, 1.0);
        for i in 1..mask.len() {
            assert!(
                mask[i] >= mask[i - 1],
                "gradient mask should be non-decreasing"
            );
        }
    }

    #[test]
    fn test_circular_mask_all_values_in_range() {
        let mask = lb_circular_mask(32, 32, 16.0, 16.0, 10.0, 3.0);
        for &v in &mask {
            assert!((0.0..=1.0).contains(&v), "mask value out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_smooth_mask_zero_radius_unchanged() {
        let mask = vec![0.1, 0.5, 0.9, 0.3];
        let result = lb_smooth_mask(&mask, 2, 2, 0);
        assert_eq!(result, mask);
    }
}
