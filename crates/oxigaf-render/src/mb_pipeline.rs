//! Velocity-based per-pixel motion blur pipeline.
//!
//! This module implements the full cinematic motion blur pipeline:
//! - [`VelocityBuffer`]: separate-channel per-pixel velocity storage
//! - [`MbConfig`]: shutter angle, sample count, exposure, depth-aware weighting
//! - [`apply_motion_blur`]: end-to-end pipeline
//! - Velocity generation utilities (`mb_velocity_*`)
//! - Primitive helpers (`mb_bilinear_sample`, `mb_triangle_weight`, …)
//!
//! # Relationship to the sibling motion-blur modules
//!
//! This is the richest of the three CPU motion-blur modules: prefer it when
//! you have a real velocity buffer and want shutter-angle control,
//! depth-aware sample rejection, stratified jitter, or velocity
//! dilation/smoothing. See the module docs of [`crate::motion_blur`] for the
//! full comparison table. This module reuses that module's
//! [`MotionBlurError`] and its `f32`-RGB bilinear tap (re-exported here as
//! [`mb_bilinear_sample`]) rather than defining its own copies.

use crate::motion_blur::MotionBlurError;

// ─────────────────────────────────────────────────────────────────────────────
// VelocityBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Per-pixel 2-D velocity (optical flow) in pixel units.
///
/// `velocity_x[i]` is the horizontal displacement of pixel `i` (pixels/frame).
/// `velocity_y[i]` is the vertical displacement of pixel `i` (pixels/frame).
/// Positive x points right; positive y points down.
#[derive(Debug, Clone)]
pub struct VelocityBuffer {
    /// Horizontal velocity per pixel, length `width * height`.
    pub velocity_x: Vec<f32>,
    /// Vertical velocity per pixel, length `width * height`.
    pub velocity_y: Vec<f32>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl VelocityBuffer {
    /// Create a zero-initialised velocity buffer.
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            velocity_x: vec![0.0_f32; n],
            velocity_y: vec![0.0_f32; n],
            width,
            height,
        }
    }

    /// Construct from two pre-allocated vectors.
    ///
    /// # Errors
    ///
    /// [`MotionBlurError::DimensionMismatch`] if either vector does not have
    /// exactly `width * height` elements.
    pub fn from_vecs(
        vx: Vec<f32>,
        vy: Vec<f32>,
        width: usize,
        height: usize,
    ) -> Result<Self, MotionBlurError> {
        let expected = width * height;
        if vx.len() != expected {
            return Err(MotionBlurError::DimensionMismatch {
                expected,
                got: vx.len(),
            });
        }
        if vy.len() != expected {
            return Err(MotionBlurError::DimensionMismatch {
                expected,
                got: vy.len(),
            });
        }
        Ok(Self {
            velocity_x: vx,
            velocity_y: vy,
            width,
            height,
        })
    }

    /// Velocity magnitude at pixel `(x, y)`.
    ///
    /// Returns `0.0` if coordinates are out of bounds.
    pub fn magnitude(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let i = y * self.width + x;
        let vx = self.velocity_x[i];
        let vy = self.velocity_y[i];
        (vx * vx + vy * vy).sqrt()
    }

    /// Maximum velocity magnitude across all pixels.
    pub fn max_magnitude(&self) -> f32 {
        let n = self.width * self.height;
        let mut max = 0.0_f32;
        for i in 0..n {
            let vx = self.velocity_x[i];
            let vy = self.velocity_y[i];
            let m = (vx * vx + vy * vy).sqrt();
            if m > max {
                max = m;
            }
        }
        max
    }

    /// Mean velocity magnitude across all pixels.
    pub fn mean_magnitude(&self) -> f32 {
        let n = self.width * self.height;
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = (0..n)
            .map(|i| {
                let vx = self.velocity_x[i];
                let vy = self.velocity_y[i];
                (vx * vx + vy * vy).sqrt()
            })
            .sum();
        sum / n as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MbConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the velocity-based per-pixel motion blur pipeline.
///
/// Distinct from [`crate::VelocityBlurConfig`] (accumulation pipeline) and
/// [`crate::MotionBlurConfig`] (image-space RGBA pipeline).
#[derive(Debug, Clone)]
pub struct MbConfig {
    /// Shutter angle in degrees.  Controls blur extent.  Must be in `(0, 360]`.
    pub shutter_angle: f32,
    /// Number of samples taken along the blur vector per pixel (≥ 1).
    pub n_samples: usize,
    /// Exposure multiplier applied before computing the blur vector.
    pub exposure: f32,
    /// Maximum blur vector length in pixels.
    pub max_blur_pixels: f32,
    /// When `true`, depth similarity is used as an additional sample weight.
    pub depth_aware: bool,
    /// Sigma for the depth-similarity Gaussian weighting.
    pub depth_sigma: f32,
    /// When `true`, sample positions are jittered with xorshift64 noise.
    pub jitter: bool,
}

impl Default for MbConfig {
    fn default() -> Self {
        Self {
            shutter_angle: 180.0,
            n_samples: 16,
            exposure: 1.0,
            max_blur_pixels: 32.0,
            depth_aware: true,
            depth_sigma: 0.5,
            jitter: true,
        }
    }
}

impl MbConfig {
    /// Validate configuration.
    ///
    /// # Errors
    ///
    /// - [`MotionBlurError::InvalidShutterAngle`] for `shutter_angle` outside `(0, 360]`.
    /// - [`MotionBlurError::InvalidSampleCount`] when `n_samples == 0`.
    pub fn validate(&self) -> Result<(), MotionBlurError> {
        if self.shutter_angle <= 0.0 || self.shutter_angle > 360.0 {
            return Err(MotionBlurError::InvalidShutterAngle {
                angle: self.shutter_angle,
            });
        }
        if self.n_samples == 0 {
            return Err(MotionBlurError::InvalidSampleCount { samples: 0 });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MbStats / MotionBlurResult
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostic statistics produced by the `mb_` pipeline.
#[derive(Debug, Clone)]
pub struct MbStats {
    /// Mean blur-vector length over all pixels (after shutter/exposure scaling).
    pub mean_blur_pixels: f32,
    /// Maximum blur-vector length over all pixels.
    pub max_blur_pixels: f32,
    /// Fraction of pixels with blur magnitude > 0.5 px.
    pub blurred_fraction: f32,
    /// Estimated sample utilisation per pixel, in `[1, n_samples]`.
    ///
    /// This is a fast, **config-derived estimate** (`1 + (n_samples - 1) *
    /// min(mean_blur/max_blur_pixels, 1)`), not a measurement of the actual
    /// per-pixel sample/weight counts evaluated by [`mb_apply`] (which
    /// always evaluates exactly `n_samples` samples for every moving
    /// pixel). Treat it as a utilisation heuristic for tuning
    /// `n_samples`/`max_blur_pixels`, not as ground truth.
    pub estimated_sample_utilization: f32,
}

/// Result of the full `apply_motion_blur` pipeline.
#[derive(Debug, Clone)]
pub struct MotionBlurResult {
    /// Motion-blurred RGB image (row-major, `width * height * 3` f32 values).
    pub image: Vec<f32>,
    /// Diagnostic statistics.
    pub stats: MbStats,
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG.  Advances `state` and returns the next pseudo-random u64.
///
/// The state must not be zero; if it becomes zero the function resets it to 1.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Bilinear sample from a flat row-major RGB image at fractional coordinates.
///
/// Coordinates outside the image are clamped to the nearest edge pixel; an
/// empty image (`width == 0` or `height == 0`) yields `[0, 0, 0]`.
/// Returns `[r, g, b]`.
///
/// The implementation is shared with [`crate::motion_blur`] (this function is
/// the public entry point for the crate-internal `motion_blur::bilinear_sample`)
/// so the two motion-blur paths cannot resample differently.
#[inline]
pub fn mb_bilinear_sample(image: &[f32], width: usize, height: usize, x: f32, y: f32) -> [f32; 3] {
    crate::motion_blur::bilinear_sample(image, width, height, x, y)
}

/// Triangle weight for temporal sample position `t ∈ [-0.5, 0.5]`.
///
/// `w(t) = max(0, 1 - 2*|t|)`.
#[inline]
pub fn mb_triangle_weight(t: f32) -> f32 {
    (1.0 - 2.0 * t.abs()).max(0.0)
}

/// Depth similarity weight: `exp(-(d1 - d2)^2 / (2 * sigma^2))`.
///
/// Returns 1.0 when `d1 == d2`.  A sigma of 0.0 is treated as `1e-6`.
#[inline]
pub fn mb_depth_weight(d1: f32, d2: f32, sigma: f32) -> f32 {
    let s = sigma.max(1e-6_f32);
    let diff = d1 - d2;
    (-(diff * diff) / (2.0 * s * s)).exp()
}

/// Generate jittered sample offsets for `n` samples using xorshift64.
///
/// Each offset is in `[-cell_width/2, cell_width/2]` where
/// `cell_width = 1/n`, centred on the uniform grid spacing.
pub fn mb_jitter_samples(n: usize, rng_state: &mut u64) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    mb_fill_jitter_samples(&mut out, rng_state);
    out
}

/// Fill `dst` with `dst.len()` jitter offsets, drawing from `rng_state`.
///
/// Allocation-free twin of [`mb_jitter_samples`]: it draws the *same*
/// sequence of values from the same state, so hot loops can reuse one scratch
/// buffer across pixels without perturbing the random stream.
fn mb_fill_jitter_samples(dst: &mut [f32], rng_state: &mut u64) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    let cell_width = 1.0 / n as f32;
    let half_cell = cell_width * 0.5;
    for slot in dst.iter_mut() {
        let bits = xorshift64(rng_state);
        let r = (bits as f64 / u64::MAX as f64) as f32;
        *slot = r * cell_width - half_cell;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Velocity generation utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a constant-velocity buffer (translational camera motion).
///
/// Every pixel receives the same velocity `(vx, vy)`.
pub fn mb_velocity_from_camera_motion(
    width: usize,
    height: usize,
    vx: f32,
    vy: f32,
) -> VelocityBuffer {
    let n = width * height;
    VelocityBuffer {
        velocity_x: vec![vx; n],
        velocity_y: vec![vy; n],
        width,
        height,
    }
}

/// Generate a velocity buffer from a depth map (camera forward motion).
///
/// `vx[i] = camera_speed * focal_scale / depth[i]` for pixels with
/// `depth > 0`; `vy` is left at zero.
///
/// # Errors
///
/// [`MotionBlurError::DimensionMismatch`] if `depth_buf.len() != width * height`.
pub fn mb_velocity_from_depth(
    depth_buf: &[f32],
    width: usize,
    height: usize,
    camera_speed: f32,
    focal_scale: f32,
) -> Result<VelocityBuffer, MotionBlurError> {
    let expected = width * height;
    if depth_buf.len() != expected {
        return Err(MotionBlurError::DimensionMismatch {
            expected,
            got: depth_buf.len(),
        });
    }
    let mut vbuf = VelocityBuffer::new(width, height);
    for (i, &d) in depth_buf.iter().enumerate() {
        if d > 0.0 {
            vbuf.velocity_x[i] = camera_speed * focal_scale / d.max(1e-4_f32);
        }
    }
    Ok(vbuf)
}

/// Generate a radial (rotating camera) velocity buffer.
///
/// Each pixel's velocity is perpendicular to the radius from `(cx, cy)` and
/// proportional to that radius and `rotation_speed` (radians/frame).
pub fn mb_velocity_rotational(
    width: usize,
    height: usize,
    cx: f32,
    cy: f32,
    rotation_speed: f32,
) -> VelocityBuffer {
    let n = width * height;
    let mut vx_buf = vec![0.0_f32; n];
    let mut vy_buf = vec![0.0_f32; n];
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let i = y * width + x;
            vx_buf[i] = -dy * rotation_speed;
            vy_buf[i] = dx * rotation_speed;
        }
    }
    VelocityBuffer {
        velocity_x: vx_buf,
        velocity_y: vy_buf,
        width,
        height,
    }
}

/// Smooth a velocity buffer with a separable Gaussian kernel (sigma in pixels).
///
/// A sigma ≤ 0 returns a clone of the input unchanged.
///
/// # Errors
///
/// [`MotionBlurError::EmptyInput`] if `velocity.width * velocity.height == 0`.
pub fn mb_smooth_velocity(
    velocity: &VelocityBuffer,
    sigma: f32,
) -> Result<VelocityBuffer, MotionBlurError> {
    let w = velocity.width;
    let h = velocity.height;
    if w * h == 0 {
        return Err(MotionBlurError::EmptyInput);
    }
    if sigma <= 0.0 {
        return Ok(velocity.clone());
    }

    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let half_kernel: Vec<f32> = (0..=radius)
        .map(|i| {
            let x = i as f32;
            (-(x * x) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let mut kernel: Vec<f32> = half_kernel.iter().rev().skip(1).copied().collect();
    kernel.extend_from_slice(&half_kernel);
    let sum: f32 = kernel.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    kernel.iter_mut().for_each(|v| *v *= inv);
    let kr = (kernel.len() / 2) as i64;

    let n = w * h;
    let mut tmp = vec![0.0_f32; n];
    let mut out_vx = vec![0.0_f32; n];
    let mut out_vy = vec![0.0_f32; n];

    let conv_h = |src: &[f32], dst: &mut [f32]| {
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sx = (x as i64 + ki as i64 - kr).clamp(0, w as i64 - 1) as usize;
                    acc += kv * src[y * w + sx];
                }
                dst[y * w + x] = acc;
            }
        }
    };

    let conv_v = |src: &[f32], dst: &mut [f32]| {
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sy = (y as i64 + ki as i64 - kr).clamp(0, h as i64 - 1) as usize;
                    acc += kv * src[sy * w + x];
                }
                dst[y * w + x] = acc;
            }
        }
    };

    conv_h(&velocity.velocity_x, &mut tmp);
    conv_v(&tmp, &mut out_vx);
    conv_h(&velocity.velocity_y, &mut tmp);
    conv_v(&tmp, &mut out_vy);

    Ok(VelocityBuffer {
        velocity_x: out_vx,
        velocity_y: out_vy,
        width: w,
        height: h,
    })
}

/// Dilate velocity to fill zero-velocity regions (nearest-nonzero spreading).
///
/// Performs `n_passes` rounds of 4-connected dilation.  Depth proximity is
/// used to pick the closest-depth neighbour when multiple candidates exist.
///
/// # Errors
///
/// [`MotionBlurError::DimensionMismatch`] if `depth_buf.len() != width * height`.
pub fn mb_dilate_velocity(
    velocity: &VelocityBuffer,
    depth_buf: &[f32],
    n_passes: usize,
) -> Result<VelocityBuffer, MotionBlurError> {
    let w = velocity.width;
    let h = velocity.height;
    let expected = w * h;
    if depth_buf.len() != expected {
        return Err(MotionBlurError::DimensionMismatch {
            expected,
            got: depth_buf.len(),
        });
    }

    let mut cur = velocity.clone();
    for _ in 0..n_passes {
        let prev = cur.clone();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let vx = prev.velocity_x[i];
                let vy = prev.velocity_y[i];
                if (vx * vx + vy * vy).sqrt() > 0.0 {
                    continue;
                }

                let my_depth = depth_buf[i];
                let mut best_ddiff = f32::MAX;
                let mut best_vx = 0.0_f32;
                let mut best_vy = 0.0_f32;
                let mut found = false;

                for (dx, dy) in [(-1_i64, 0_i64), (1, 0), (0, -1), (0, 1)] {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || nx >= w as i64 || ny < 0 || ny >= h as i64 {
                        continue;
                    }
                    let ni = ny as usize * w + nx as usize;
                    let nvx = prev.velocity_x[ni];
                    let nvy = prev.velocity_y[ni];
                    if (nvx * nvx + nvy * nvy).sqrt() <= 0.0 {
                        continue;
                    }
                    let ddiff = (depth_buf[ni] - my_depth).abs();
                    if ddiff < best_ddiff {
                        best_ddiff = ddiff;
                        best_vx = nvx;
                        best_vy = nvy;
                        found = true;
                    }
                }
                if found {
                    cur.velocity_x[i] = best_vx;
                    cur.velocity_y[i] = best_vy;
                }
            }
        }
    }
    Ok(cur)
}

// ─────────────────────────────────────────────────────────────────────────────
// Core motion blur algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Apply motion blur to an RGB image using per-pixel velocity.
///
/// For each pixel, the velocity is scaled by `(shutter_angle/360) * exposure`,
/// clamped to `max_blur_pixels`, then `n_samples` points are sampled along
/// `[-0.5, 0.5]` of the blur vector.  Each sample is weighted by the triangle
/// function and optionally by depth similarity.
///
/// Pixels with zero velocity magnitude are passed through unchanged.
///
/// # Errors
///
/// - [`MotionBlurError::DimensionMismatch`] for inconsistent buffer sizes.
/// - [`MotionBlurError::InvalidShutterAngle`] / [`MotionBlurError::InvalidSampleCount`].
pub fn mb_apply(
    image: &[f32],
    velocity: &VelocityBuffer,
    depth_buf: Option<&[f32]>,
    config: &MbConfig,
) -> Result<Vec<f32>, MotionBlurError> {
    config.validate()?;

    let w = velocity.width;
    let h = velocity.height;
    let expected = w * h * 3;
    if image.len() != expected {
        return Err(MotionBlurError::DimensionMismatch {
            expected,
            got: image.len(),
        });
    }
    if let Some(d) = depth_buf {
        if d.len() != w * h {
            return Err(MotionBlurError::DimensionMismatch {
                expected: w * h,
                got: d.len(),
            });
        }
    }

    let scale = (config.shutter_angle / 360.0) * config.exposure;
    let n = config.n_samples;
    let mut output = image.to_vec();
    let mut rng_state: u64 = 0xDEAD_BEEF_1234_5678;

    // Reusable per-pixel jitter scratch, allocated once for the whole call
    // instead of once per pixel. With `jitter == false` the buffer stays all
    // zeros for the entire pass (nothing ever writes it), which is exactly
    // what the old per-pixel `vec![0.0; n]` produced.
    let mut jitters = vec![0.0_f32; n];

    for py in 0..h {
        for px in 0..w {
            let pi = py * w + px;
            let raw_vx = velocity.velocity_x[pi] * scale;
            let raw_vy = velocity.velocity_y[pi] * scale;
            let mag = (raw_vx * raw_vx + raw_vy * raw_vy).sqrt();
            if mag < 1e-8 {
                continue;
            }

            let (blur_vx, blur_vy) = if mag > config.max_blur_pixels {
                let inv = config.max_blur_pixels / mag;
                (raw_vx * inv, raw_vy * inv)
            } else {
                (raw_vx, raw_vy)
            };

            // NOTE: the refill must stay *after* the `mag < 1e-8` early
            // `continue` above. `rng_state` advances only for pixels that
            // actually get blurred, so hoisting the draw to the top of the
            // pixel body would shift the jitter stream on every image with
            // static regions.
            if config.jitter {
                mb_fill_jitter_samples(&mut jitters, &mut rng_state);
            }

            let center_depth = depth_buf.map(|d| d[pi]).unwrap_or(0.0);
            let mut acc = [0.0_f32; 3];
            let mut weight_sum = 0.0_f32;

            for (k, &jitter_val) in jitters.iter().enumerate() {
                let t_uniform = if n == 1 {
                    0.0_f32
                } else {
                    (k as f32 / (n - 1) as f32) - 0.5
                };
                let t = t_uniform + jitter_val;

                let sx = px as f32 + t * blur_vx;
                let sy = py as f32 + t * blur_vy;
                let sample = mb_bilinear_sample(image, w, h, sx, sy);

                let tw = mb_triangle_weight(t);
                let dw = if config.depth_aware {
                    let sample_x = sx.clamp(0.0, w as f32 - 1.0).round() as usize;
                    let sample_y = sy.clamp(0.0, h as f32 - 1.0).round() as usize;
                    let si = sample_y * w + sample_x;
                    let sd = depth_buf.map(|d| d[si]).unwrap_or(center_depth);
                    mb_depth_weight(center_depth, sd, config.depth_sigma)
                } else {
                    1.0
                };

                let wc = tw * dw;
                acc[0] += sample[0] * wc;
                acc[1] += sample[1] * wc;
                acc[2] += sample[2] * wc;
                weight_sum += wc;
            }

            if weight_sum > 1e-8 {
                let inv = 1.0 / weight_sum;
                let ob = pi * 3;
                output[ob] = acc[0] * inv;
                output[ob + 1] = acc[1] * inv;
                output[ob + 2] = acc[2] * inv;
            }
        }
    }
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Full pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an [`MbStats`] *estimate* from a velocity buffer and config.
///
/// This is a fast, config-only estimate that never runs the actual blur
/// pass — see [`MbStats::estimated_sample_utilization`] for why it should
/// not be read as a measurement.
pub fn mb_compute_stats(velocity: &VelocityBuffer, config: &MbConfig) -> MbStats {
    let n = velocity.width * velocity.height;
    if n == 0 {
        return MbStats {
            mean_blur_pixels: 0.0,
            max_blur_pixels: 0.0,
            blurred_fraction: 0.0,
            estimated_sample_utilization: 0.0,
        };
    }

    let scale = (config.shutter_angle / 360.0) * config.exposure;
    let mut sum_mag = 0.0_f32;
    let mut max_mag = 0.0_f32;
    let mut blurred_count = 0usize;

    for i in 0..n {
        let vx = velocity.velocity_x[i] * scale;
        let vy = velocity.velocity_y[i] * scale;
        let mag = (vx * vx + vy * vy).sqrt().min(config.max_blur_pixels);
        sum_mag += mag;
        if mag > max_mag {
            max_mag = mag;
        }
        if mag > 0.5 {
            blurred_count += 1;
        }
    }

    let mean_blur = sum_mag / n as f32;
    let blurred_fraction = blurred_count as f32 / n as f32;
    let estimated_sample_utilization = if mean_blur < 1e-6 {
        1.0
    } else {
        let util = (mean_blur / config.max_blur_pixels).min(1.0);
        1.0 + (config.n_samples as f32 - 1.0) * util
    };

    MbStats {
        mean_blur_pixels: mean_blur,
        max_blur_pixels: max_mag,
        blurred_fraction,
        estimated_sample_utilization,
    }
}

/// Full motion blur pipeline: validate → apply → compute stats.
///
/// # Errors
///
/// Propagates errors from [`mb_apply`].
pub fn apply_motion_blur(
    image: &[f32],
    velocity: &VelocityBuffer,
    depth_buf: Option<&[f32]>,
    config: &MbConfig,
) -> Result<MotionBlurResult, MotionBlurError> {
    let blurred = mb_apply(image, velocity, depth_buf, config)?;
    let stats = mb_compute_stats(velocity, config);
    Ok(MotionBlurResult {
        image: blurred,
        stats,
    })
}

/// Format [`MbConfig`] as a human-readable string.
pub fn mb_format_config(config: &MbConfig) -> String {
    format!(
        "MbConfig {{ shutter_angle: {:.1}°, n_samples: {}, exposure: {:.2}, \
         max_blur_pixels: {:.1}, depth_aware: {}, depth_sigma: {:.3}, jitter: {} }}",
        config.shutter_angle,
        config.n_samples,
        config.exposure,
        config.max_blur_pixels,
        config.depth_aware,
        config.depth_sigma,
        config.jitter,
    )
}

/// Format [`MbStats`] as a human-readable string.
pub fn mb_format_stats(stats: &MbStats) -> String {
    format!(
        "MbStats {{ mean_blur: {:.2}px, max_blur: {:.2}px, blurred_fraction: {:.3}, \
         estimated_sample_utilization: {:.2} }}",
        stats.mean_blur_pixels,
        stats.max_blur_pixels,
        stats.blurred_fraction,
        stats.estimated_sample_utilization,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── MbConfig ──────────────────────────────────────────────────────────────
    #[test]
    fn mb_config_default_fields() {
        let c = MbConfig::default();
        assert!(approx(c.shutter_angle, 180.0));
        assert_eq!(c.n_samples, 16);
        assert!(approx(c.exposure, 1.0));
        assert!(approx(c.max_blur_pixels, 32.0));
        assert!(c.depth_aware);
        assert!(approx(c.depth_sigma, 0.5));
        assert!(c.jitter);
    }
    #[test]
    fn mb_config_validate_ok() {
        assert!(MbConfig::default().validate().is_ok());
    }
    #[test]
    fn mb_config_validate_bad_shutter_zero() {
        let c = MbConfig {
            shutter_angle: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(MotionBlurError::InvalidShutterAngle { .. })
        ));
    }
    #[test]
    fn mb_config_validate_bad_shutter_too_large() {
        let c = MbConfig {
            shutter_angle: 361.0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(MotionBlurError::InvalidShutterAngle { .. })
        ));
    }
    #[test]
    fn mb_config_validate_zero_samples() {
        let c = MbConfig {
            n_samples: 0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(MotionBlurError::InvalidSampleCount { .. })
        ));
    }

    // ── VelocityBuffer ────────────────────────────────────────────────────────
    #[test]
    fn velocity_buffer_new_all_zeros() {
        let vb = VelocityBuffer::new(4, 3);
        assert_eq!(vb.velocity_x.len(), 12);
        assert_eq!(vb.velocity_y.len(), 12);
        assert!(vb.velocity_x.iter().all(|&v| v == 0.0));
        assert!(vb.velocity_y.iter().all(|&v| v == 0.0));
    }
    #[test]
    fn velocity_buffer_from_vecs_ok() {
        let vb = VelocityBuffer::from_vecs(vec![1.0; 6], vec![2.0; 6], 3, 2).expect("ok");
        assert!(approx(vb.velocity_x[0], 1.0));
        assert!(approx(vb.velocity_y[0], 2.0));
    }
    #[test]
    fn velocity_buffer_from_vecs_mismatch_x() {
        let res = VelocityBuffer::from_vecs(vec![0.0; 5], vec![0.0; 6], 3, 2);
        assert!(matches!(
            res,
            Err(MotionBlurError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn velocity_buffer_from_vecs_mismatch_y() {
        let res = VelocityBuffer::from_vecs(vec![0.0; 6], vec![0.0; 7], 3, 2);
        assert!(matches!(
            res,
            Err(MotionBlurError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn velocity_buffer_magnitude_345() {
        let mut vb = VelocityBuffer::new(3, 3);
        vb.velocity_x[4] = 3.0;
        vb.velocity_y[4] = 4.0;
        assert!(approx(vb.magnitude(1, 1), 5.0));
    }
    #[test]
    fn velocity_buffer_magnitude_oob() {
        assert!(approx(VelocityBuffer::new(3, 3).magnitude(100, 100), 0.0));
    }
    #[test]
    fn velocity_buffer_max_magnitude() {
        let mut vb = VelocityBuffer::new(2, 2);
        vb.velocity_x[0] = 3.0;
        vb.velocity_y[0] = 4.0;
        vb.velocity_x[1] = 1.0;
        vb.velocity_y[1] = 0.0;
        assert!(approx(vb.max_magnitude(), 5.0));
    }
    #[test]
    fn velocity_buffer_mean_magnitude() {
        let mut vb = VelocityBuffer::new(2, 1);
        vb.velocity_x[0] = 3.0;
        vb.velocity_y[0] = 4.0;
        assert!(approx(vb.mean_magnitude(), 2.5));
    }

    // ── mb_velocity_from_camera_motion ────────────────────────────────────────
    #[test]
    fn mb_velocity_camera_motion_uniform() {
        let vb = mb_velocity_from_camera_motion(4, 4, 2.0, -1.0);
        assert!(vb.velocity_x.iter().all(|&v| approx(v, 2.0)));
        assert!(vb.velocity_y.iter().all(|&v| approx(v, -1.0)));
    }
    #[test]
    fn mb_velocity_camera_motion_size() {
        let vb = mb_velocity_from_camera_motion(5, 3, 1.0, 0.0);
        assert_eq!(vb.velocity_x.len(), 15);
    }

    // ── mb_velocity_from_depth ────────────────────────────────────────────────
    #[test]
    fn mb_velocity_from_depth_zero_bg() {
        let vb = mb_velocity_from_depth(&[0.0_f32; 4], 2, 2, 1.0, 1.0).expect("ok");
        assert!(vb.velocity_x.iter().all(|&v| approx(v, 0.0)));
    }
    #[test]
    fn mb_velocity_from_depth_nonzero() {
        let vb = mb_velocity_from_depth(&[1.0_f32; 4], 2, 2, 1.0, 1.0).expect("ok");
        assert!(vb.velocity_x.iter().all(|&v| v > 0.0));
    }
    #[test]
    fn mb_velocity_from_depth_mismatch() {
        let res = mb_velocity_from_depth(&[1.0; 3], 2, 2, 1.0, 1.0);
        assert!(matches!(
            res,
            Err(MotionBlurError::DimensionMismatch { .. })
        ));
    }

    // ── mb_velocity_rotational ────────────────────────────────────────────────
    #[test]
    fn mb_velocity_rotational_center_zero() {
        let vb = mb_velocity_rotational(5, 5, 2.0, 2.0, 1.0);
        assert!(vb.magnitude(2, 2) < 1e-6);
    }
    #[test]
    fn mb_velocity_rotational_corner_nonzero() {
        let vb = mb_velocity_rotational(5, 5, 2.0, 2.0, 1.0);
        assert!(vb.magnitude(0, 0) > 0.0);
    }

    // ── mb_smooth_velocity ────────────────────────────────────────────────────
    #[test]
    fn mb_smooth_velocity_zero_remains_zero() {
        let vb = VelocityBuffer::new(4, 4);
        let s = mb_smooth_velocity(&vb, 1.0).expect("ok");
        assert!(s.velocity_x.iter().all(|&v| approx(v, 0.0)));
    }
    #[test]
    fn mb_smooth_velocity_uniform_unchanged() {
        let vb = mb_velocity_from_camera_motion(4, 4, 3.0, 1.0);
        let s = mb_smooth_velocity(&vb, 1.0).expect("ok");
        for (&a, &b) in vb.velocity_x.iter().zip(s.velocity_x.iter()) {
            assert!((a - b).abs() < 0.01, "uniform vx changed: {a} vs {b}");
        }
    }
    #[test]
    fn mb_smooth_velocity_empty_error() {
        let vb = VelocityBuffer::new(0, 0);
        assert!(matches!(
            mb_smooth_velocity(&vb, 1.0),
            Err(MotionBlurError::EmptyInput)
        ));
    }
    #[test]
    fn mb_smooth_velocity_zero_sigma_identity() {
        let vb = mb_velocity_from_camera_motion(3, 3, 5.0, 2.0);
        let s = mb_smooth_velocity(&vb, 0.0).expect("ok");
        for (&a, &b) in vb.velocity_x.iter().zip(s.velocity_x.iter()) {
            assert!(approx(a, b));
        }
    }

    // ── mb_dilate_velocity ────────────────────────────────────────────────────
    #[test]
    fn mb_dilate_velocity_propagates() {
        let mut vb = VelocityBuffer::new(3, 1);
        vb.velocity_x[0] = 5.0;
        let depth = vec![1.0_f32; 3];
        let d = mb_dilate_velocity(&vb, &depth, 2).expect("ok");
        assert!(d.velocity_x[1] > 0.0 || d.velocity_x[2] > 0.0);
    }
    #[test]
    fn mb_dilate_velocity_mismatch() {
        let vb = VelocityBuffer::new(2, 2);
        assert!(matches!(
            mb_dilate_velocity(&vb, &[1.0; 3], 1),
            Err(MotionBlurError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn mb_dilate_velocity_zero_passes() {
        let vb = mb_velocity_from_camera_motion(3, 3, 1.0, 0.0);
        let depth = vec![1.0_f32; 9];
        let d = mb_dilate_velocity(&vb, &depth, 0).expect("ok");
        assert!(approx(d.velocity_x[0], 1.0));
    }

    // ── mb_bilinear_sample ────────────────────────────────────────────────────
    #[test]
    fn mb_bilinear_sample_exact_pixel() {
        let img = vec![
            1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
        ];
        let p = mb_bilinear_sample(&img, 2, 2, 0.0, 0.0);
        assert!(approx(p[0], 1.0) && approx(p[1], 0.0) && approx(p[2], 0.0));
    }
    #[test]
    fn mb_bilinear_sample_uniform_midpoint() {
        let img = vec![1.0_f32; 2 * 2 * 3];
        let p = mb_bilinear_sample(&img, 2, 2, 0.5, 0.5);
        assert!(approx(p[0], 1.0) && approx(p[1], 1.0) && approx(p[2], 1.0));
    }
    #[test]
    fn mb_bilinear_sample_clamping_negative() {
        let img = vec![0.5_f32; 2 * 2 * 3];
        let p = mb_bilinear_sample(&img, 2, 2, -5.0, -5.0);
        assert!(approx(p[0], 0.5));
    }
    #[test]
    fn mb_bilinear_sample_clamping_overflow() {
        let img = vec![0.5_f32; 2 * 2 * 3];
        let p = mb_bilinear_sample(&img, 2, 2, 100.0, 100.0);
        assert!(approx(p[0], 0.5));
    }
    #[test]
    fn mb_bilinear_sample_one_pixel_wide_no_panic() {
        // Regression: `width - 1.0 - f32::EPSILON` is negative for width == 1,
        // which used to panic inside `f32::clamp` (requires min <= max).
        let img = vec![0.3_f32, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1];
        let p = mb_bilinear_sample(&img, 1, 3, 5.0, 1.0);
        assert!(approx(p[0], 0.6) && approx(p[1], 0.7) && approx(p[2], 0.8));
    }
    #[test]
    fn mb_bilinear_sample_one_pixel_tall_no_panic() {
        let img = vec![0.2_f32, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let p = mb_bilinear_sample(&img, 3, 1, 1.0, -7.0);
        assert!(approx(p[0], 0.5) && approx(p[1], 0.6) && approx(p[2], 0.7));
    }
    #[test]
    fn mb_bilinear_sample_1x1_no_panic() {
        let img = vec![0.42_f32, 0.24, 0.99];
        let p = mb_bilinear_sample(&img, 1, 1, -3.0, 8.0);
        assert!(approx(p[0], 0.42) && approx(p[1], 0.24) && approx(p[2], 0.99));
    }

    // ── mb_triangle_weight ────────────────────────────────────────────────────
    #[test]
    fn mb_triangle_weight_center() {
        assert!(approx(mb_triangle_weight(0.0), 1.0));
    }
    #[test]
    fn mb_triangle_weight_pos_edge() {
        assert!(approx(mb_triangle_weight(0.5), 0.0));
    }
    #[test]
    fn mb_triangle_weight_neg_edge() {
        assert!(approx(mb_triangle_weight(-0.5), 0.0));
    }
    #[test]
    fn mb_triangle_weight_quarter() {
        assert!(approx(mb_triangle_weight(0.25), 0.5));
    }
    #[test]
    fn mb_triangle_weight_beyond() {
        assert!(approx(mb_triangle_weight(1.0), 0.0));
    }

    // ── mb_depth_weight ───────────────────────────────────────────────────────
    #[test]
    fn mb_depth_weight_same() {
        assert!(approx(mb_depth_weight(1.0, 1.0, 0.5), 1.0));
    }
    #[test]
    fn mb_depth_weight_large_diff() {
        assert!(mb_depth_weight(0.0, 100.0, 0.5) < 0.01);
    }
    #[test]
    fn mb_depth_weight_sigma_scale() {
        let w1 = mb_depth_weight(0.0, 1.0, 0.5);
        let w2 = mb_depth_weight(0.0, 1.0, 2.0);
        assert!(w2 > w1, "larger sigma should give larger weight");
    }

    // ── mb_jitter_samples ─────────────────────────────────────────────────────
    #[test]
    fn mb_jitter_samples_length() {
        let j = mb_jitter_samples(8, &mut 12345_u64);
        assert_eq!(j.len(), 8);
    }
    #[test]
    fn mb_jitter_samples_small_values() {
        let j = mb_jitter_samples(16, &mut 99999_u64);
        for &v in &j {
            assert!(v.abs() <= 0.5, "jitter {v} out of bound");
        }
    }
    #[test]
    fn mb_jitter_samples_deterministic() {
        let j1 = mb_jitter_samples(8, &mut 42_u64);
        let j2 = mb_jitter_samples(8, &mut 42_u64);
        for (&a, &b) in j1.iter().zip(j2.iter()) {
            assert!(approx(a, b));
        }
    }
    #[test]
    fn mb_fill_jitter_samples_matches_allocating_twin() {
        // Regression (per-pixel allocation removal): the in-place fill must
        // draw exactly the sequence `mb_jitter_samples` draws, from the same
        // state, and must leave the state at the same place.
        for &(n, seed) in &[(1_usize, 7_u64), (4, 42), (8, 1), (16, 0xDEAD_BEEF)] {
            let mut state_alloc = seed;
            let expected = mb_jitter_samples(n, &mut state_alloc);

            let mut state_fill = seed;
            let mut buf = vec![f32::NAN; n];
            mb_fill_jitter_samples(&mut buf, &mut state_fill);

            assert_eq!(buf, expected, "value drift for n={n}, seed={seed}");
            assert_eq!(
                state_fill, state_alloc,
                "rng state drift for n={n}, seed={seed}"
            );
        }
    }

    #[test]
    fn mb_fill_jitter_samples_empty_is_noop() {
        let mut state = 12345_u64;
        let mut empty: [f32; 0] = [];
        mb_fill_jitter_samples(&mut empty, &mut state);
        assert_eq!(state, 12345_u64, "empty fill must not advance the rng");
    }

    #[test]
    fn mb_apply_jitter_stream_skips_static_pixels() {
        // Regression: `rng_state` advances only for pixels that actually get
        // blurred (the `mag < 1e-8` early-continue fires first). A hoisted
        // buffer that refills unconditionally would shift the stream for any
        // image containing static regions, so a half-static velocity buffer
        // must not produce the same result as an all-moving one, and each
        // must be bit-reproducible across calls.
        let w = 4_usize;
        let h = 2_usize;
        let img: Vec<f32> = (0..(w * h * 3)).map(|i| (i % 7) as f32 * 0.1).collect();
        let cfg = MbConfig {
            jitter: true,
            depth_aware: false,
            n_samples: 5,
            ..MbConfig::default()
        };

        let all_moving = mb_velocity_from_camera_motion(w, h, 3.0, 1.0);
        let mut half_static = mb_velocity_from_camera_motion(w, h, 3.0, 1.0);
        for i in 0..(w * h) {
            if i.is_multiple_of(2) {
                half_static.velocity_x[i] = 0.0;
                half_static.velocity_y[i] = 0.0;
            }
        }

        let moving_a = mb_apply(&img, &all_moving, None, &cfg).expect("all-moving blur");
        let moving_b = mb_apply(&img, &all_moving, None, &cfg).expect("all-moving blur again");
        assert_eq!(moving_a, moving_b, "mb_apply must be deterministic");

        let mixed_a = mb_apply(&img, &half_static, None, &cfg).expect("half-static blur");
        let mixed_b = mb_apply(&img, &half_static, None, &cfg).expect("half-static blur again");
        assert_eq!(mixed_a, mixed_b, "mb_apply must be deterministic");

        // Odd (moving) pixels consume jitter draws 0..n in the mixed case but
        // draws n..2n in the all-moving case, so at least one must differ.
        assert!(
            moving_a
                .iter()
                .zip(mixed_a.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "static pixels must not consume jitter draws"
        );
    }

    #[test]
    fn mb_apply_no_jitter_buffer_stays_zeroed() {
        // With `jitter: false` the hoisted scratch buffer is never written, so
        // every pixel must see the same pure-uniform sample positions the old
        // per-pixel `vec![0.0; n]` produced: a uniform image survives intact.
        let w = 5_usize;
        let h = 3_usize;
        let img = vec![0.42_f32; w * h * 3];
        let cfg = MbConfig {
            jitter: false,
            depth_aware: false,
            n_samples: 7,
            ..MbConfig::default()
        };
        let vb = mb_velocity_from_camera_motion(w, h, 2.0, -1.5);
        let out = mb_apply(&img, &vb, None, &cfg).expect("blur");
        for &v in &out {
            assert!(approx(v, 0.42), "uniform image changed: {v}");
        }
    }

    #[test]
    fn mb_jitter_samples_different_seeds() {
        let j1 = mb_jitter_samples(8, &mut 1_u64);
        let j2 = mb_jitter_samples(8, &mut 2_u64);
        assert!(j1
            .iter()
            .zip(j2.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-8));
    }
    #[test]
    fn mb_jitter_samples_zero_n() {
        assert!(mb_jitter_samples(0, &mut 1_u64).is_empty());
    }

    // ── mb_apply ──────────────────────────────────────────────────────────────
    #[test]
    fn mb_apply_zero_velocity_passthrough() {
        let img = vec![0.5_f32; 4 * 4 * 3];
        let vb = VelocityBuffer::new(4, 4);
        let cfg = MbConfig {
            jitter: false,
            ..MbConfig::default()
        };
        let out = mb_apply(&img, &vb, None, &cfg).expect("ok");
        for (&a, &b) in img.iter().zip(out.iter()) {
            assert!(approx(a, b));
        }
    }
    #[test]
    fn mb_apply_uniform_image_unchanged() {
        let img = vec![0.7_f32; 4 * 4 * 3];
        let vb = mb_velocity_from_camera_motion(4, 4, 5.0, 3.0);
        let cfg = MbConfig {
            jitter: false,
            depth_aware: false,
            ..MbConfig::default()
        };
        let out = mb_apply(&img, &vb, None, &cfg).expect("ok");
        for &v in &out {
            assert!((v - 0.7).abs() < 0.01, "got {v}");
        }
    }
    #[test]
    fn mb_apply_n_samples_one_no_crash() {
        let img: Vec<f32> = (0..9 * 3).map(|i| i as f32 / 100.0).collect();
        let vb = mb_velocity_from_camera_motion(3, 3, 2.0, 1.0);
        let cfg = MbConfig {
            n_samples: 1,
            jitter: false,
            depth_aware: false,
            ..MbConfig::default()
        };
        assert!(mb_apply(&img, &vb, None, &cfg).is_ok());
    }
    #[test]
    fn mb_apply_large_velocity_blurs() {
        let img: Vec<f32> = (0..8 * 8 * 3)
            .map(|i| if (i / 3) % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let vb = mb_velocity_from_camera_motion(8, 8, 10.0, 0.0);
        let cfg = MbConfig {
            jitter: false,
            depth_aware: false,
            ..MbConfig::default()
        };
        let out = mb_apply(&img, &vb, None, &cfg).expect("ok");
        let changed = img
            .iter()
            .zip(out.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-3);
        assert!(changed, "large velocity should blur the image");
    }
    #[test]
    fn mb_apply_dimension_mismatch() {
        let vb = VelocityBuffer::new(4, 4);
        let cfg = MbConfig {
            jitter: false,
            ..MbConfig::default()
        };
        let res = mb_apply(&[0.0; 10], &vb, None, &cfg);
        assert!(matches!(
            res,
            Err(MotionBlurError::DimensionMismatch { .. })
        ));
    }
    #[test]
    fn mb_apply_invalid_shutter_zero() {
        let vb = VelocityBuffer::new(4, 4);
        let cfg = MbConfig {
            shutter_angle: 0.0,
            ..MbConfig::default()
        };
        let res = mb_apply(&[0.0; 4 * 4 * 3], &vb, None, &cfg);
        assert!(matches!(
            res,
            Err(MotionBlurError::InvalidShutterAngle { .. })
        ));
    }

    // ── apply_motion_blur pipeline ────────────────────────────────────────────
    #[test]
    fn apply_motion_blur_zero_vel_passthrough() {
        let img = vec![0.3_f32; 4 * 4 * 3];
        let vb = VelocityBuffer::new(4, 4);
        let cfg = MbConfig {
            jitter: false,
            ..MbConfig::default()
        };
        let res = apply_motion_blur(&img, &vb, None, &cfg).expect("ok");
        for (&a, &b) in img.iter().zip(res.image.iter()) {
            assert!(approx(a, b));
        }
    }
    #[test]
    fn apply_motion_blur_zero_vel_stats() {
        let img = vec![0.3_f32; 4 * 4 * 3];
        let vb = VelocityBuffer::new(4, 4);
        let cfg = MbConfig {
            jitter: false,
            ..MbConfig::default()
        };
        let res = apply_motion_blur(&img, &vb, None, &cfg).expect("ok");
        assert!(approx(res.stats.blurred_fraction, 0.0));
    }
    #[test]
    fn apply_motion_blur_nonzero_stats() {
        let img = vec![0.5_f32; 4 * 4 * 3];
        let vb = mb_velocity_from_camera_motion(4, 4, 5.0, 0.0);
        let cfg = MbConfig {
            jitter: false,
            depth_aware: false,
            ..MbConfig::default()
        };
        let res = apply_motion_blur(&img, &vb, None, &cfg).expect("ok");
        assert!(res.stats.mean_blur_pixels > 0.0);
    }

    // ── mb_compute_stats ──────────────────────────────────────────────────────
    #[test]
    fn mb_compute_stats_zero() {
        let stats = mb_compute_stats(&VelocityBuffer::new(4, 4), &MbConfig::default());
        assert!(approx(stats.mean_blur_pixels, 0.0));
        assert!(approx(stats.max_blur_pixels, 0.0));
        assert!(approx(stats.blurred_fraction, 0.0));
    }
    #[test]
    fn mb_compute_stats_nonzero() {
        let vb = mb_velocity_from_camera_motion(4, 4, 10.0, 0.0);
        let stats = mb_compute_stats(&vb, &MbConfig::default());
        assert!(stats.mean_blur_pixels > 0.0);
    }

    // ── mb_format_config / mb_format_stats ────────────────────────────────────
    #[test]
    fn mb_format_config_nonempty() {
        let s = mb_format_config(&MbConfig::default());
        assert!(!s.is_empty() && s.contains("shutter_angle"));
    }
    #[test]
    fn mb_format_stats_nonempty() {
        let stats = mb_compute_stats(&VelocityBuffer::new(2, 2), &MbConfig::default());
        let s = mb_format_stats(&stats);
        assert!(!s.is_empty() && s.contains("mean_blur"));
    }

    // ── shutter_angle effect ──────────────────────────────────────────────────
    #[test]
    fn shutter_angle_doubles_blur() {
        let vb = mb_velocity_from_camera_motion(4, 4, 10.0, 0.0);
        let s180 = mb_compute_stats(
            &vb,
            &MbConfig {
                shutter_angle: 180.0,
                ..MbConfig::default()
            },
        );
        let s360 = mb_compute_stats(
            &vb,
            &MbConfig {
                shutter_angle: 360.0,
                ..MbConfig::default()
            },
        );
        assert!(s360.mean_blur_pixels >= s180.mean_blur_pixels);
    }

    // ── depth_aware effect ────────────────────────────────────────────────────
    #[test]
    fn depth_aware_differs_from_no_depth() {
        let w = 4;
        let h = 4;
        let img: Vec<f32> = (0..w * h * 3).map(|i| i as f32 / 100.0).collect();
        let vb = mb_velocity_from_camera_motion(w, h, 8.0, 0.0);
        let depth: Vec<f32> = (0..w * h).map(|i| (i as f32 + 0.1) * 0.5).collect();
        let cfg_d = MbConfig {
            depth_aware: true,
            jitter: false,
            n_samples: 4,
            ..MbConfig::default()
        };
        let cfg_nd = MbConfig {
            depth_aware: false,
            jitter: false,
            n_samples: 4,
            ..MbConfig::default()
        };
        let od = mb_apply(&img, &vb, Some(&depth), &cfg_d).expect("ok");
        let on = mb_apply(&img, &vb, Some(&depth), &cfg_nd).expect("ok");
        assert!(
            od.iter()
                .zip(on.iter())
                .any(|(&a, &b)| (a - b).abs() > 1e-6),
            "depth_aware vs no-depth should differ"
        );
    }

    // ── jitter determinism ────────────────────────────────────────────────────
    #[test]
    fn jitter_same_seed_deterministic() {
        let img = vec![0.5_f32; 4 * 4 * 3];
        let vb = mb_velocity_from_camera_motion(4, 4, 5.0, 2.0);
        let cfg = MbConfig {
            jitter: true,
            depth_aware: false,
            ..MbConfig::default()
        };
        let out1 = mb_apply(&img, &vb, None, &cfg).expect("ok");
        let out2 = mb_apply(&img, &vb, None, &cfg).expect("ok");
        for (&a, &b) in out1.iter().zip(out2.iter()) {
            assert!(approx(a, b));
        }
    }
}
