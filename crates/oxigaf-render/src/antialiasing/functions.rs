//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AaConfig, AaError, AaMethod, AaStats, AliasingStats, MipSplatConfig};

/// Numerically stable sigmoid: `1 / (1 + exp(-x))`.
#[inline]
pub(super) fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}
/// Logit (inverse sigmoid): `ln(p / (1 - p))`, numerically clamped.
///
/// Clamps `p` to `[1e-7, 1 - 1e-7]` before computing the logarithm so the
/// result is always finite.
#[inline]
pub(super) fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-7_f32, 1.0_f32 - 1e-7_f32);
    (p / (1.0_f32 - p)).ln()
}
/// Compute the screen-space radius of a Gaussian given its 3-D scale and
/// camera parameters.
///
/// Uses a simple perspective projection formula:
///
/// ```text
/// screen_radius_px = scale_3d * focal_length / camera_distance
/// ```
///
/// # Arguments
///
/// * `scale_3d` — largest of the 3 Gaussian scales (in world units, **not**
///   log-space).
/// * `camera_distance` — Euclidean distance from the camera to the Gaussian
///   centre (world units).  Clamped to a minimum of `1e-7` to prevent
///   division by zero.
/// * `focal_length` — vertical focal length of the image in pixels (`f_y`).
///
/// The result is clamped to `[0.0, 10_000.0]`.
pub fn compute_screen_radius_px(scale_3d: f32, camera_distance: f32, focal_length: f32) -> f32 {
    let safe_dist = camera_distance.max(1e-7_f32);
    let radius = scale_3d * focal_length / safe_dist;
    radius.clamp(0.0_f32, 10_000.0_f32)
}
/// Compute the opacity scale factor based on the Gaussian's screen-space radius.
///
/// Implements a linear ramp between [`MipSplatConfig::opacity_ramp_min_px`]
/// and [`MipSplatConfig::opacity_ramp_max_px`]:
///
/// | Screen radius               | Returned multiplier          |
/// |-----------------------------|------------------------------|
/// | < `opacity_ramp_min_px`     | `0.0`                        |
/// | > `opacity_ramp_max_px`     | `1.0`                        |
/// | in between                  | linear interpolation `(0, 1)` |
///
/// The multiplier is always in `[0.0, 1.0]`.
pub fn opacity_scale_from_screen_radius(screen_radius_px: f32, config: &MipSplatConfig) -> f32 {
    if screen_radius_px < config.opacity_ramp_min_px {
        return 0.0_f32;
    }
    if screen_radius_px > config.opacity_ramp_max_px {
        return 1.0_f32;
    }
    let range = config.opacity_ramp_max_px - config.opacity_ramp_min_px;
    if range <= 0.0_f32 {
        return 1.0_f32;
    }
    let t = (screen_radius_px - config.opacity_ramp_min_px) / range;
    t.clamp(0.0_f32, 1.0_f32)
}
/// Compute the 3-D scale multiplier required to ensure a minimum screen-space
/// projected extent.
///
/// If the Gaussian already projects to at least `config.min_2d_radius_px`
/// pixels, the minimum-radius term is `1.0` (no change); otherwise it is the
/// factor by which the Gaussian's 3-D scale must be multiplied so that the
/// projected radius equals `min_2d_radius_px`.
///
/// When `config.use_distance_lod` is set, a second, independent term is
/// computed: `camera_distance / config.reference_distance`, clamped to
/// `[1.0, config.max_distance_scale]`. This scales up Gaussians farther than
/// `reference_distance` even if their projected radius already clears
/// `min_2d_radius_px`, which is useful as a coarse distance-LOD bias
/// (larger splats at a distance leave fewer sub-pixel cracks). With the
/// default `max_distance_scale = 1.0` this term is always `1.0`, i.e. a
/// no-op, regardless of `use_distance_lod` — raise `max_distance_scale`
/// to actually enable it.
///
/// The two terms are combined by `max` (whichever wants more expansion
/// wins) and the returned value is clamped to `[1.0, 10.0]` (never shrink,
/// at most 10× expansion).
///
/// # Arguments
///
/// * `scale_3d` — largest Gaussian scale in world units (not log-space).
/// * `camera_distance` — distance from camera to Gaussian centre (world units).
/// * `focal_length` — focal length in pixels.
pub fn compute_scale_compensation(
    scale_3d: f32,
    camera_distance: f32,
    focal_length: f32,
    config: &MipSplatConfig,
) -> f32 {
    let screen_radius = compute_screen_radius_px(scale_3d, camera_distance, focal_length);
    let min_radius_term = if screen_radius >= config.min_2d_radius_px {
        1.0_f32
    } else {
        let safe_scale = scale_3d.max(1e-10_f32);
        let safe_dist = camera_distance.max(1e-7_f32);
        config.min_2d_radius_px * safe_dist / (focal_length * safe_scale)
    };
    let distance_lod_term = if config.use_distance_lod {
        let safe_reference = config.reference_distance.max(1e-7_f32);
        let raw = camera_distance.max(0.0_f32) / safe_reference;
        raw.clamp(1.0_f32, config.max_distance_scale.max(1.0_f32))
    } else {
        1.0_f32
    };
    min_radius_term
        .max(distance_lod_term)
        .clamp(1.0_f32, 10.0_f32)
}
/// Apply Mip-Splatting anti-aliasing to a Gaussian model.
///
/// Iterates over every Gaussian and computes:
///
/// 1. The Euclidean distance from `camera_pos` to the Gaussian centre.
/// 2. The maximum 3-D scale (`exp(max(scales[i]))`).
/// 3. The projected screen-space radius in pixels.
/// 4. A **scale compensation** factor (≥ 1) to bring sub-pixel Gaussians up to
///    the minimum projected size.
/// 5. An **opacity multiplier** (in `[0, 1]`) that fades small Gaussians.
///
/// The stored logit-space opacities are updated in place in the returned
/// vector.  If the opacity multiplier is zero the logit is set to
/// `f32::NEG_INFINITY` (probability 0).
///
/// # Arguments
///
/// * `positions` — per-Gaussian world-space position `[x, y, z]`.
/// * `scales` — per-Gaussian log-space scales `[s0, s1, s2]`.
/// * `opacities` — per-Gaussian logit-space opacities.
/// * `camera_pos` — camera position in world space `[x, y, z]`.
/// * `focal_length` — vertical focal length in pixels.
/// * `config` — anti-aliasing configuration.
///
/// # Returns
///
/// `(new_opacities_logit, scale_compensations, stats)`
///
/// * `new_opacities_logit` — updated logit opacities, same length as input.
/// * `scale_compensations` — per-Gaussian scale compensation factors (≥ 1.0).
/// * `stats` — aggregate statistics.
///
/// # Errors
///
/// [`AaError::InvalidParam`] if `scales` or `opacities` does not have the
/// same length as `positions`.
///
/// # Panics
///
/// Does not panic; all slice accesses are bounds-checked.
pub fn try_apply_antialiasing(
    positions: &[[f32; 3]],
    scales: &[[f32; 3]],
    opacities: &[f32],
    camera_pos: [f32; 3],
    focal_length: f32,
    config: &MipSplatConfig,
) -> Result<(Vec<f32>, Vec<f32>, AliasingStats), AaError> {
    let n = positions.len();
    if scales.len() != n || opacities.len() != n {
        return Err(AaError::InvalidParam(format!(
            "apply_antialiasing: positions.len()={n}, scales.len()={}, \
             opacities.len()={} must all match",
            scales.len(),
            opacities.len()
        )));
    }
    let mut new_opacities = Vec::with_capacity(n);
    let mut scale_compensations = Vec::with_capacity(n);
    let mut num_scaled_up: usize = 0;
    let mut num_faded: usize = 0;
    let mut num_culled: usize = 0;
    let mut total_scale_comp: f32 = 0.0;
    let mut total_opacity_reduction: f32 = 0.0;
    for i in 0..n {
        let pos = positions[i];
        let scale_log = scales[i];
        let opacity_logit = opacities[i];
        let dx = pos[0] - camera_pos[0];
        let dy = pos[1] - camera_pos[1];
        let dz = pos[2] - camera_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let max_log_scale = scale_log[0].max(scale_log[1]).max(scale_log[2]);
        let max_scale_3d = max_log_scale.exp();
        let screen_radius = compute_screen_radius_px(max_scale_3d, distance, focal_length);
        let scale_comp = compute_scale_compensation(max_scale_3d, distance, focal_length, config);
        let opacity_mult = opacity_scale_from_screen_radius(screen_radius, config);
        if scale_comp > 1.0_f32 {
            num_scaled_up += 1;
        }
        if opacity_mult == 0.0_f32 {
            num_culled += 1;
        } else if opacity_mult < 1.0_f32 {
            num_faded += 1;
        }
        total_scale_comp += scale_comp;
        total_opacity_reduction += 1.0_f32 - opacity_mult;
        let new_opacity_logit = if opacity_mult <= 0.0_f32 {
            f32::NEG_INFINITY
        } else if opacity_mult >= 1.0_f32 {
            opacity_logit
        } else {
            let p = sigmoid(opacity_logit);
            let new_p = p * opacity_mult;
            logit(new_p)
        };
        new_opacities.push(new_opacity_logit);
        scale_compensations.push(scale_comp);
    }
    let mean_scale_compensation = if n > 0 {
        total_scale_comp / n as f32
    } else {
        1.0_f32
    };
    let mean_opacity_reduction = if n > 0 {
        total_opacity_reduction / n as f32
    } else {
        0.0_f32
    };
    let stats = AliasingStats {
        num_gaussians: n,
        num_scaled_up,
        num_faded,
        num_culled,
        mean_scale_compensation,
        mean_opacity_reduction,
    };
    Ok((new_opacities, scale_compensations, stats))
}
/// Apply Mip-Splatting anti-aliasing to a Gaussian model.
///
/// Compatibility wrapper around [`try_apply_antialiasing`] that preserves
/// this function's original infallible signature. On a length mismatch
/// between `positions`, `scales`, and `opacities` — which
/// [`try_apply_antialiasing`] reports as `Err(AaError::InvalidParam)` — this
/// wrapper logs a `tracing::error!` and falls back to returning empty
/// vectors with a zeroed [`AliasingStats`], exactly as this function always
/// has. New callers should prefer [`try_apply_antialiasing`], which reports
/// the mismatch instead of silently discarding data; this wrapper exists so
/// existing callers of the non-`Result` signature keep compiling.
///
/// See [`try_apply_antialiasing`] for the full parameter and algorithm
/// documentation.
pub fn apply_antialiasing(
    positions: &[[f32; 3]],
    scales: &[[f32; 3]],
    opacities: &[f32],
    camera_pos: [f32; 3],
    focal_length: f32,
    config: &MipSplatConfig,
) -> (Vec<f32>, Vec<f32>, AliasingStats) {
    match try_apply_antialiasing(
        positions,
        scales,
        opacities,
        camera_pos,
        focal_length,
        config,
    ) {
        Ok(result) => result,
        Err(err) => {
            tracing::error!(
                error = %err,
                "apply_antialiasing: falling back to empty result; call \
                 try_apply_antialiasing to handle this as an error instead"
            );
            let stats = AliasingStats {
                num_gaussians: 0,
                num_scaled_up: 0,
                num_faded: 0,
                num_culled: 0,
                mean_scale_compensation: 1.0,
                mean_opacity_reduction: 0.0,
            };
            (Vec::new(), Vec::new(), stats)
        }
    }
}
/// BT.709 luminance: `0.2126·r + 0.7152·g + 0.0722·b`.
#[inline]
pub fn aa_luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126_f32 * r + 0.7152_f32 * g + 0.0722_f32 * b
}
/// Build a per-pixel luminance map from a flat RGB image (`H×W×3`, `f32`).
///
/// Returns `Err(AaError::SizeMismatch)` when the slice length does not match
/// `width * height * 3`.
pub fn aa_luminance_map(img: &[f32], width: usize, height: usize) -> Result<Vec<f32>, AaError> {
    let expected = width * height * 3;
    if img.len() != expected {
        return Err(AaError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels: 3,
        });
    }
    let mut luma = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        let r = img[i * 3];
        let g = img[i * 3 + 1];
        let b = img[i * 3 + 2];
        luma.push(aa_luminance(r, g, b));
    }
    Ok(luma)
}
/// Clamp-to-border pixel fetch (returns `[0,0,0]` for out-of-bounds coords).
///
/// This is intentionally clamp-to-**border** (not clamp-to-edge): callers
/// that need an actual image value at the boundary — e.g. bilinear
/// resampling near an edge — should use [`aa_sample_pixel_edge`] instead,
/// or this will pull in black and darken the result.
#[inline]
pub fn aa_sample_pixel(img: &[f32], width: usize, height: usize, x: i32, y: i32) -> [f32; 3] {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return [0.0_f32; 3];
    }
    let idx = (y as usize * width + x as usize) * 3;
    [img[idx], img[idx + 1], img[idx + 2]]
}
/// Clamp-to-**edge** pixel fetch: out-of-bounds coordinates are clamped to
/// the nearest valid pixel rather than returning black. This is the correct
/// fetch for any resampling done near the image boundary (bilinear taps,
/// neighbour taps for edge-detection filters); using the clamp-to-border
/// [`aa_sample_pixel`] there pulls in a false black sample and darkens
/// border pixels proportionally to the blend amount.
#[inline]
pub fn aa_sample_pixel_edge(img: &[f32], width: usize, height: usize, x: i32, y: i32) -> [f32; 3] {
    if width == 0 || height == 0 {
        return [0.0_f32; 3];
    }
    let cx = x.clamp(0, width as i32 - 1) as usize;
    let cy = y.clamp(0, height as i32 - 1) as usize;
    let idx = (cy * width + cx) * 3;
    [img[idx], img[idx + 1], img[idx + 2]]
}
/// Bilinear sample at fractional pixel coordinates (clamp-to-edge).
pub fn aa_bilinear_sample(img: &[f32], width: usize, height: usize, x: f32, y: f32) -> [f32; 3] {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let tx = x - x.floor();
    let ty = y - y.floor();
    let p00 = aa_sample_pixel_edge(img, width, height, x0, y0);
    let p10 = aa_sample_pixel_edge(img, width, height, x1, y0);
    let p01 = aa_sample_pixel_edge(img, width, height, x0, y1);
    let p11 = aa_sample_pixel_edge(img, width, height, x1, y1);
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        let top = p00[c] * (1.0 - tx) + p10[c] * tx;
        let bot = p01[c] * (1.0 - tx) + p11[c] * tx;
        out[c] = top * (1.0 - ty) + bot * ty;
    }
    out
}
/// Compute a per-pixel edge-strength map.
///
/// For each pixel the maximum luminance contrast against its four NSEW
/// neighbours is computed.  If contrast exceeds `threshold` the edge strength
/// is `contrast / max_contrast`; otherwise it is `0.0`.
pub fn aa_edge_map(luma: &[f32], width: usize, height: usize, threshold: f32) -> Vec<f32> {
    let n = width * height;
    let mut edges = vec![0.0_f32; n];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let center = luma[idx];
            let north = if y > 0 {
                luma[(y - 1) * width + x]
            } else {
                center
            };
            let south = if y + 1 < height {
                luma[(y + 1) * width + x]
            } else {
                center
            };
            let west = if x > 0 {
                luma[y * width + x - 1]
            } else {
                center
            };
            let east = if x + 1 < width {
                luma[y * width + x + 1]
            } else {
                center
            };
            let local_max = center.max(north).max(south).max(west).max(east);
            let local_min = center.min(north).min(south).min(west).min(east);
            let contrast = local_max - local_min;
            if contrast > threshold {
                let max_range = local_max.max(1e-6_f32);
                edges[idx] = (contrast / max_range).clamp(0.0_f32, 1.0_f32);
            }
        }
    }
    edges
}
/// Count pixels whose luminance contrast exceeds `threshold`.
pub fn aa_edge_count(luma: &[f32], width: usize, height: usize, threshold: f32) -> usize {
    let edges = aa_edge_map(luma, width, height, threshold);
    edges.iter().filter(|&&e| e > 0.0_f32).count()
}
#[inline]
fn check_size(img: &[f32], width: usize, height: usize) -> Result<(), AaError> {
    let expected = width * height * 3;
    if img.len() != expected {
        return Err(AaError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels: 3,
        });
    }
    if width < 4 || height < 4 {
        return Err(AaError::ImageTooSmall { width, height });
    }
    Ok(())
}
/// Count how many consecutive texels (up to `max_steps`), starting one
/// texel away from `(x, y)` and walking in the given direction, still show
/// local NSEW contrast at or above `threshold`. The walk stops (without
/// panicking) at the first texel that falls below `threshold` or at the
/// image boundary, whichever comes first. Used by [`apply_fxaa`]'s
/// edge-length search.
#[allow(clippy::too_many_arguments)]
fn fxaa_edge_run(
    luma: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    along_x: bool,
    forward: bool,
    threshold: f32,
    max_steps: usize,
) -> usize {
    let mut count = 0usize;
    for step in 1..=max_steps {
        let delta = step as isize;
        let delta = if forward { delta } else { -delta };
        let (sx, sy) = if along_x {
            (x as isize + delta, y as isize)
        } else {
            (x as isize, y as isize + delta)
        };
        if sx < 0 || sy < 0 || sx as usize >= width || sy as usize >= height {
            break;
        }
        let (sx, sy) = (sx as usize, sy as usize);
        let idx = sy * width + sx;
        let center = luma[idx];
        let north = if sy > 0 { luma[idx - width] } else { center };
        let south = if sy + 1 < height {
            luma[idx + width]
        } else {
            center
        };
        let west = if sx > 0 { luma[idx - 1] } else { center };
        let east = if sx + 1 < width {
            luma[idx + 1]
        } else {
            center
        };
        let local_max = center.max(north).max(south).max(west).max(east);
        let local_min = center.min(north).min(south).min(west).min(east);
        if local_max - local_min < threshold {
            break;
        }
        count += 1;
    }
    count
}
/// FXAA-style luminance-space edge detection and subpixel blending.
///
/// Each pixel is examined with a 5-tap (NSEW + center) luma sample.  When the
/// local contrast exceeds the configured thresholds a blend amount is derived
/// from the 3×3 neighbourhood filter and applied by bilinear-sampling in the
/// dominant gradient direction.
///
/// The blend is further scaled by an **edge-length search**: from each edge
/// pixel, `config.search_steps` texels are checked in both directions along
/// the edge (see this module's private `fxaa_edge_run` helper).
/// `search_steps` is the contrast run length, in texels on each side,
/// required to grant full-strength blending;
/// a run shorter than that (an isolated corner or speck rather than a long,
/// straight edge) is blended at a proportionally reduced strength, never
/// below 50%. Consequently a *larger* `search_steps` is a *stricter*
/// requirement — small/short edges get relatively less blending as
/// `search_steps` grows, which suppresses over-blurring fine detail while
/// still fully smoothing long edges.
pub fn apply_fxaa(
    img: &[f32],
    width: usize,
    height: usize,
    config: &AaConfig,
) -> Result<Vec<f32>, AaError> {
    check_size(img, width, height)?;
    let luma = aa_luminance_map(img, width, height)?;
    let mut out = img.to_vec();
    for y in 0..height {
        for x in 0..width {
            let xi = x as i32;
            let yi = y as i32;
            let lc = luma[y * width + x];
            let ln = if y > 0 { luma[(y - 1) * width + x] } else { lc };
            let ls = if y + 1 < height {
                luma[(y + 1) * width + x]
            } else {
                lc
            };
            let lw = if x > 0 { luma[y * width + x - 1] } else { lc };
            let le = if x + 1 < width {
                luma[y * width + x + 1]
            } else {
                lc
            };
            let range_max = lc.max(ln).max(ls).max(lw).max(le);
            let range_min = lc.min(ln).min(ls).min(lw).min(le);
            let range = range_max - range_min;
            let edge_threshold_val =
                (config.edge_threshold_min).max(range_max * config.edge_threshold);
            if range < edge_threshold_val {
                continue;
            }
            let lnw = if y > 0 && x > 0 {
                luma[(y - 1) * width + x - 1]
            } else {
                lc
            };
            let lne = if y > 0 && x + 1 < width {
                luma[(y - 1) * width + x + 1]
            } else {
                lc
            };
            let lsw = if y + 1 < height && x > 0 {
                luma[(y + 1) * width + x - 1]
            } else {
                lc
            };
            let lse = if y + 1 < height && x + 1 < width {
                luma[(y + 1) * width + x + 1]
            } else {
                lc
            };
            let filter = (2.0 * (ln + ls + lw + le) + lnw + lne + lsw + lse) / 12.0;
            let filter_range = (filter - lc).abs();
            let blend_raw = (filter_range / range.max(1e-8_f32)).clamp(0.0, 1.0);
            let grad_h = (ln + ls - 2.0 * lc).abs();
            let grad_v = (lw + le - 2.0 * lc).abs();
            let is_horizontal = grad_h >= grad_v;
            // Edge-length search: walk up to `search_steps` texels in both
            // directions *along* the edge (perpendicular to the blend
            // direction) to check how far the contrast run extends. A
            // pixel embedded in a long, straight edge confirms the run for
            // the full search window in both directions and gets full
            // blend strength; an isolated corner or speck runs out of
            // contrast almost immediately and its blend is tapered down
            // (never below half strength), which avoids over-blurring
            // small detail while still smoothing long edges fully.
            let search_steps = config.search_steps.max(1);
            let along_x = is_horizontal;
            let pos_run = fxaa_edge_run(
                &luma,
                width,
                height,
                x,
                y,
                along_x,
                true,
                edge_threshold_val,
                search_steps,
            );
            let neg_run = fxaa_edge_run(
                &luma,
                width,
                height,
                x,
                y,
                along_x,
                false,
                edge_threshold_val,
                search_steps,
            );
            let edge_confidence =
                0.5 + 0.5 * (pos_run.min(neg_run) as f32 / search_steps as f32).min(1.0);
            let blend = blend_raw * blend_raw * config.subpixel_quality * edge_confidence;
            let (sx, sy) = if is_horizontal {
                (xi as f32, yi as f32 + 0.5 * blend)
            } else {
                (xi as f32 + 0.5 * blend, yi as f32)
            };
            let sampled = aa_bilinear_sample(img, width, height, sx, sy);
            let idx = (y * width + x) * 3;
            out[idx] = sampled[0];
            out[idx + 1] = sampled[1];
            out[idx + 2] = sampled[2];
        }
    }
    Ok(out)
}
/// Simplified Morphological Anti-Aliasing.
///
/// Uses a 3×3 Sobel gradient magnitude to detect edges (not a Roberts-cross
/// operator — Sobel magnitudes run roughly 4–8× larger for the same visual
/// edge, so `edge_threshold` should be tuned against Sobel's range).  At
/// each edge pixel the output is a 1/3 – 1/3 – 1/3 blend of the pixel and
/// its two dominant-axis neighbours, softening the step discontinuity.
pub fn apply_smaa_lite(
    img: &[f32],
    width: usize,
    height: usize,
    config: &AaConfig,
) -> Result<Vec<f32>, AaError> {
    check_size(img, width, height)?;
    let luma = aa_luminance_map(img, width, height)?;
    let mut out = img.to_vec();
    for y in 0..height {
        for x in 0..width {
            let lc = luma[y * width + x];
            let ln = if y > 0 { luma[(y - 1) * width + x] } else { lc };
            let ls = if y + 1 < height {
                luma[(y + 1) * width + x]
            } else {
                lc
            };
            let lw = if x > 0 { luma[y * width + x - 1] } else { lc };
            let le = if x + 1 < width {
                luma[y * width + x + 1]
            } else {
                lc
            };
            let lnw = if y > 0 && x > 0 {
                luma[(y - 1) * width + x - 1]
            } else {
                lc
            };
            let lne_c = if y > 0 && x + 1 < width {
                luma[(y - 1) * width + x + 1]
            } else {
                lc
            };
            let lsw = if y + 1 < height && x > 0 {
                luma[(y + 1) * width + x - 1]
            } else {
                lc
            };
            let lse = if y + 1 < height && x + 1 < width {
                luma[(y + 1) * width + x + 1]
            } else {
                lc
            };
            let gx = (lne_c + 2.0 * le + lse - lnw - 2.0 * lw - lsw).abs();
            let gy = (lnw + 2.0 * ln + lne_c - lsw - 2.0 * ls - lse).abs();
            let edge_strength = (gx * gx + gy * gy).sqrt();
            if edge_strength < config.edge_threshold {
                continue;
            }
            let grad_h = (ln - lc).abs() + (ls - lc).abs();
            let grad_v = (lw - lc).abs() + (le - lc).abs();
            let idx = (y * width + x) * 3;
            // Clamp-to-edge taps: at the image border, `aa_sample_pixel`
            // would pull in a false black sample and darken the blend by up
            // to a third (see `aa_sample_pixel_edge` doc).
            let (n1, n2) = if grad_h >= grad_v {
                let p_north = aa_sample_pixel_edge(img, width, height, x as i32, y as i32 - 1);
                let p_south = aa_sample_pixel_edge(img, width, height, x as i32, y as i32 + 1);
                (p_north, p_south)
            } else {
                let p_west = aa_sample_pixel_edge(img, width, height, x as i32 - 1, y as i32);
                let p_east = aa_sample_pixel_edge(img, width, height, x as i32 + 1, y as i32);
                (p_west, p_east)
            };
            for c in 0..3 {
                out[idx + c] = (img[idx + c] + n1[c] + n2[c]) / 3.0;
            }
        }
    }
    Ok(out)
}
/// Temporal Anti-Aliasing: exponential moving average with a previous frame.
///
/// `output[i] = blend_factor × previous[i] + (1 − blend_factor) × current[i]`
///
/// `blend_factor` must be in `[0, 1]`.
pub fn apply_temporal_aa(
    current: &[f32],
    previous: &[f32],
    width: usize,
    height: usize,
    blend_factor: f32,
) -> Result<Vec<f32>, AaError> {
    if !(0.0..=1.0).contains(&blend_factor) {
        return Err(AaError::InvalidParam(format!(
            "blend_factor {blend_factor} is not in [0, 1]"
        )));
    }
    check_size(current, width, height)?;
    check_size(previous, width, height)?;
    let len = current.len();
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(blend_factor * previous[i] + (1.0 - blend_factor) * current[i]);
    }
    Ok(out)
}
/// Box-filter downsample anti-aliasing.
///
/// Averages non-overlapping `factor × factor` blocks into a single output
/// pixel.  `factor` must be `2` or `4`.  The output dimensions are
/// `(width / factor) × (height / factor)`.
///
/// Returns `AaError::InvalidParam` if `factor` is not 2 or 4, or if the image
/// dimensions are not evenly divisible by `factor`.
pub fn apply_supersampling_aa(
    img: &[f32],
    width: usize,
    height: usize,
    factor: u32,
) -> Result<Vec<f32>, AaError> {
    if factor != 2 && factor != 4 {
        return Err(AaError::InvalidParam(format!(
            "supersampling factor {factor} must be 2 or 4"
        )));
    }
    if img.len() != width * height * 3 {
        return Err(AaError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels: 3,
        });
    }
    let f = factor as usize;
    if width < f || height < f || !width.is_multiple_of(f) || !height.is_multiple_of(f) {
        return Err(AaError::InvalidParam(format!(
            "image {width}×{height} not evenly divisible by factor {factor}"
        )));
    }
    let out_w = width / f;
    let out_h = height / f;
    let mut out = vec![0.0_f32; out_w * out_h * 3];
    let count = (f * f) as f32;
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut sum = [0.0_f32; 3];
            for dy in 0..f {
                for dx in 0..f {
                    let sy = oy * f + dy;
                    let sx = ox * f + dx;
                    let sidx = (sy * width + sx) * 3;
                    for c in 0..3 {
                        sum[c] += img[sidx + c];
                    }
                }
            }
            let oidx = (oy * out_w + ox) * 3;
            for c in 0..3 {
                out[oidx + c] = sum[c] / count;
            }
        }
    }
    Ok(out)
}
/// Apply image-space anti-aliasing according to `config.method`.
///
/// This is the main entry-point.  For [`AaMethod::Supersampling`] the returned
/// image will be smaller than the input.  For all other methods the output has
/// the same dimensions as the input.
///
/// Note: this is a separate function from `apply_antialiasing` (which operates
/// on Gaussian model data). Use `apply_image_aa` when you have an already-rendered
/// RGB frame buffer.
///
/// **`AaMethod::Temporal` is a passthrough here.** This entry point has no
/// previous-frame parameter, so there is no history to blend against — it
/// validates the size and returns `img` unchanged (the correct behaviour for
/// a first frame with no history anyway). To get actual temporal blending,
/// call [`apply_image_aa_with_history`] with the previous frame's output.
pub fn apply_image_aa(
    img: &[f32],
    width: usize,
    height: usize,
    config: &AaConfig,
) -> Result<Vec<f32>, AaError> {
    match &config.method {
        AaMethod::Fxaa => apply_fxaa(img, width, height, config),
        AaMethod::Smaa => apply_smaa_lite(img, width, height, config),
        AaMethod::Temporal { blend_factor } => {
            let _ = blend_factor;
            check_size(img, width, height)?;
            tracing::warn!(
                "apply_image_aa: AaMethod::Temporal has no previous frame to \
                 blend against through this entry point, so the image is \
                 returned unchanged (correct for a first frame, a silent \
                 no-op for any later one); call apply_image_aa_with_history \
                 with Some(previous_frame) for actual temporal blending"
            );
            Ok(img.to_vec())
        }
        AaMethod::Supersampling { factor } => apply_supersampling_aa(img, width, height, *factor),
    }
}
/// Apply image-space anti-aliasing according to `config.method`, with access
/// to the previous frame for [`AaMethod::Temporal`].
///
/// For [`AaMethod::Temporal`], if `previous` is `Some`, this actually blends
/// against it via [`apply_temporal_aa`] (real temporal blending); if
/// `previous` is `None` it falls back to the same passthrough
/// [`apply_image_aa`] uses (appropriate for a first frame with no history
/// yet). For every other method this simply delegates to [`apply_image_aa`]
/// and `previous` is ignored.
pub fn apply_image_aa_with_history(
    img: &[f32],
    previous: Option<&[f32]>,
    width: usize,
    height: usize,
    config: &AaConfig,
) -> Result<Vec<f32>, AaError> {
    match (&config.method, previous) {
        (AaMethod::Temporal { blend_factor }, Some(prev)) => {
            apply_temporal_aa(img, prev, width, height, *blend_factor)
        }
        _ => apply_image_aa(img, width, height, config),
    }
}
/// Estimate quality improvement introduced by anti-aliasing.
///
/// Computes edge counts before and after using a fixed 5% luminance-threshold,
/// as well as pixel-level difference statistics.
pub fn aa_quality_estimate(
    original: &[f32],
    antialiased: &[f32],
    width: usize,
    height: usize,
) -> Result<AaStats, AaError> {
    check_size(original, width, height)?;
    check_size(antialiased, width, height)?;
    let luma_orig = aa_luminance_map(original, width, height)?;
    let luma_aa = aa_luminance_map(antialiased, width, height)?;
    let threshold = 0.05_f32;
    let edge_pixels_original = aa_edge_count(&luma_orig, width, height, threshold);
    let edge_pixels_after = aa_edge_count(&luma_aa, width, height, threshold);
    let smoothing_ratio = if edge_pixels_original == 0 {
        0.0_f32
    } else {
        1.0 - (edge_pixels_after as f32 / edge_pixels_original as f32)
    };
    let len = original.len();
    let mut sum_diff = 0.0_f32;
    let mut max_diff = 0.0_f32;
    for i in 0..len {
        let d = (original[i] - antialiased[i]).abs();
        sum_diff += d;
        if d > max_diff {
            max_diff = d;
        }
    }
    let mean_difference = if len > 0 { sum_diff / len as f32 } else { 0.0 };
    Ok(AaStats {
        edge_pixels_original,
        edge_pixels_after,
        smoothing_ratio,
        mean_difference,
        max_difference: max_diff,
    })
}
/// Human-readable summary of an [`AaConfig`].
pub fn format_aa_config(config: &AaConfig) -> String {
    let method = match &config.method {
        AaMethod::Fxaa => "FXAA".to_string(),
        AaMethod::Smaa => "SMAA-lite".to_string(),
        AaMethod::Temporal { blend_factor } => {
            format!("Temporal(blend={blend_factor:.3})")
        }
        AaMethod::Supersampling { factor } => format!("Supersampling({factor}×)"),
    };
    format!(
        "AaConfig {{ method={method}, edge_threshold={:.4}, edge_threshold_min={:.4}, \
         subpixel_quality={:.3}, search_steps={} }}",
        config.edge_threshold,
        config.edge_threshold_min,
        config.subpixel_quality,
        config.search_steps
    )
}
/// Human-readable summary of [`AaStats`].
pub fn format_aa_stats(stats: &AaStats) -> String {
    format!(
        "AaStats {{ edges_orig={}, edges_after={}, smoothing_ratio={:.4}, \
         mean_diff={:.6}, max_diff={:.6} }}",
        stats.edge_pixels_original,
        stats.edge_pixels_after,
        stats.smoothing_ratio,
        stats.mean_difference,
        stats.max_difference
    )
}

// Regression tests for fixes made directly to this file. The canonical,
// broader test suite for this module lives in `antialiasing/tests.rs`; this
// module is additive and does not duplicate it.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aa_bilinear_sample_clamps_to_edge_not_border() {
        // Regression for: FXAA/SMAA darkened border pixels because the
        // bilinear sampler pulled in a false black sample past the image
        // edge instead of clamping to the last valid pixel.
        let img = vec![0.9_f32; 4 * 4 * 3];
        let left = aa_bilinear_sample(&img, 4, 4, -0.5, 0.0);
        let right = aa_bilinear_sample(&img, 4, 4, 3.5, 0.0);
        let top = aa_bilinear_sample(&img, 4, 4, 0.0, -0.5);
        let bottom = aa_bilinear_sample(&img, 4, 4, 0.0, 3.5);
        for px in [left, right, top, bottom] {
            for c in 0..3 {
                assert!(
                    (px[c] - 0.9).abs() < 1e-5,
                    "sampling just past a uniform image's border should stay \
                     ~0.9 (clamp-to-edge), got {px:?} (clamp-to-border would \
                     darken toward ~0.45)"
                );
            }
        }
    }

    #[test]
    fn test_aa_sample_pixel_still_clamps_to_border() {
        // aa_sample_pixel itself must keep its original clamp-to-border
        // contract (pinned by antialiasing/tests.rs); only the *bilinear*
        // and SMAA neighbour-tap paths were switched to clamp-to-edge.
        let img = vec![1.0_f32; 4 * 4 * 3];
        assert_eq!(aa_sample_pixel(&img, 4, 4, -1, 0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_try_apply_antialiasing_mismatched_lengths_errors() {
        let cfg = MipSplatConfig::default();
        let positions = vec![[0.0_f32; 3]; 3];
        let scales = vec![[0.0_f32; 3]; 2];
        let opacities = vec![0.0_f32; 3];
        let result = try_apply_antialiasing(&positions, &scales, &opacities, [0.0; 3], 500.0, &cfg);
        assert!(
            matches!(result, Err(AaError::InvalidParam(_))),
            "expected InvalidParam, got {result:?}"
        );
    }

    #[test]
    fn test_apply_antialiasing_compat_shim_matches_try_on_success() {
        let cfg = MipSplatConfig::default();
        let positions = vec![[0.0_f32, 0.0, 10.0]];
        let scales = vec![[0.0_f32, 0.0, 0.0]];
        let opacities = vec![0.0_f32];
        let (new_ops, comps, stats) =
            apply_antialiasing(&positions, &scales, &opacities, [0.0; 3], 500.0, &cfg);
        let (new_ops2, comps2, stats2) =
            try_apply_antialiasing(&positions, &scales, &opacities, [0.0; 3], 500.0, &cfg)
                .expect("matching lengths should succeed");
        assert_eq!(new_ops, new_ops2);
        assert_eq!(comps, comps2);
        assert_eq!(stats.num_gaussians, stats2.num_gaussians);
    }

    #[test]
    fn test_apply_antialiasing_compat_shim_still_empty_on_mismatch() {
        // The old infallible entry point is pinned (by antialiasing/tests.rs)
        // to keep returning an empty result rather than a Result on
        // mismatch; only the new try_ variant reports the error.
        let cfg = MipSplatConfig::default();
        let positions = vec![[0.0_f32; 3]; 3];
        let scales = vec![[0.0_f32; 3]; 2];
        let opacities = vec![0.0_f32; 3];
        let (new_ops, comps, stats) =
            apply_antialiasing(&positions, &scales, &opacities, [0.0; 3], 500.0, &cfg);
        assert!(new_ops.is_empty());
        assert!(comps.is_empty());
        assert_eq!(stats.num_gaussians, 0);
    }

    #[test]
    fn test_apply_smaa_lite_does_not_darken_top_border_edge() {
        // Regression for: `apply_smaa_lite`'s north/south/east/west
        // neighbour taps used the clamp-to-*border* `aa_sample_pixel`
        // (returns black past the image edge) instead of clamp-to-edge, so
        // a horizontal edge touching row 0 pulled in a false black sample
        // from y = -1 and darkened the whole top row's blend by up to a
        // third. With the fix (`aa_sample_pixel_edge`), the north tap
        // clamps to the pixel's own row instead.
        let w = 6usize;
        let h = 6usize;
        let bright = 0.9_f32;
        let dark = 0.1_f32;
        let mut img = vec![dark; w * h * 3];
        for x in 0..w {
            let idx = x * 3;
            img[idx] = bright;
            img[idx + 1] = bright;
            img[idx + 2] = bright;
        }
        let cfg = AaConfig {
            method: AaMethod::Smaa,
            edge_threshold: 0.01,
            ..AaConfig::default()
        };
        let out = apply_smaa_lite(&img, w, h, &cfg).expect("smaa ok");
        // Once the north tap correctly clamps to row 0 (bright) instead of
        // pulling in black from y = -1: (bright + bright + dark) / 3.
        let expected = (bright + bright + dark) / 3.0;
        let buggy = (bright + dark) / 3.0;
        for x in 0..w {
            let v = out[x * 3];
            assert!(
                (v - expected).abs() < 1e-4,
                "top-row pixel {x} should blend toward {expected:.3} \
                 (clamp-to-edge north tap), got {v:.3}; a clamp-to-border \
                 regression would instead give ~{buggy:.3}"
            );
        }
    }

    #[test]
    fn test_apply_fxaa_search_steps_affects_blend_strength() {
        // Regression for: `search_steps` was documented as an FXAA edge-walk
        // parameter but never read. A long, straight vertical edge is used
        // so the edge-length search actually has something to measure;
        // search_steps=1 vs search_steps=12 must produce materially
        // different total output deviation, proving the field now
        // influences behaviour (verified against a reference reimplementation
        // to land the exact threshold with margin).
        let w = 12;
        let h = 8;
        let mut img = vec![0.0_f32; w * h * 3];
        for y in 0..h {
            for x in 6..w {
                let idx = (y * w + x) * 3;
                img[idx] = 1.0;
                img[idx + 1] = 1.0;
                img[idx + 2] = 1.0;
            }
        }
        let cfg_a = AaConfig {
            search_steps: 1,
            ..AaConfig::default()
        };
        let cfg_b = AaConfig {
            search_steps: 12,
            ..AaConfig::default()
        };
        let out_a = apply_fxaa(&img, w, h, &cfg_a).expect("fxaa ok");
        let out_b = apply_fxaa(&img, w, h, &cfg_b).expect("fxaa ok");
        let total_dev =
            |out: &[f32]| -> f32 { img.iter().zip(out).map(|(a, b)| (a - b).abs()).sum() };
        let dev_a = total_dev(&out_a);
        let dev_b = total_dev(&out_b);
        assert!(
            (dev_a - dev_b).abs() > 0.05,
            "search_steps should measurably change FXAA's blend strength: \
             search_steps=1 total_dev={dev_a:.4}, search_steps=12 total_dev={dev_b:.4}"
        );
    }

    #[test]
    fn test_compute_scale_compensation_distance_lod_disabled_by_default() {
        let cfg = MipSplatConfig::default();
        assert!(!cfg.use_distance_lod);
        let comp = compute_scale_compensation(1.0, 1000.0, 500.0, &cfg);
        assert!(
            (comp - 1.0).abs() < 1e-5,
            "LOD disabled (and max_distance_scale defaults to 1.0): expected \
             no scaling regardless of distance, got {comp}"
        );
    }

    #[test]
    fn test_compute_scale_compensation_distance_lod_enabled() {
        let cfg = MipSplatConfig {
            use_distance_lod: true,
            max_distance_scale: 4.0,
            reference_distance: 10.0,
            ..MipSplatConfig::default()
        };
        // Large Gaussian (min-radius term == 1.0) at 4x reference_distance:
        // the LOD term should dominate and clamp at max_distance_scale.
        let comp_far = compute_scale_compensation(1.0, 40.0, 500.0, &cfg);
        assert!(
            (comp_far - 4.0).abs() < 1e-4,
            "expected LOD term to clamp at max_distance_scale=4.0, got {comp_far}"
        );
        // At the reference distance itself, the LOD term is a no-op.
        let comp_at_ref = compute_scale_compensation(1.0, 10.0, 500.0, &cfg);
        assert!(
            (comp_at_ref - 1.0).abs() < 1e-4,
            "expected no LOD scaling at reference_distance, got {comp_at_ref}"
        );
    }

    #[test]
    fn test_apply_image_aa_with_history_blends_when_previous_given() {
        let w = 8;
        let h = 8;
        let current = vec![0.0_f32; w * h * 3];
        let previous = vec![1.0_f32; w * h * 3];
        let cfg = AaConfig {
            method: AaMethod::Temporal { blend_factor: 0.5 },
            ..AaConfig::default()
        };
        let out = apply_image_aa_with_history(&current, Some(&previous), w, h, &cfg)
            .expect("temporal with history should succeed");
        for &v in &out {
            assert!(
                (v - 0.5).abs() < 1e-5,
                "blend_factor=0.5 of current=0 and previous=1 should give 0.5, got {v}"
            );
        }
    }

    #[test]
    fn test_apply_image_aa_with_history_passthrough_without_previous() {
        let w = 8;
        let h = 8;
        let current = vec![0.3_f32; w * h * 3];
        let cfg = AaConfig {
            method: AaMethod::Temporal { blend_factor: 0.5 },
            ..AaConfig::default()
        };
        let out = apply_image_aa_with_history(&current, None, w, h, &cfg)
            .expect("temporal without history should pass through");
        assert_eq!(out, current);
    }

    #[test]
    fn test_apply_image_aa_with_history_non_temporal_delegates() {
        let w = 8;
        let h = 8;
        let img = vec![0.5_f32; w * h * 3];
        let cfg = AaConfig {
            method: AaMethod::Fxaa,
            ..AaConfig::default()
        };
        let out = apply_image_aa_with_history(&img, None, w, h, &cfg).expect("fxaa ok");
        assert_eq!(out.len(), img.len());
    }
}
