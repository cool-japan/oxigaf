//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    RayMarchResult, TransferFunction, VolumeGrid, VolumetricCamera, VolumetricIntegration,
    VolumetricRay, VolumetricRenderConfig, VolumetricRenderError, VolumetricStats,
};

/// Splat 3DGS Gaussians into a voxel density grid using isotropic Gaussian
/// footprints (scale used as sigma; rotation ignored for simplicity).
///
/// `positions` has stride 3 (x,y,z), `scales` has stride 3 (sx,sy,sz),
/// `opacities` has stride 1.
pub fn vr_gaussians_to_volume(
    positions: &[f32],
    scales: &[f32],
    opacities: &[f32],
    n_gaussians: usize,
    grid: &mut VolumeGrid,
) -> Result<(), VolumetricRenderError> {
    if positions.len() != n_gaussians * 3 {
        return Err(VolumetricRenderError::BufferLengthMismatch {
            field: "positions",
            expected: n_gaussians * 3,
            got: positions.len(),
        });
    }
    if scales.len() != n_gaussians * 3 {
        return Err(VolumetricRenderError::BufferLengthMismatch {
            field: "scales",
            expected: n_gaussians * 3,
            got: scales.len(),
        });
    }
    if opacities.len() != n_gaussians {
        return Err(VolumetricRenderError::BufferLengthMismatch {
            field: "opacities",
            expected: n_gaussians,
            got: opacities.len(),
        });
    }
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    for g in 0..n_gaussians {
        let gx = positions[g * 3];
        let gy = positions[g * 3 + 1];
        let gz = positions[g * 3 + 2];
        let sigma = (scales[g * 3] + scales[g * 3 + 1] + scales[g * 3 + 2]) / 3.0;
        let sigma = sigma.max(1e-6_f32);
        let opacity = opacities[g].clamp(0.0, 1.0);
        let r_world = 3.0 * sigma;
        let x_lo = (gx - r_world - grid.origin[0]) / grid.voxel_size[0];
        let x_hi = (gx + r_world - grid.origin[0]) / grid.voxel_size[0];
        let y_lo = (gy - r_world - grid.origin[1]) / grid.voxel_size[1];
        let y_hi = (gy + r_world - grid.origin[1]) / grid.voxel_size[1];
        let z_lo = (gz - r_world - grid.origin[2]) / grid.voxel_size[2];
        let z_hi = (gz + r_world - grid.origin[2]) / grid.voxel_size[2];
        let ix0 = (x_lo.floor() as i64).clamp(0, nx as i64 - 1) as usize;
        let ix1 = (x_hi.ceil() as i64).clamp(0, nx as i64 - 1) as usize;
        let iy0 = (y_lo.floor() as i64).clamp(0, ny as i64 - 1) as usize;
        let iy1 = (y_hi.ceil() as i64).clamp(0, ny as i64 - 1) as usize;
        let iz0 = (z_lo.floor() as i64).clamp(0, nz as i64 - 1) as usize;
        let iz1 = (z_hi.ceil() as i64).clamp(0, nz as i64 - 1) as usize;
        let inv2s2 = 0.5 / (sigma * sigma);
        for iz in iz0..=iz1 {
            for iy in iy0..=iy1 {
                for ix in ix0..=ix1 {
                    let wx = grid.origin[0] + (ix as f32 + 0.5) * grid.voxel_size[0];
                    let wy = grid.origin[1] + (iy as f32 + 0.5) * grid.voxel_size[1];
                    let wz = grid.origin[2] + (iz as f32 + 0.5) * grid.voxel_size[2];
                    let d2 = (wx - gx).powi(2) + (wy - gy).powi(2) + (wz - gz).powi(2);
                    let weight = opacity * (-d2 * inv2s2).exp();
                    grid.data[iz * ny * nx + iy * nx + ix] += weight;
                }
            }
        }
    }
    Ok(())
}
/// Slab-method ray–AABB intersection against the volume grid's bounding box.
///
/// Returns `Some((t_near, t_far))` if the ray intersects the box (even if the
/// ray origin is inside), or `None` if it misses.
pub fn vr_ray_aabb_intersect(ray: &VolumetricRay, grid: &VolumeGrid) -> Option<(f32, f32)> {
    let min = grid.origin;
    let max = [
        grid.origin[0] + grid.nx as f32 * grid.voxel_size[0],
        grid.origin[1] + grid.ny as f32 * grid.voxel_size[1],
        grid.origin[2] + grid.nz as f32 * grid.voxel_size[2],
    ];
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    for i in 0..3 {
        let d = ray.direction[i];
        let o = ray.origin[i];
        if d.abs() < f32::EPSILON {
            if o < min[i] || o > max[i] {
                return None;
            }
        } else {
            let t1 = (min[i] - o) / d;
            let t2 = (max[i] - o) / d;
            let (tlo, thi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            t_near = t_near.max(tlo);
            t_far = t_far.min(thi);
        }
    }
    if t_near > t_far {
        return None;
    }
    if t_far < 0.0 {
        return None;
    }
    Some((t_near.max(0.0), t_far))
}
#[inline(always)]
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}
/// Returns a value in `[0, 1)` from the PRNG.
#[inline(always)]
fn xorshift64_f32(state: &mut u64) -> f32 {
    let v = xorshift64(state);
    (v as f32) / (u64::MAX as f32 + 1.0)
}
/// March a single ray through the volume and integrate colour + opacity.
pub fn vr_march_ray(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
) -> RayMarchResult {
    let Some((t_entry, t_exit)) = vr_ray_aabb_intersect(ray, volume) else {
        return RayMarchResult::default();
    };
    let step = config.step_size;
    let mut prng = config.jitter_seed;
    if prng == 0 {
        prng = 1;
    }
    let jitter_offset = if config.jitter {
        xorshift64_f32(&mut prng) * step
    } else {
        0.0
    };
    let t_start = t_entry + jitter_offset;
    let integration = config.integration;
    match integration {
        VolumetricIntegration::FrontToBack => {
            vr_march_front_to_back(ray, volume, tf, config, t_start, t_exit, t_entry)
        }
        VolumetricIntegration::BackToFront => {
            vr_march_back_to_front(ray, volume, tf, config, t_start, t_exit, t_entry)
        }
        VolumetricIntegration::Mip => {
            vr_march_mip(ray, volume, tf, config, t_start, t_exit, t_entry)
        }
        VolumetricIntegration::Avg => {
            vr_march_avg(ray, volume, tf, config, t_start, t_exit, t_entry)
        }
    }
}
fn vr_march_front_to_back(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    t_start: f32,
    t_exit: f32,
    t_entry: f32,
) -> RayMarchResult {
    let mut color = [0.0_f32; 3];
    let mut alpha = 0.0_f32;
    let mut n_steps = 0usize;
    let mut t = t_start;
    while t <= t_exit && n_steps < config.max_steps {
        let p = ray.at(t);
        let density = volume.sample_trilinear(p[0], p[1], p[2]);
        let (s_color, s_alpha) = tf.evaluate(density);
        let s_alpha = 1.0 - (1.0 - s_alpha).powf(config.step_size);
        let contrib = (1.0 - alpha) * s_alpha;
        color[0] += contrib * s_color[0];
        color[1] += contrib * s_color[1];
        color[2] += contrib * s_color[2];
        alpha += contrib;
        n_steps += 1;
        if alpha >= config.early_termination_alpha {
            break;
        }
        t += config.step_size;
    }
    RayMarchResult {
        color,
        alpha,
        n_steps,
        t_entry,
        t_exit,
    }
}
fn vr_march_back_to_front(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    t_start: f32,
    t_exit: f32,
    t_entry: f32,
) -> RayMarchResult {
    let mut samples: Vec<([f32; 3], f32)> = Vec::new();
    let mut t = t_start;
    while t <= t_exit && samples.len() < config.max_steps {
        let p = ray.at(t);
        let density = volume.sample_trilinear(p[0], p[1], p[2]);
        let (s_color, s_alpha) = tf.evaluate(density);
        let s_alpha = 1.0 - (1.0 - s_alpha).powf(config.step_size);
        samples.push((s_color, s_alpha));
        t += config.step_size;
    }
    let n_steps = samples.len();
    let mut color = [0.0_f32; 3];
    let mut alpha = 0.0_f32;
    for (s_color, s_alpha) in samples.iter().rev() {
        color[0] = s_color[0] * s_alpha + color[0] * (1.0 - s_alpha);
        color[1] = s_color[1] * s_alpha + color[1] * (1.0 - s_alpha);
        color[2] = s_color[2] * s_alpha + color[2] * (1.0 - s_alpha);
        alpha = s_alpha + alpha * (1.0 - s_alpha);
    }
    RayMarchResult {
        color,
        alpha,
        n_steps,
        t_entry,
        t_exit,
    }
}
fn vr_march_mip(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    t_start: f32,
    t_exit: f32,
    t_entry: f32,
) -> RayMarchResult {
    let mut max_density = 0.0_f32;
    let mut t = t_start;
    let mut n_steps = 0usize;
    while t <= t_exit && n_steps < config.max_steps {
        let p = ray.at(t);
        let density = volume.sample_trilinear(p[0], p[1], p[2]);
        if density > max_density {
            max_density = density;
        }
        n_steps += 1;
        t += config.step_size;
    }
    let (color, alpha) = tf.evaluate(max_density);
    RayMarchResult {
        color,
        alpha,
        n_steps,
        t_entry,
        t_exit,
    }
}
fn vr_march_avg(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    t_start: f32,
    t_exit: f32,
    t_entry: f32,
) -> RayMarchResult {
    let mut sum_color = [0.0_f32; 3];
    let mut sum_alpha = 0.0_f32;
    let mut t = t_start;
    let mut n_steps = 0usize;
    while t <= t_exit && n_steps < config.max_steps {
        let p = ray.at(t);
        let density = volume.sample_trilinear(p[0], p[1], p[2]);
        let (s_color, s_alpha) = tf.evaluate(density);
        sum_color[0] += s_color[0];
        sum_color[1] += s_color[1];
        sum_color[2] += s_color[2];
        sum_alpha += s_alpha;
        n_steps += 1;
        t += config.step_size;
    }
    if n_steps == 0 {
        return RayMarchResult::default();
    }
    let inv_n = 1.0 / n_steps as f32;
    RayMarchResult {
        color: [
            sum_color[0] * inv_n,
            sum_color[1] * inv_n,
            sum_color[2] * inv_n,
        ],
        alpha: (sum_alpha * inv_n).clamp(0.0, 1.0),
        n_steps,
        t_entry,
        t_exit,
    }
}
/// Render the full image as an RGBA `f32` flat buffer (width × height × 4).
pub fn vr_render_image(
    volume: &VolumeGrid,
    tf: &TransferFunction,
    camera: &VolumetricCamera,
    config: &VolumetricRenderConfig,
) -> Result<Vec<[f32; 4]>, VolumetricRenderError> {
    if volume.nx == 0 || volume.ny == 0 || volume.nz == 0 {
        return Err(VolumetricRenderError::ZeroSizeVolume {
            nx: volume.nx,
            ny: volume.ny,
            nz: volume.nz,
        });
    }
    if camera.width == 0 || camera.height == 0 {
        return Err(VolumetricRenderError::InvalidCamera(
            "width and height must be > 0".into(),
        ));
    }
    let w = camera.width as usize;
    let h = camera.height as usize;
    let mut out = vec![[0.0_f32; 4]; w * h];
    let base_seed = config.jitter_seed;
    for py in 0..h {
        for px in 0..w {
            let mut pixel_config = config.clone();
            if config.jitter {
                let mut s = base_seed ^ ((py as u64).wrapping_mul(0x_9e37_79b9) ^ px as u64);
                if s == 0 {
                    s = 1;
                }
                pixel_config.jitter_seed = s;
            }
            let ray = camera.generate_ray(px as f32, py as f32);
            let result = vr_march_ray(&ray, volume, tf, &pixel_config);
            out[py * w + px] = [
                result.color[0],
                result.color[1],
                result.color[2],
                result.alpha,
            ];
        }
    }
    Ok(out)
}
/// Render the full image as a flat RGBA `u8` buffer (width × height × 4).
/// Colours are clamped to `[0, 1]` and multiplied by 255.
pub fn vr_render_image_u8(
    volume: &VolumeGrid,
    tf: &TransferFunction,
    camera: &VolumetricCamera,
    config: &VolumetricRenderConfig,
) -> Result<Vec<u8>, VolumetricRenderError> {
    let rgba_f32 = vr_render_image(volume, tf, camera, config)?;
    let mut out = Vec::with_capacity(rgba_f32.len() * 4);
    for [r, g, b, a] in rgba_f32 {
        out.push((r.clamp(0.0, 1.0) * 255.0) as u8);
        out.push((g.clamp(0.0, 1.0) * 255.0) as u8);
        out.push((b.clamp(0.0, 1.0) * 255.0) as u8);
        out.push((a.clamp(0.0, 1.0) * 255.0) as u8);
    }
    Ok(out)
}
/// Build a coarse occupancy grid by max-pooling the density grid.
///
/// Each occupancy voxel covers `factor × factor × factor` original voxels.
/// An occupancy voxel is `> 0` if any child voxel density exceeds `threshold`.
pub fn vr_build_occupancy_grid(volume: &VolumeGrid, threshold: f32, factor: usize) -> VolumeGrid {
    let factor = factor.max(1);
    let onx = volume.nx.div_ceil(factor);
    let ony = volume.ny.div_ceil(factor);
    let onz = volume.nz.div_ceil(factor);
    let vox_size = [
        volume.voxel_size[0] * factor as f32,
        volume.voxel_size[1] * factor as f32,
        volume.voxel_size[2] * factor as f32,
    ];
    let mut occ = VolumeGrid::new(onx, ony, onz, volume.origin, vox_size);
    for oz in 0..onz {
        for oy in 0..ony {
            for ox in 0..onx {
                let mut occupied = 0.0_f32;
                'outer: for dz in 0..factor {
                    for dy in 0..factor {
                        for dx in 0..factor {
                            let ix = ox * factor + dx;
                            let iy = oy * factor + dy;
                            let iz = oz * factor + dz;
                            if ix < volume.nx && iy < volume.ny && iz < volume.nz {
                                let d = volume.density_at(ix, iy, iz);
                                if d > threshold {
                                    occupied = 1.0;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
                occ.data[oz * ony * onx + oy * onx + ox] = occupied;
            }
        }
    }
    occ
}
/// Returns `true` if the current marching position is in empty space according
/// to the occupancy grid (density ≈ 0 in the coarse grid).
pub fn vr_can_skip(occupancy: &VolumeGrid, ray: &VolumetricRay, t: f32, _step_size: f32) -> bool {
    let p = ray.at(t);
    occupancy.sample_nearest(p[0], p[1], p[2]) < 0.5
}
/// Compute statistics from a slice of ray march results.
pub fn vr_compute_stats(results: &[RayMarchResult]) -> VolumetricStats {
    if results.is_empty() {
        return VolumetricStats {
            n_rays: 0,
            mean_steps_per_ray: 0.0,
            max_steps_per_ray: 0,
            mean_alpha: 0.0,
            fully_opaque_rays: 0,
            empty_rays: 0,
        };
    }
    let n = results.len();
    let mut total_steps = 0usize;
    let mut max_steps = 0usize;
    let mut total_alpha = 0.0_f32;
    let mut fully_opaque = 0usize;
    let mut empty = 0usize;
    for r in results {
        total_steps += r.n_steps;
        if r.n_steps > max_steps {
            max_steps = r.n_steps;
        }
        total_alpha += r.alpha;
        if r.alpha > 0.99 {
            fully_opaque += 1;
        }
        if r.n_steps == 0 {
            empty += 1;
        }
    }
    VolumetricStats {
        n_rays: n,
        mean_steps_per_ray: total_steps as f32 / n as f32,
        max_steps_per_ray: max_steps,
        mean_alpha: total_alpha / n as f32,
        fully_opaque_rays: fully_opaque,
        empty_rays: empty,
    }
}
/// Format a `VolumetricStats` as a human-readable string.
pub fn vr_format_stats(stats: &VolumetricStats) -> String {
    format!(
        "VolumetricStats {{ n_rays: {}, mean_steps: {:.2}, max_steps: {}, \
         mean_alpha: {:.4}, fully_opaque: {}, empty: {} }}",
        stats.n_rays,
        stats.mean_steps_per_ray,
        stats.max_steps_per_ray,
        stats.mean_alpha,
        stats.fully_opaque_rays,
        stats.empty_rays,
    )
}
/// Format a `VolumetricRenderConfig` as a human-readable string.
pub fn vr_format_config(config: &VolumetricRenderConfig) -> String {
    format!(
        "VolumetricRenderConfig {{ step_size: {}, max_steps: {}, \
         early_term_alpha: {}, integration: {:?}, jitter: {} }}",
        config.step_size,
        config.max_steps,
        config.early_termination_alpha,
        config.integration,
        config.jitter,
    )
}
#[inline]
pub(super) fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub(super) fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub(super) fn vec3_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub(super) fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
pub(super) fn vec3_norm(a: [f32; 3]) -> [f32; 3] {
    let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if len < f32::EPSILON {
        a
    } else {
        [a[0] / len, a[1] / len, a[2] / len]
    }
}
