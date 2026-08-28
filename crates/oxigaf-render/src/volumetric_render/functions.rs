//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use rayon::prelude::*;

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
    if grid.is_empty() {
        // Nothing to splat into: `(nx as i64 - 1)` below would be -1 for a
        // zero-size dimension, which panics inside `i64::clamp(0, ..)`
        // (min > max). There is no valid voxel to write to either way.
        return Ok(());
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
///
/// Does not use empty-space skipping; see [`vr_march_ray_with_occupancy`] for
/// a variant that jumps over the sample positions an occupancy grid (see
/// [`vr_build_occupancy_grid`]) proves are empty, while sampling the very
/// same positions everywhere else.
pub fn vr_march_ray(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
) -> RayMarchResult {
    vr_march_ray_impl(ray, volume, tf, config, None)
}
/// March a single ray, skipping through regions `occupancy` (see
/// [`vr_build_occupancy_grid`]) marks empty, instead of sampling every
/// `config.step_size` along the way.
///
/// This is an *optimization*, not an approximation: it visits a strict subset
/// of the sample positions [`vr_march_ray`] visits (the sample lattice is
/// preserved exactly), skipping only positions that lie
/// inside a coarse cell the grid marks empty, and folding the contribution
/// those skipped samples would have made back in analytically. Given an
/// occupancy grid built by [`vr_build_occupancy_grid`] (whose dilation makes
/// "cell is empty" imply "every trilinear sample inside it reads a density at
/// or below the build threshold") and a volume that is exactly zero there,
/// the result is identical to [`vr_march_ray`].
///
/// A hand-built occupancy grid that is *not* conservatively dilated gets no
/// such guarantee: marching may then miss density near a coarse-cell border.
pub fn vr_march_ray_with_occupancy(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    occupancy: &VolumeGrid,
) -> RayMarchResult {
    vr_march_ray_impl(ray, volume, tf, config, Some(occupancy))
}
fn vr_march_ray_impl(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    occupancy: Option<&VolumeGrid>,
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
    let span = MarchSpan {
        t_start: t_entry + jitter_offset,
        t_exit,
        t_entry,
        step,
    };
    let integration = config.integration;
    match integration {
        VolumetricIntegration::FrontToBack => {
            vr_march_front_to_back(ray, volume, tf, config, span, occupancy)
        }
        VolumetricIntegration::BackToFront => {
            vr_march_back_to_front(ray, volume, tf, config, span, occupancy)
        }
        VolumetricIntegration::Mip => vr_march_mip(ray, volume, tf, config, span, occupancy),
        VolumetricIntegration::Avg => vr_march_avg(ray, volume, tf, config, span, occupancy),
    }
}

/// The parametric span a marcher walks, together with its sample lattice.
///
/// Sample positions are `t_start + k * step` for `k` in
/// `0..lattice_len(max_steps)`. Both the plain and the occupancy-aware march
/// address samples through this same closed form, so a skipped sample and a
/// taken sample sit at bit-identical positions -- which is what lets
/// empty-space skipping stay an optimization rather than perturbing the
/// quadrature.
#[derive(Debug, Clone, Copy)]
struct MarchSpan {
    /// Position of lattice point 0 (ray entry plus any jitter offset).
    t_start: f32,
    /// Parametric exit point of the volume's AABB.
    t_exit: f32,
    /// Parametric entry point of the volume's AABB (reported verbatim).
    t_entry: f32,
    /// Spacing between lattice points (`config.step_size`).
    step: f32,
}

impl MarchSpan {
    /// Position of lattice point `k`.
    #[inline]
    fn t_at(&self, k: usize) -> f32 {
        self.t_start + (k as f32) * self.step
    }

    /// Number of lattice points on `[t_start, t_exit]`, capped at
    /// `max_steps`.
    ///
    /// A non-positive or non-finite `step` yields the `max_steps` cap, which
    /// reproduces the previous `t += step` loop's behaviour (it made no
    /// progress and stopped on the step cap) without spinning forever.
    fn lattice_len(&self, max_steps: usize) -> usize {
        let extent = self.t_exit - self.t_start;
        // `is_finite` first so a NaN span (from a NaN `t_exit`/`t_start`)
        // yields an empty lattice instead of an undefined comparison.
        if !extent.is_finite() || extent < 0.0 {
            return 0;
        }
        let raw = (extent / self.step).floor();
        let count = if raw.is_finite() {
            (raw as usize).saturating_add(1)
        } else {
            usize::MAX
        };
        count.min(max_steps)
    }
}

/// Occupancy value at or above which a coarse cell counts as occupied.
const OCCUPIED_THRESHOLD: f32 = 0.5;

/// Distance along `ray` from `p` to where it leaves the occupancy cell that
/// contains `p`.
///
/// The cell index is clamped exactly the way [`VolumeGrid::sample_nearest`]
/// clamps it, so the slab this measures is always the very cell whose
/// occupancy value was read.
fn occupancy_cell_exit_distance(occupancy: &VolumeGrid, ray: &VolumetricRay, p: [f32; 3]) -> f32 {
    let dims = [occupancy.nx, occupancy.ny, occupancy.nz];
    let mut best = f32::INFINITY;
    for axis in 0..3 {
        let vs = occupancy.voxel_size[axis];
        // A NaN voxel size fails `is_finite`, so it is skipped rather than
        // silently comparing false in every direction.
        if !vs.is_finite() || vs <= 0.0 || dims[axis] == 0 {
            continue;
        }
        let dir = ray.direction[axis];
        if dir.abs() < 1e-12 {
            // Parallel to this pair of slabs: never leaves through them.
            continue;
        }
        let u = (p[axis] - occupancy.origin[axis]) / vs;
        // `u.floor() as usize` saturates at 0 for negative/NaN inputs, which
        // is the clamp `sample_nearest` applies too.
        let cell = (u.floor() as usize).min(dims[axis] - 1) as f32;
        let bound = occupancy.origin[axis]
            + if dir > 0.0 {
                (cell + 1.0) * vs
            } else {
                cell * vs
            };
        let t_axis = (bound - p[axis]) / dir;
        if t_axis < best {
            best = t_axis;
        }
    }
    if best.is_finite() {
        best.max(0.0)
    } else {
        0.0
    }
}

/// If lattice point `k` sits in a cell `occupancy` marks empty, the first
/// lattice index at or beyond that cell's exit; otherwise `None`.
///
/// The returned index is always greater than `k`, so a marcher cannot stall,
/// and every lattice point strictly between the two lies inside the empty
/// cell (the segment from `p` to the cell exit stays in the cell, which is
/// convex).
fn occupancy_skip_target(
    occupancy: &VolumeGrid,
    ray: &VolumetricRay,
    span: &MarchSpan,
    k: usize,
) -> Option<usize> {
    let t = span.t_at(k);
    let p = ray.at(t);
    if occupancy.sample_nearest(p[0], p[1], p[2]) >= OCCUPIED_THRESHOLD {
        return None;
    }
    let delta = occupancy_cell_exit_distance(occupancy, ray, p);
    let steps = (delta / span.step).ceil();
    let steps = if steps.is_finite() && steps >= 1.0 {
        // `as usize` saturates, so an enormous stride cannot wrap.
        steps as usize
    } else {
        1
    };
    Some(k.saturating_add(steps))
}

/// Per-sample opacity and colour of a sample taken in empty space.
///
/// A skipped sample is a sample of density 0, and a transfer function is free
/// to map density 0 to a non-zero opacity (a global fog). `alpha_per_step` is
/// already converted to the per-`step_size` opacity the marchers composite.
#[derive(Debug, Clone, Copy)]
struct EmptySample {
    color: [f32; 3],
    alpha_per_step: f32,
}

impl EmptySample {
    fn new(tf: &TransferFunction, step: f32) -> Self {
        let (color, alpha) = tf.evaluate(0.0);
        let alpha = alpha.clamp(0.0, 1.0);
        Self {
            color,
            alpha_per_step: 1.0 - (1.0 - alpha).powf(step),
        }
    }

    /// `true` when skipped samples cannot change an alpha-composited result,
    /// which is the case for every transfer function that maps empty space to
    /// full transparency (all of the built-in ones).
    #[inline]
    fn is_transparent(&self) -> bool {
        self.alpha_per_step <= 0.0
    }

    /// Composite `n` empty samples front-to-back into `(color, alpha)`,
    /// stopping as soon as `alpha` reaches `stop_alpha`.
    ///
    /// Returns `true` if the run was cut short by `stop_alpha` (the caller
    /// must then stop marching, exactly as the sample-by-sample loop would).
    /// The closed form is the standard geometric series for a run of
    /// identical samples: after `m` of them the remaining transmittance is
    /// `T * (1 - a)^m`.
    fn composite_run(
        &self,
        color: &mut [f32; 3],
        alpha: &mut f32,
        n: usize,
        stop_alpha: Option<f32>,
    ) -> bool {
        if self.is_transparent() || n == 0 {
            return false;
        }
        let transmittance = 1.0 - *alpha;
        let a = self.alpha_per_step;
        let mut m = n;
        let mut terminated = false;
        if let Some(stop) = stop_alpha {
            // Smallest m with `1 - T*(1-a)^m >= stop`.
            let remaining = (1.0 - stop).max(0.0);
            if transmittance > remaining {
                let base = 1.0 - a;
                if base <= 0.0 {
                    // A single opaque sample already reaches `stop`.
                    m = 1;
                    terminated = true;
                } else {
                    let needed = ((remaining / transmittance).ln() / base.ln()).ceil();
                    if needed.is_finite() && needed >= 0.0 && (needed as usize) <= n {
                        m = (needed as usize).max(1);
                        terminated = true;
                    }
                }
            } else {
                // Already terminated before the run started.
                return true;
            }
        }
        let survive = (1.0 - a).powi(m.min(i32::MAX as usize) as i32);
        let gained = transmittance * (1.0 - survive);
        color[0] += gained * self.color[0];
        color[1] += gained * self.color[1];
        color[2] += gained * self.color[2];
        *alpha += gained;
        terminated
    }
}

fn vr_march_front_to_back(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    span: MarchSpan,
    occupancy: Option<&VolumeGrid>,
) -> RayMarchResult {
    let mut color = [0.0_f32; 3];
    let mut alpha = 0.0_f32;
    let mut n_steps = 0usize;
    let lattice_len = span.lattice_len(config.max_steps);
    let empty = EmptySample::new(tf, config.step_size);
    let mut k = 0usize;
    while k < lattice_len {
        if let Some(occ) = occupancy {
            if let Some(k_next) = occupancy_skip_target(occ, ray, &span, k) {
                let skipped = k_next.min(lattice_len) - k;
                let terminated = empty.composite_run(
                    &mut color,
                    &mut alpha,
                    skipped,
                    Some(config.early_termination_alpha),
                );
                k = k_next;
                if terminated {
                    break;
                }
                continue;
            }
        }
        let p = ray.at(span.t_at(k));
        let density = volume.sample_trilinear(p[0], p[1], p[2]);
        let (s_color, s_alpha) = tf.evaluate(density);
        // Defensive clamp: `s_alpha` should already be in [0, 1] for any
        // `TransferFunction` built via `TransferFunction::new`, but `points`
        // is a public field, and `(1.0 - s_alpha).powf(step_size)` below
        // would otherwise produce NaN for any s_alpha > 1 (a negative base
        // raised to a fractional power).
        let s_alpha = s_alpha.clamp(0.0, 1.0);
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
        k += 1;
    }
    RayMarchResult {
        color,
        alpha,
        n_steps,
        t_entry: span.t_entry,
        t_exit: span.t_exit,
    }
}
fn vr_march_back_to_front(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    span: MarchSpan,
    occupancy: Option<&VolumeGrid>,
) -> RayMarchResult {
    // Back-to-front Porter-Duff "over" compositing over a front-to-back
    // ordered sample sequence produces the same result (up to
    // floating-point rounding) as accumulating those same samples
    // front-to-back with the "under" operator -- the same formula
    // `vr_march_front_to_back` uses -- via the standard compositing-algebra
    // identity: `A over (B over C) == (A over B) over C` for the Porter-Duff
    // `over` operator, so replaying the same samples in the opposite order
    // with the complementary formula yields the same composite. This avoids
    // buffering every sample into a per-ray `Vec` (previously up to
    // `max_steps`, 1000 by default, i.e. up to 16 KB per ray) purely to
    // replay it in reverse. `BackToFront`'s only behavioural difference
    // from `FrontToBack` is that it never early-terminates (see below).
    let mut color = [0.0_f32; 3];
    let mut alpha = 0.0_f32;
    let mut n_steps = 0usize;
    let lattice_len = span.lattice_len(config.max_steps);
    let empty = EmptySample::new(tf, config.step_size);
    let mut k = 0usize;
    while k < lattice_len {
        if let Some(occ) = occupancy {
            if let Some(k_next) = occupancy_skip_target(occ, ray, &span, k) {
                let skipped = k_next.min(lattice_len) - k;
                // No early-termination threshold: `BackToFront` marches the
                // whole ray, so the whole run is folded in.
                empty.composite_run(&mut color, &mut alpha, skipped, None);
                k = k_next;
                continue;
            }
        }
        let p = ray.at(span.t_at(k));
        let density = volume.sample_trilinear(p[0], p[1], p[2]);
        let (s_color, s_alpha) = tf.evaluate(density);
        // See the matching comment in `vr_march_front_to_back`.
        let s_alpha = s_alpha.clamp(0.0, 1.0);
        let s_alpha = 1.0 - (1.0 - s_alpha).powf(config.step_size);
        let contrib = (1.0 - alpha) * s_alpha;
        color[0] += contrib * s_color[0];
        color[1] += contrib * s_color[1];
        color[2] += contrib * s_color[2];
        alpha += contrib;
        n_steps += 1;
        // Deliberately no early termination: unlike `vr_march_front_to_back`,
        // `BackToFront` always marches the full ray.
        k += 1;
    }
    RayMarchResult {
        color,
        alpha,
        n_steps,
        t_entry: span.t_entry,
        t_exit: span.t_exit,
    }
}
fn vr_march_mip(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    span: MarchSpan,
    occupancy: Option<&VolumeGrid>,
) -> RayMarchResult {
    let mut max_density = 0.0_f32;
    let lattice_len = span.lattice_len(config.max_steps);
    let mut n_steps = 0usize;
    let mut k = 0usize;
    while k < lattice_len {
        if let Some(occ) = occupancy {
            if let Some(k_next) = occupancy_skip_target(occ, ray, &span, k) {
                // Skipped samples read density 0, which can never raise
                // `max_density` (it starts at 0), so there is nothing to fold
                // in here.
                k = k_next;
                continue;
            }
        }
        let p = ray.at(span.t_at(k));
        let density = volume.sample_trilinear(p[0], p[1], p[2]);
        if density > max_density {
            max_density = density;
        }
        n_steps += 1;
        k += 1;
    }
    let (color, alpha) = tf.evaluate(max_density);
    RayMarchResult {
        color,
        alpha,
        n_steps,
        t_entry: span.t_entry,
        t_exit: span.t_exit,
    }
}
fn vr_march_avg(
    ray: &VolumetricRay,
    volume: &VolumeGrid,
    tf: &TransferFunction,
    config: &VolumetricRenderConfig,
    span: MarchSpan,
    occupancy: Option<&VolumeGrid>,
) -> RayMarchResult {
    let mut sum_color = [0.0_f32; 3];
    let mut sum_alpha = 0.0_f32;
    let lattice_len = span.lattice_len(config.max_steps);
    // `Avg` divides by the number of samples, so -- unlike every other
    // integration mode -- dropping a sample changes the answer. Skipped
    // lattice points are therefore still *counted*, and the value they would
    // have contributed (a density-0 sample) is added in bulk, which keeps the
    // occupancy-aware march exactly equal to the plain one.
    let (empty_color, empty_alpha) = tf.evaluate(0.0);
    let mut n_steps = 0usize;
    let mut n_lattice = 0usize;
    let mut k = 0usize;
    while k < lattice_len {
        if let Some(occ) = occupancy {
            if let Some(k_next) = occupancy_skip_target(occ, ray, &span, k) {
                let skipped = k_next.min(lattice_len) - k;
                let skipped_f = skipped as f32;
                sum_color[0] += empty_color[0] * skipped_f;
                sum_color[1] += empty_color[1] * skipped_f;
                sum_color[2] += empty_color[2] * skipped_f;
                sum_alpha += empty_alpha * skipped_f;
                n_lattice += skipped;
                k = k_next;
                continue;
            }
        }
        let p = ray.at(span.t_at(k));
        let density = volume.sample_trilinear(p[0], p[1], p[2]);
        let (s_color, s_alpha) = tf.evaluate(density);
        sum_color[0] += s_color[0];
        sum_color[1] += s_color[1];
        sum_color[2] += s_color[2];
        sum_alpha += s_alpha;
        n_steps += 1;
        n_lattice += 1;
        k += 1;
    }
    if n_lattice == 0 {
        return RayMarchResult::default();
    }
    let inv_n = 1.0 / n_lattice as f32;
    RayMarchResult {
        color: [
            sum_color[0] * inv_n,
            sum_color[1] * inv_n,
            sum_color[2] * inv_n,
        ],
        alpha: (sum_alpha * inv_n).clamp(0.0, 1.0),
        n_steps,
        t_entry: span.t_entry,
        t_exit: span.t_exit,
    }
}
/// Render the full image as an RGBA `f32` flat buffer (width × height × 4).
pub fn vr_render_image(
    volume: &VolumeGrid,
    tf: &TransferFunction,
    camera: &VolumetricCamera,
    config: &VolumetricRenderConfig,
) -> Result<Vec<[f32; 4]>, VolumetricRenderError> {
    vr_render_image_impl(volume, tf, camera, config, None)
}
/// Render the full image, skipping through regions `occupancy` (see
/// [`vr_build_occupancy_grid`]) marks empty -- see
/// [`vr_march_ray_with_occupancy`].
pub fn vr_render_image_with_occupancy(
    volume: &VolumeGrid,
    tf: &TransferFunction,
    camera: &VolumetricCamera,
    config: &VolumetricRenderConfig,
    occupancy: &VolumeGrid,
) -> Result<Vec<[f32; 4]>, VolumetricRenderError> {
    vr_render_image_impl(volume, tf, camera, config, Some(occupancy))
}
fn vr_render_image_impl(
    volume: &VolumeGrid,
    tf: &TransferFunction,
    camera: &VolumetricCamera,
    config: &VolumetricRenderConfig,
    occupancy: Option<&VolumeGrid>,
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
    // Parallelise over scanlines: each row's pixels are independent (no
    // shared mutable state), matching the `par_chunks_mut` pattern used
    // elsewhere in this crate (e.g. `denoising.rs`).
    out.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
        for (px, pixel) in row.iter_mut().enumerate() {
            let mut pixel_config = config.clone();
            if config.jitter {
                let mut s = base_seed ^ ((py as u64).wrapping_mul(0x_9e37_79b9) ^ px as u64);
                if s == 0 {
                    s = 1;
                }
                pixel_config.jitter_seed = s;
            }
            let ray = camera.generate_ray(px as f32, py as f32);
            let result = vr_march_ray_impl(&ray, volume, tf, &pixel_config, occupancy);
            *pixel = [
                result.color[0],
                result.color[1],
                result.color[2],
                result.alpha,
            ];
        }
    });
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
/// An occupancy voxel is `> 0` if any voxel density in that block **or in a
/// one-voxel halo around it** exceeds `threshold`.
///
/// # Why the halo
///
/// The marchers read the volume with [`VolumeGrid::sample_trilinear`], whose
/// stencil for a point inside coarse cell `c` reaches one fine voxel beyond
/// `c` on every side: the point's voxel-space coordinate is
/// `(p - origin)/voxel_size - 0.5`, so its `floor` can be the voxel just
/// below the block and `floor + 1` the voxel just above it. Pooling only the
/// block's own voxels would therefore let a cell be marked empty while
/// trilinear samples inside it still pick up density from the occupied cell
/// next door -- and empty-space skipping would silently drop that density.
/// Dilating by one fine voxel makes "cell is empty" imply "every trilinear
/// sample taken inside this cell reads a density of at most `threshold`",
/// which is what makes the skipping in [`vr_march_ray_with_occupancy`] an
/// optimization rather than an approximation.
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
        // Block plus one-voxel halo, clamped to the grid (`saturating_sub`
        // keeps `ox == 0` from wrapping).
        let z_lo = (oz * factor).saturating_sub(1);
        let z_hi = ((oz + 1) * factor + 1).min(volume.nz);
        for oy in 0..ony {
            let y_lo = (oy * factor).saturating_sub(1);
            let y_hi = ((oy + 1) * factor + 1).min(volume.ny);
            for ox in 0..onx {
                let x_lo = (ox * factor).saturating_sub(1);
                let x_hi = ((ox + 1) * factor + 1).min(volume.nx);
                let mut occupied = 0.0_f32;
                'outer: for iz in z_lo..z_hi {
                    for iy in y_lo..y_hi {
                        for ix in x_lo..x_hi {
                            if volume.density_at(ix, iy, iz) > threshold {
                                occupied = 1.0;
                                break 'outer;
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
///
/// This is the point test only. The marchers additionally compute how far the
/// ray stays inside that empty cell (see [`vr_march_ray_with_occupancy`]);
/// advancing by a fixed stride instead can jump clean over an occupied
/// neighbour.
pub fn vr_can_skip(occupancy: &VolumeGrid, ray: &VolumetricRay, t: f32, _step_size: f32) -> bool {
    let p = ray.at(t);
    occupancy.sample_nearest(p[0], p[1], p[2]) < OCCUPIED_THRESHOLD
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// This file's own regression tests. The sibling `tests.rs` (a separate
// module, `super::tests`, not this one) holds the broader existing test
// suite for the whole `volumetric_render` module.
#[cfg(test)]
mod tests {
    use super::*;
    // `TransferPoint` isn't needed by non-test code in this file, so it's
    // imported only here rather than added to the module-level `use`
    // above (which would otherwise be unused outside `#[cfg(test)]`).
    use super::super::types::TransferPoint;

    fn cfg_with(integration: VolumetricIntegration, jitter: bool) -> VolumetricRenderConfig {
        VolumetricRenderConfig {
            step_size: 0.1,
            integration,
            jitter,
            ..VolumetricRenderConfig::default()
        }
    }

    // ── vr_gaussians_to_volume: zero-size grid ────────────────────────────────

    #[test]
    fn test_vr_gaussians_to_volume_zero_size_no_panic() {
        // Regression test: `(x_lo.floor() as i64).clamp(0, nx as i64 - 1)`
        // used to panic (min > max) for any zero-size grid dimension.
        let mut grid = VolumeGrid::new(0, 4, 4, [0.0; 3], [1.0; 3]);
        let positions = [0.0_f32, 0.0, 0.0];
        let scales = [1.0_f32, 1.0, 1.0];
        let opacities = [1.0_f32];
        let result = vr_gaussians_to_volume(&positions, &scales, &opacities, 1, &mut grid);
        assert!(result.is_ok());
    }

    // ── Empty-space skipping ───────────────────────────────────────────────────

    #[test]
    fn test_vr_march_ray_with_occupancy_matches_plain_march_when_fully_occupied() {
        // Sanity check: when every occupancy voxel is "occupied" (built from
        // a volume with density everywhere), `vr_can_skip` never returns
        // true, so `vr_march_ray_with_occupancy` must behave identically to
        // `vr_march_ray`.
        let volume = VolumeGrid::from_fn(8, 8, 8, [-1.0; 3], [0.25; 3], |_, _, _| 0.8);
        let occupancy = vr_build_occupancy_grid(&volume, 0.1, 2);
        let tf = TransferFunction::grayscale(1.0);
        let cfg = cfg_with(VolumetricIntegration::FrontToBack, false);
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);

        let plain = vr_march_ray(&ray, &volume, &tf, &cfg);
        let with_occ = vr_march_ray_with_occupancy(&ray, &volume, &tf, &cfg, &occupancy);

        assert!((plain.alpha - with_occ.alpha).abs() < 1e-4);
        for c in 0..3 {
            assert!((plain.color[c] - with_occ.color[c]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_vr_march_ray_with_occupancy_skips_empty_space() {
        // Regression test for the "empty-space skipping is built but never
        // used" bug: marching through a completely empty volume with an
        // occupancy grid available must take measurably fewer steps than
        // marching without one (the ray should fast-forward through the
        // whole volume instead of sampling at every `step_size`), while
        // still reporting the same (fully transparent) result.
        let volume = VolumeGrid::from_fn(32, 32, 32, [-1.0; 3], [1.0 / 16.0; 3], |_, _, _| 0.0);
        let occupancy = vr_build_occupancy_grid(&volume, 0.1, 4);
        let tf = TransferFunction::grayscale(1.0);
        // BackToFront never early-terminates, so the step count directly
        // reflects how much of the ray was actually sampled.
        let cfg = cfg_with(VolumetricIntegration::BackToFront, false);
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);

        let plain = vr_march_ray(&ray, &volume, &tf, &cfg);
        let with_occ = vr_march_ray_with_occupancy(&ray, &volume, &tf, &cfg, &occupancy);

        assert!(
            with_occ.n_steps < plain.n_steps,
            "expected fewer steps with occupancy skipping: plain={} with_occ={}",
            plain.n_steps,
            with_occ.n_steps
        );
        assert!(with_occ.alpha < 1e-4, "empty volume should be transparent");
        assert!(plain.alpha < 1e-4, "empty volume should be transparent");
    }

    /// A sparse volume: a small solid ball at the origin inside a 16³ grid
    /// spanning [-1, 1], so most of the volume (and most coarse occupancy
    /// cells, even after the one-voxel dilation) is genuinely empty.
    fn sparse_ball_volume() -> VolumeGrid {
        VolumeGrid::from_fn(16, 16, 16, [-1.0; 3], [0.125; 3], |x, y, z| {
            if x * x + y * y + z * z < 0.25 * 0.25 {
                1.0
            } else {
                0.0
            }
        })
    }

    #[test]
    fn test_vr_render_image_with_occupancy_matches_plain() {
        // End-to-end check that the occupancy-aware image render is wired
        // correctly and agrees with the plain renderer.
        //
        // The volume is deliberately sparse: with a volume that fills its
        // grid, the dilated occupancy grid can end up entirely occupied, no
        // skip ever fires, and this test would pass while exercising
        // nothing. The `empty_cells` assertion below pins that down, and
        // `test_vr_march_ray_with_occupancy_is_exact_on_sparse_volume`
        // checks a ray that provably skips *and* still matches.
        let volume = sparse_ball_volume();
        let occupancy = vr_build_occupancy_grid(&volume, 0.1, 2);
        let empty_cells = occupancy.data.iter().filter(|&&v| v < 0.5).count();
        assert!(
            empty_cells > occupancy.data.len() / 2,
            "test setup must leave most coarse cells empty, got {empty_cells} of {}",
            occupancy.data.len()
        );

        let tf = TransferFunction::grayscale(1.0);
        let camera = VolumetricCamera::default_front(6, 6);

        for integration in [
            VolumetricIntegration::FrontToBack,
            VolumetricIntegration::BackToFront,
            VolumetricIntegration::Mip,
            VolumetricIntegration::Avg,
        ] {
            let cfg = cfg_with(integration, false);
            let plain = vr_render_image(&volume, &tf, &camera, &cfg).expect("plain render");
            let with_occ = vr_render_image_with_occupancy(&volume, &tf, &camera, &cfg, &occupancy)
                .expect("occupancy render");

            assert_eq!(plain.len(), with_occ.len());
            for (a, b) in plain.iter().zip(with_occ.iter()) {
                for c in 0..4 {
                    assert!(
                        (a[c] - b[c]).abs() < 1e-3,
                        "{integration:?} pixel mismatch: {a:?} vs {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_vr_march_ray_with_occupancy_is_exact_on_sparse_volume() {
        // The core contract: empty-space skipping is an *optimization*, so on
        // a ray that actually skips (strictly fewer samples) the result must
        // still match the plain march. Both halves are asserted together --
        // either alone can be satisfied by a broken implementation (skip
        // nothing, or skip everything and return garbage).
        //
        // The two failure modes this pins down are exactly what the fixed
        // coarse-voxel stride produced: it advanced a whole coarse voxel from
        // wherever the sample landed (jumping over occupied cells), and the
        // post-skip samples no longer sat on the plain march's lattice, so
        // even the density it did reach was integrated at different
        // positions. The image-level version of this comparison came back
        // more than an order of magnitude too transparent.
        let volume = sparse_ball_volume();
        let occupancy = vr_build_occupancy_grid(&volume, 0.1, 2);
        let tf = TransferFunction::grayscale(1.0);
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);

        for integration in [
            VolumetricIntegration::FrontToBack,
            VolumetricIntegration::BackToFront,
            VolumetricIntegration::Mip,
            VolumetricIntegration::Avg,
        ] {
            let cfg = cfg_with(integration, false);
            let plain = vr_march_ray(&ray, &volume, &tf, &cfg);
            let with_occ = vr_march_ray_with_occupancy(&ray, &volume, &tf, &cfg, &occupancy);

            assert!(
                with_occ.n_steps < plain.n_steps,
                "{integration:?}: skipping must save samples, plain={} with_occ={}",
                plain.n_steps,
                with_occ.n_steps
            );
            assert!(
                plain.alpha > 0.1,
                "{integration:?}: the probe ray must actually hit the ball, alpha={}",
                plain.alpha
            );
            assert!(
                (plain.alpha - with_occ.alpha).abs() < 1e-5,
                "{integration:?}: alpha {} vs {}",
                plain.alpha,
                with_occ.alpha
            );
            for c in 0..3 {
                assert!(
                    (plain.color[c] - with_occ.color[c]).abs() < 1e-5,
                    "{integration:?}: colour[{c}] {} vs {}",
                    plain.color[c],
                    with_occ.color[c]
                );
            }
        }
    }

    #[test]
    fn test_occupancy_grid_dilates_by_one_voxel_for_the_trilinear_stencil() {
        // A single occupied fine voxel must mark its own coarse cell *and*
        // every coarse cell whose trilinear stencil can reach it -- i.e. the
        // neighbouring cells it is adjacent to. Voxel (2,0,0) of a 4³ grid
        // with factor 2 sits at the low edge of coarse cell (1,0,0); a
        // trilinear sample taken inside cell (0,0,0) can still read it, so
        // that cell must be marked occupied too.
        let mut volume = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        volume.data[2] = 1.0; // (ix, iy, iz) = (2, 0, 0)
        let occ = vr_build_occupancy_grid(&volume, 0.5, 2);
        assert_eq!((occ.nx, occ.ny, occ.nz), (2, 2, 2));

        let cell = |ox: usize, oy: usize, oz: usize| occ.data[oz * 4 + oy * 2 + ox];
        assert!(cell(1, 0, 0) > 0.5, "the cell containing the voxel");
        assert!(
            cell(0, 0, 0) > 0.5,
            "the adjacent cell whose trilinear stencil reaches the voxel"
        );
        // The dilation is exactly one voxel, not more: cells whose halo does
        // not reach voxel (2,0,0) stay empty. Cell (0,1,0) covers y = 2..3
        // with halo y = 1..3, which excludes y = 0.
        assert!(
            cell(0, 1, 0) < 0.5,
            "cell (0,1,0)'s halo spans y = 1..3 and must not reach y = 0"
        );
        assert!(
            cell(1, 1, 1) < 0.5,
            "cell (1,1,1)'s halo spans y = z = 1..3 and must not reach y = z = 0"
        );
    }

    #[test]
    fn test_occupancy_skip_stops_at_the_empty_cell_boundary() {
        // Guards the "skip only to the current empty cell's exit" rule
        // directly: a ray crosses a thin occupied slab, and any skip that
        // overshoots the empty cell it started in would step straight past
        // the slab. (Unlike
        // `test_vr_march_ray_with_occupancy_is_exact_on_sparse_volume`, this
        // geometry is not one the old fixed stride happened to miss -- it
        // pins the invariant, not the historical failure.)
        let volume = VolumeGrid::from_fn(16, 16, 16, [-1.0; 3], [0.125; 3], |_, _, z| {
            if (0.0..0.25).contains(&z) {
                1.0
            } else {
                0.0
            }
        });
        let occupancy = vr_build_occupancy_grid(&volume, 0.1, 2);
        let tf = TransferFunction::grayscale(1.0);
        let cfg = cfg_with(VolumetricIntegration::BackToFront, false);
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);

        let plain = vr_march_ray(&ray, &volume, &tf, &cfg);
        let with_occ = vr_march_ray_with_occupancy(&ray, &volume, &tf, &cfg, &occupancy);

        assert!(
            plain.alpha > 0.1,
            "the slab must be visible: {}",
            plain.alpha
        );
        assert!(
            with_occ.n_steps < plain.n_steps,
            "skipping should still save samples outside the slab"
        );
        assert!(
            (plain.alpha - with_occ.alpha).abs() < 1e-5,
            "the slab must not be skipped over: {} vs {}",
            plain.alpha,
            with_occ.alpha
        );
    }

    #[test]
    fn test_occupancy_skip_preserves_avg_divisor() {
        // `Avg` divides by the number of samples, so a skipped sample is not
        // free the way it is for the alpha-compositing modes: dropping it
        // would raise the mean. The skipped lattice points must still be
        // counted (and their density-0 contribution added), which is what
        // keeps the occupancy-aware average equal to the plain one.
        let volume = sparse_ball_volume();
        let occupancy = vr_build_occupancy_grid(&volume, 0.1, 2);
        let tf = TransferFunction::grayscale(1.0);
        let cfg = cfg_with(VolumetricIntegration::Avg, false);
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);

        let plain = vr_march_ray(&ray, &volume, &tf, &cfg);
        let with_occ = vr_march_ray_with_occupancy(&ray, &volume, &tf, &cfg, &occupancy);

        assert!(plain.alpha > 0.0, "the probe ray must hit the ball");
        assert!(
            (plain.alpha - with_occ.alpha).abs() < 1e-5,
            "average alpha must not depend on how many samples were skipped: \
             {} vs {}",
            plain.alpha,
            with_occ.alpha
        );
    }

    #[test]
    fn test_occupancy_skip_folds_in_non_transparent_empty_space() {
        // A transfer function is free to give density 0 a non-zero opacity
        // (a global fog). Skipped samples then *do* contribute, and the
        // marcher folds their contribution in analytically instead of
        // dropping it. Without that fold the occupancy march would come out
        // markedly more transparent than the plain one.
        let volume = sparse_ball_volume();
        let occupancy = vr_build_occupancy_grid(&volume, 0.1, 2);
        let fog = TransferFunction {
            points: vec![
                TransferPoint {
                    density: 0.0,
                    color: [0.2, 0.2, 0.6],
                    opacity: 0.25,
                },
                TransferPoint {
                    density: 1.0,
                    color: [1.0, 1.0, 1.0],
                    opacity: 1.0,
                },
            ],
        };
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);

        for integration in [
            VolumetricIntegration::FrontToBack,
            VolumetricIntegration::BackToFront,
        ] {
            let cfg = cfg_with(integration, false);
            let plain = vr_march_ray(&ray, &volume, &fog, &cfg);
            let with_occ = vr_march_ray_with_occupancy(&ray, &volume, &fog, &cfg, &occupancy);

            assert!(
                (plain.alpha - with_occ.alpha).abs() < 1e-3,
                "{integration:?}: fog contribution of skipped samples must be \
                 folded in: {} vs {}",
                plain.alpha,
                with_occ.alpha
            );
            for c in 0..3 {
                assert!(
                    (plain.color[c] - with_occ.color[c]).abs() < 1e-3,
                    "{integration:?}: colour[{c}] {} vs {}",
                    plain.color[c],
                    with_occ.color[c]
                );
            }
        }
    }

    // ── Defensive s_alpha clamp (opacity > 1 must not produce NaN) ────────────

    #[test]
    fn test_march_front_to_back_clamps_out_of_range_opacity() {
        // Regression test: `(1.0 - s_alpha).powf(step_size)` produces NaN
        // for any s_alpha > 1 (a negative base raised to a fractional
        // power). `TransferFunction::new` now rejects out-of-range
        // opacities, but `points` is a public field, so this constructs one
        // directly to exercise the defensive clamp in the marcher itself.
        let volume = VolumeGrid::from_fn(4, 4, 4, [-1.0; 3], [0.5; 3], |_, _, _| 0.5);
        let tf = TransferFunction {
            points: vec![
                TransferPoint {
                    density: 0.0,
                    color: [1.0, 0.0, 0.0],
                    opacity: 0.0,
                },
                TransferPoint {
                    density: 1.0,
                    color: [1.0, 0.0, 0.0],
                    opacity: 2.5, // out of [0, 1], bypassing `new`
                },
            ],
        };
        let cfg = cfg_with(VolumetricIntegration::FrontToBack, false);
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);

        let result = vr_march_ray(&ray, &volume, &tf, &cfg);
        assert!(result.alpha.is_finite(), "alpha = {}", result.alpha);
        for c in result.color {
            assert!(c.is_finite(), "color component = {c}");
        }
    }

    #[test]
    fn test_march_back_to_front_clamps_out_of_range_opacity() {
        let volume = VolumeGrid::from_fn(4, 4, 4, [-1.0; 3], [0.5; 3], |_, _, _| 0.5);
        let tf = TransferFunction {
            points: vec![
                TransferPoint {
                    density: 0.0,
                    color: [0.0, 1.0, 0.0],
                    opacity: 0.0,
                },
                TransferPoint {
                    density: 1.0,
                    color: [0.0, 1.0, 0.0],
                    opacity: 3.0, // out of [0, 1], bypassing `new`
                },
            ],
        };
        let cfg = cfg_with(VolumetricIntegration::BackToFront, false);
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);

        let result = vr_march_ray(&ray, &volume, &tf, &cfg);
        assert!(result.alpha.is_finite(), "alpha = {}", result.alpha);
        for c in result.color {
            assert!(c.is_finite(), "color component = {c}");
        }
    }
}
