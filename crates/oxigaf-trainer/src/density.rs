//! Adaptive density control: split, clone, prune, and opacity reset.
//!
//! Follows the strategy from 3D Gaussian Splatting:
//!
//! 1. **Accumulate** the norm of the position gradient for each Gaussian over
//!    several iterations.
//! 2. **Split** Gaussians with high gradient *and* large scale → two smaller
//!    Gaussians displaced along the principal axis.
//! 3. **Clone** Gaussians with high gradient *and* small scale → duplicate.
//! 4. **Prune** Gaussians with low opacity or excessively large screen extent.
//! 5. Periodically **reset** all opacities to a low value.

use rand::{Rng, RngExt};

use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

use crate::config::DensityConfig;
use crate::optimizer::Gradients;

/// World-space scale (post-`exp()`) above which a Gaussian is pruned outright
/// as pathologically large, expressed as a multiple of
/// `DensityConfig::split_scale_threshold`. There is no camera available in
/// [`DensityController::densify_and_prune`] to compute a true (pixel-space)
/// screen extent, so this is a world-space proxy for the module doc's
/// "excessively large screen extent" prune criterion.
const MAX_SCALE_PRUNE_MULTIPLIER: f32 = 100.0;

// ---------------------------------------------------------------------------
// DensifyResult
// ---------------------------------------------------------------------------

/// Describes what changed during a densify-and-prune pass so that the
/// optimiser can adjust its bookkeeping.
#[derive(Debug, Clone)]
pub struct DensifyResult {
    /// Boolean mask over the **original** model: `true` = kept, `false` = removed.
    pub keep_mask: Vec<bool>,
    /// Number of *new* Gaussians appended after compaction.
    pub num_added: usize,
}

// ---------------------------------------------------------------------------
// DensityController
// ---------------------------------------------------------------------------

/// Manages gradient accumulation and adaptive density control.
#[derive(Debug, Clone)]
pub struct DensityController {
    config: DensityConfig,
    /// Accumulated position-gradient norms per Gaussian.
    grad_accum: Vec<f32>,
    /// Number of accumulation steps per Gaussian.
    grad_count: Vec<u32>,
}

impl DensityController {
    /// Create a controller for a model of size `n`.
    pub fn new(config: DensityConfig, n: usize) -> Self {
        Self {
            config,
            grad_accum: vec![0.0; n],
            grad_count: vec![0; n],
        }
    }

    // ----- gradient accumulation -------------------------------------------

    /// Add the current step's position-gradient norms to the accumulator.
    pub fn accumulate_gradients(&mut self, gradients: &Gradients) {
        let n = gradients.num_gaussians().min(self.grad_accum.len());
        for i in 0..n {
            let gx = gradients.position[i * 3];
            let gy = gradients.position[i * 3 + 1];
            let gz = gradients.position[i * 3 + 2];
            let norm = (gx * gx + gy * gy + gz * gz).sqrt();
            self.grad_accum[i] += norm;
            self.grad_count[i] += 1;
        }
    }

    // ----- densify & prune -------------------------------------------------

    /// Run the full adaptive density-control pass.
    ///
    /// Modifies `model` in place (adds / removes Gaussians) and returns a
    /// [`DensifyResult`] that the optimiser should use to update its state.
    pub fn densify_and_prune(
        &mut self,
        model: &mut GaussianModel,
        rng: &mut impl Rng,
    ) -> DensifyResult {
        let n = model.len();
        let avg_grads = self.average_gradients();

        let mut to_split: Vec<usize> = Vec::new();
        let mut to_clone: Vec<usize> = Vec::new();
        let mut to_prune: Vec<usize> = Vec::new();

        // World-space scale above which a Gaussian is pruned outright as
        // pathologically large (see `MAX_SCALE_PRUNE_MULTIPLIER`), implementing
        // the module doc's "excessively large ... extent" prune criterion.
        let max_scale_prune_threshold =
            self.config.split_scale_threshold * MAX_SCALE_PRUNE_MULTIPLIER;

        for i in 0..n {
            let opacity = sigmoid(model.gaussians[i].opacity);
            let max_scale = model.gaussians[i]
                .scale
                .iter()
                .map(|s| s.exp())
                .fold(0.0_f32, f32::max);

            // Prune: low opacity, or pathologically large world-space scale.
            if opacity < self.config.min_opacity || max_scale > max_scale_prune_threshold {
                to_prune.push(i);
                continue;
            }

            // Densify: high average gradient.
            if i < avg_grads.len() && avg_grads[i] > self.config.grad_threshold {
                if max_scale > self.config.split_scale_threshold {
                    to_split.push(i);
                } else {
                    to_clone.push(i);
                }
            }
        }

        // --- Enforce max_gaussians BEFORE creating any new Gaussians -------
        //
        // `DensifyResult::keep_mask` describes the ORIGINAL model and
        // `num_added` must equal exactly how many Gaussians are appended --
        // both are consumed by the optimizer (see `handle_densify`) to
        // resize its per-parameter Adam state in lock-step with the model
        // buffers. Capping the total by discarding already-appended
        // Gaussians after the fact would desync `num_added` from what was
        // actually appended, so the cap is enforced here instead: only as
        // many split/clone candidates are admitted as fit the remaining
        // budget, highest average-gradient candidates first.
        let n_kept_untouched = n - to_prune.len() - to_split.len();
        let budget = self.config.max_gaussians.saturating_sub(n_kept_untouched);
        if to_split.len() * 2 + to_clone.len() > budget {
            let mut candidates: Vec<(usize, bool, f32)> = to_split
                .iter()
                .map(|&i| (i, true, avg_grads.get(i).copied().unwrap_or(0.0)))
                .chain(
                    to_clone
                        .iter()
                        .map(|&i| (i, false, avg_grads.get(i).copied().unwrap_or(0.0))),
                )
                .collect();
            candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

            let mut admitted_split = Vec::new();
            let mut admitted_clone = Vec::new();
            let mut remaining = budget;
            for (idx, is_split, _grad) in candidates {
                let cost = if is_split { 2 } else { 1 };
                if remaining >= cost {
                    remaining -= cost;
                    if is_split {
                        admitted_split.push(idx);
                    } else {
                        admitted_clone.push(idx);
                    }
                }
            }
            let n_requested = to_split.len() + to_clone.len();
            to_split = admitted_split;
            to_clone = admitted_clone;

            tracing::warn!(
                "Density control: max_gaussians={} cap reached; admitted {} of {} \
                 requested split/clone candidates this pass (prioritised by gradient magnitude).",
                self.config.max_gaussians,
                to_split.len() + to_clone.len(),
                n_requested,
            );
        }

        // --- Create new Gaussians ---
        let sh_per = sh_channels(model.sh_degree);

        let mut new_gaussians: Vec<GaussianAttributes> = Vec::new();
        let mut new_sh: Vec<f32> = Vec::new();
        let mut new_faces: Vec<u32> = Vec::new();
        let mut new_bary: Vec<[f32; 3]> = Vec::new();
        let mut new_offsets: Vec<[f32; 3]> = Vec::new();
        let mut new_rigid: Vec<bool> = Vec::new();

        let scale_reduction = 1.6_f32.ln();

        // Splits → 2 children each.
        for &i in &to_split {
            let g = model.gaussians[i];
            for _ in 0..2 {
                // Sample the displacement in the Gaussian's LOCAL frame (its
                // principal axes), then rotate it into world space -- an
                // anisotropic Gaussian rotated away from the world axes must
                // have its children displaced along its own (rotated)
                // principal axes, not the raw world axes, or they no longer
                // tile the parent's ellipsoid.
                let local_offset = [
                    g.scale[0].exp() * random_normal(rng),
                    g.scale[1].exp() * random_normal(rng),
                    g.scale[2].exp() * random_normal(rng),
                ];
                let offset = rotate_by_quaternion(g.rotation, local_offset);
                let child = GaussianAttributes {
                    position: [
                        g.position[0] + offset[0],
                        g.position[1] + offset[1],
                        g.position[2] + offset[2],
                    ],
                    _pad0: 0.0,
                    rotation: g.rotation,
                    scale: [
                        g.scale[0] - scale_reduction,
                        g.scale[1] - scale_reduction,
                        g.scale[2] - scale_reduction,
                    ],
                    opacity: g.opacity,
                };
                new_gaussians.push(child);
                new_sh.extend_from_slice(&model.sh_coeffs[i * sh_per..(i + 1) * sh_per]);
                new_faces.push(model.face_indices[i]);
                new_bary.push(model.barycentric[i]);
                new_offsets.push(model.local_offsets[i]);
                new_rigid.push(model.is_rigid[i]);
            }
        }

        // Clones → 1 copy each.
        for &i in &to_clone {
            new_gaussians.push(model.gaussians[i]);
            new_sh.extend_from_slice(&model.sh_coeffs[i * sh_per..(i + 1) * sh_per]);
            new_faces.push(model.face_indices[i]);
            new_bary.push(model.barycentric[i]);
            new_offsets.push(model.local_offsets[i]);
            new_rigid.push(model.is_rigid[i]);
        }

        let num_added = new_gaussians.len();

        // --- Build keep-mask and compact ------------------------------------
        let mut keep_mask = vec![true; n];
        for &i in &to_split {
            keep_mask[i] = false; // originals replaced by children
        }
        for &i in &to_prune {
            keep_mask[i] = false;
        }

        compact_model(model, &keep_mask);

        // Append new Gaussians.
        model.gaussians.extend(new_gaussians);
        model.sh_coeffs.extend(new_sh);
        model.face_indices.extend(new_faces);
        model.barycentric.extend(new_bary);
        model.local_offsets.extend(new_offsets);
        model.is_rigid.extend(new_rigid);

        // The split/clone admission budget above guarantees densification
        // alone cannot push the model past `max_gaussians`; this can only
        // still fire if pruning (low opacity / pathological world-space
        // scale) was not aggressive enough to bring an already-oversized
        // model back under the cap on its own.
        if model.len() > self.config.max_gaussians {
            tracing::warn!(
                "Model has {} Gaussians, still above cap {} even after admitting no new \
                 split/clone candidates this pass. Pruning (min_opacity / world-space \
                 max-scale) alone is not shrinking the model enough to fit under the cap.",
                model.len(),
                self.config.max_gaussians,
            );
        }

        // Reset accumulator for the new model size.
        self.reset_accumulator(model.len());

        tracing::info!(
            "Density control: split={}, clone={}, prune={}, total={}",
            to_split.len(),
            to_clone.len(),
            to_prune.len(),
            model.len(),
        );

        DensifyResult {
            keep_mask,
            num_added,
        }
    }

    // ----- opacity reset ---------------------------------------------------

    /// Set every Gaussian's inverse-sigmoid opacity to `value` (typically a
    /// low value like −2, corresponding to σ ≈ 0.12).
    pub fn reset_opacity(model: &mut GaussianModel, value: f32) {
        for g in &mut model.gaussians {
            g.opacity = value;
        }
        tracing::info!(
            "Reset all opacities to inv_sigmoid = {value} (σ = {:.4})",
            sigmoid(value),
        );
    }

    // ----- internal helpers ------------------------------------------------

    fn average_gradients(&self) -> Vec<f32> {
        self.grad_accum
            .iter()
            .zip(self.grad_count.iter())
            .map(|(&acc, &cnt)| if cnt > 0 { acc / cnt as f32 } else { 0.0 })
            .collect()
    }

    fn reset_accumulator(&mut self, n: usize) {
        self.grad_accum = vec![0.0; n];
        self.grad_count = vec![0; n];
    }
}

// ===========================================================================
// Free helpers
// ===========================================================================

/// Compact all vectors in a [`GaussianModel`] according to a boolean mask.
fn compact_model(model: &mut GaussianModel, keep: &[bool]) {
    let sh_per = sh_channels(model.sh_degree);

    let mut g = Vec::new();
    let mut sh = Vec::new();
    let mut fi = Vec::new();
    let mut ba = Vec::new();
    let mut lo = Vec::new();
    let mut ri = Vec::new();

    for (i, &k) in keep.iter().enumerate() {
        if k {
            g.push(model.gaussians[i]);
            sh.extend_from_slice(&model.sh_coeffs[i * sh_per..(i + 1) * sh_per]);
            fi.push(model.face_indices[i]);
            ba.push(model.barycentric[i]);
            lo.push(model.local_offsets[i]);
            ri.push(model.is_rigid[i]);
        }
    }

    model.gaussians = g;
    model.sh_coeffs = sh;
    model.face_indices = fi;
    model.barycentric = ba;
    model.local_offsets = lo;
    model.is_rigid = ri;
}

/// Number of SH coefficients per Gaussian for a given SH degree.
#[inline]
fn sh_channels(degree: u32) -> usize {
    ((degree + 1) * (degree + 1) * 3) as usize
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Box–Muller transform: two uniform samples → one standard-normal sample.
fn random_normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-10);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Rotate vector `v` by unit quaternion `q = (x, y, z, w)` (Hamilton
/// convention, scalar-last "xyzw" layout matching
/// [`GaussianAttributes::rotation`]).
///
/// Uses the standard optimized quaternion-vector rotation formula (avoids a
/// full quaternion multiply):
///
/// ```text
/// t  = 2 * (q.xyz × v)
/// v' = v + q.w * t + (q.xyz × t)
/// ```
///
/// If `q` is not (numerically close to) a unit quaternion -- in particular
/// the zero quaternion, which has no defined rotation -- `v` is returned
/// unchanged rather than propagating NaNs from a zero-length normalization.
fn rotate_by_quaternion(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if norm_sq.is_nan() || norm_sq <= 1e-12 {
        return v;
    }
    // Normalise defensively -- callers should already pass unit quaternions,
    // but a non-unit input would otherwise silently scale `v`.
    let inv_norm = norm_sq.sqrt().recip();
    let qv = [q[0] * inv_norm, q[1] * inv_norm, q[2] * inv_norm];
    let w = q[3] * inv_norm;

    let cross = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };

    let c1 = cross(qv, v);
    let t = [2.0 * c1[0], 2.0 * c1[1], 2.0 * c1[2]];
    let c2 = cross(qv, t);
    [
        v[0] + w * t[0] + c2[0],
        v[1] + w * t[1] + c2[1],
        v[2] + w * t[2] + c2[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf_render::gaussian::GaussianAttributes;
    use rand::SeedableRng;

    fn make_model(n: usize) -> GaussianModel {
        let sh_degree = 0_u32;
        let sh_per = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        GaussianModel {
            gaussians: vec![
                GaussianAttributes {
                    position: [0.0; 3],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-5.0; 3],
                    opacity: 0.0,
                };
                n
            ],
            sh_coeffs: vec![0.0; n * sh_per],
            sh_degree,
            face_indices: vec![0; n],
            barycentric: vec![[1.0, 0.0, 0.0]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![true; n],
        }
    }

    #[test]
    fn prune_removes_low_opacity() {
        let mut model = make_model(5);
        // Set two Gaussians to very low opacity (sigmoid ≈ 0).
        model.gaussians[1].opacity = -10.0;
        model.gaussians[3].opacity = -10.0;

        let cfg = DensityConfig {
            min_opacity: 0.005,
            grad_threshold: 999.0, // no densification
            ..DensityConfig::default()
        };
        let mut ctrl = DensityController::new(cfg, 5);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let result = ctrl.densify_and_prune(&mut model, &mut rng);

        assert_eq!(model.len(), 3);
        assert_eq!(result.keep_mask, vec![true, false, true, false, true]);
        assert_eq!(result.num_added, 0);
    }

    #[test]
    fn densify_enforces_max_gaussians_cap_before_creating_children() {
        let n = 10;
        let mut model = make_model(n);
        // High opacity so nothing is pruned by the opacity check.
        for g in &mut model.gaussians {
            g.opacity = 10.0; // sigmoid(10) ≈ 1.0
        }

        let cfg = DensityConfig {
            min_opacity: 0.005,
            grad_threshold: 0.0001, // low threshold: every Gaussian densifies
            split_scale_threshold: 0.01,
            max_gaussians: 12, // 10 kept + only 2 new clones fit
            ..DensityConfig::default()
        };
        let mut ctrl = DensityController::new(cfg, n);

        // Give every Gaussian a nonzero, above-threshold position gradient so
        // all 10 are densify-eligible. Scale stays at the default -5.0
        // (exp(-5.0) ≈ 0.0067 < split_scale_threshold=0.01), so every
        // candidate is a *clone* (cost 1 new slot each), making the expected
        // admitted count exact and easy to check.
        let mut grads = Gradients::zeros(n, 3);
        for i in 0..n {
            grads.position[i * 3] = 1.0;
        }
        ctrl.accumulate_gradients(&grads);

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let result = ctrl.densify_and_prune(&mut model, &mut rng);

        assert!(
            model.len() <= 12,
            "max_gaussians cap must be enforced, got {}",
            model.len()
        );
        assert_eq!(
            model.len(),
            12,
            "exactly 2 clone candidates should fit the budget"
        );
        assert_eq!(result.num_added, 2);
        // keep_mask covers the ORIGINAL model, so it must stay length `n`.
        assert_eq!(result.keep_mask.len(), n);
        assert!(
            result.keep_mask.iter().all(|&k| k),
            "no Gaussian was pruned or split"
        );
    }

    #[test]
    fn densify_prunes_pathologically_large_scale_gaussians() {
        let mut model = make_model(4);
        // Give one Gaussian a pathologically large world-space scale:
        // exp(6.0) ≈ 403, far beyond 100 * split_scale_threshold(0.01) = 1.0.
        model.gaussians[2].scale = [6.0, 6.0, 6.0];
        // Keep opacity healthy on all four so only the scale check can prune.
        for g in &mut model.gaussians {
            g.opacity = 10.0;
        }

        let cfg = DensityConfig {
            min_opacity: 0.005,
            grad_threshold: 999.0, // no densification
            split_scale_threshold: 0.01,
            ..DensityConfig::default()
        };
        let mut ctrl = DensityController::new(cfg, 4);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let result = ctrl.densify_and_prune(&mut model, &mut rng);

        assert_eq!(model.len(), 3, "the oversized Gaussian must be pruned");
        assert_eq!(
            result.keep_mask,
            vec![true, true, false, true],
            "only index 2 (oversized scale) should be dropped"
        );
    }

    #[test]
    fn split_offset_is_rotated_into_world_frame() {
        let mut model = make_model(1);
        // Extreme anisotropy: local-X is "fat" (scale=1.0), local-Y/Z are
        // needle-thin (scale=1e-4), so the split displacement is completely
        // dominated by the local-X component regardless of the exact random
        // draw.
        model.gaussians[0].scale = [0.0, (1e-4f32).ln(), (1e-4f32).ln()];
        // 90 degree rotation around world Z: local +X maps to world +Y.
        let half = std::f32::consts::FRAC_PI_4;
        model.gaussians[0].rotation = [0.0, 0.0, half.sin(), half.cos()];
        model.gaussians[0].opacity = 10.0; // avoid the low-opacity prune path

        let cfg = DensityConfig {
            min_opacity: 0.005,
            grad_threshold: 0.0001,
            split_scale_threshold: 0.5, // exp(0.0)=1.0 > 0.5 -> split, not clone
            ..DensityConfig::default()
        };
        let mut ctrl = DensityController::new(cfg, 1);
        let mut grads = Gradients::zeros(1, 3);
        grads.position[0] = 1.0; // nonzero gradient -> densify-eligible
        ctrl.accumulate_gradients(&grads);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = ctrl.densify_and_prune(&mut model, &mut rng);

        assert_eq!(result.num_added, 2, "a split produces exactly 2 children");
        assert_eq!(model.len(), 2);
        let p0 = model.gaussians[0].position;
        let p1 = model.gaussians[1].position;
        let dx = (p0[0] - p1[0]).abs();
        let dy = (p0[1] - p1[1]).abs();
        let dz = (p0[2] - p1[2]).abs();
        assert!(
            dy > dx * 10.0 && dy > dz * 10.0,
            "children should separate along the rotated local-X axis \
             (world Y), got dx={dx} dy={dy} dz={dz}"
        );
        // Sanity: they must actually be separated at all (not a degenerate
        // same-point split).
        assert!(dy > 1e-6, "children should not be coincident, dy={dy}");
    }

    #[test]
    fn rotate_by_quaternion_identity_is_noop() {
        let identity = [0.0, 0.0, 0.0, 1.0];
        let v = [1.0, 2.0, 3.0];
        let out = rotate_by_quaternion(identity, v);
        for (a, b) in v.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "identity rotation must be a no-op");
        }
    }

    #[test]
    fn rotate_by_quaternion_90deg_z_maps_x_to_y() {
        let half = std::f32::consts::FRAC_PI_4;
        let q = [0.0, 0.0, half.sin(), half.cos()]; // 90 deg around Z
        let out = rotate_by_quaternion(q, [1.0, 0.0, 0.0]);
        assert!((out[0] - 0.0).abs() < 1e-5, "x={}", out[0]);
        assert!((out[1] - 1.0).abs() < 1e-5, "y={}", out[1]);
        assert!((out[2] - 0.0).abs() < 1e-5, "z={}", out[2]);
    }

    #[test]
    fn rotate_by_quaternion_zero_quaternion_returns_input_unchanged() {
        let zero = [0.0, 0.0, 0.0, 0.0];
        let v = [1.0, -2.0, 3.5];
        let out = rotate_by_quaternion(zero, v);
        assert_eq!(out, v, "zero quaternion must not produce NaNs");
    }

    // Regression: the early-return guard used to be `if !(norm_sq > 1e-12)`,
    // which returns `v` unchanged both when `norm_sq` is small/zero *and*
    // when it is NaN (NaN comparisons are always `false`, so `!(NaN > x)` is
    // `true`). Rewritten as `norm_sq.is_nan() || norm_sq <= 1e-12` to satisfy
    // clippy's `neg_cmp_op_on_partial_ord`; a NaN-quaternion input must still
    // take the early-return path rather than falling through into
    // `norm_sq.sqrt().recip()` and propagating NaN into the output.
    #[test]
    fn rotate_by_quaternion_nan_quaternion_returns_input_unchanged() {
        let nan_q = [f32::NAN, 0.0, 0.0, 1.0];
        let v = [1.0, -2.0, 3.5];
        let out = rotate_by_quaternion(nan_q, v);
        assert_eq!(out, v, "NaN quaternion must not produce/propagate NaNs");
    }
}
