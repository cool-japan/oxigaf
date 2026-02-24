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

        for i in 0..n {
            let opacity = sigmoid(model.gaussians[i].opacity);
            let max_scale = model.gaussians[i]
                .scale
                .iter()
                .map(|s| s.exp())
                .fold(0.0_f32, f32::max);

            // Prune: low opacity.
            if opacity < self.config.min_opacity {
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
                let offset = [
                    g.scale[0].exp() * random_normal(rng),
                    g.scale[1].exp() * random_normal(rng),
                    g.scale[2].exp() * random_normal(rng),
                ];
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

        // Enforce hard cap (soft warning only — a proper implementation would
        // prune by opacity ranking).
        if model.len() > self.config.max_gaussians {
            tracing::warn!(
                "Model has {} Gaussians (cap {}). Consider additional pruning.",
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
}
