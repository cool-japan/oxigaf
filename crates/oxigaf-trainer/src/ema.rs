//! Exponential Moving Average (EMA) shadow copy of [`GaussianModel`] parameters.
//!
//! EMA smooths model parameters over training, reducing high-frequency noise
//! and providing a more stable snapshot for inference.  A separate "shadow"
//! copy of every learnable parameter is maintained and updated at each step:
//!
//! ```text
//! shadow = effective_decay * shadow_prev + (1 − effective_decay) * current
//! ```
//!
//! Bias correction is applied during the warm-up phase so that the shadow
//! copy does not start at zero:
//!
//! ```text
//! effective_decay = min(decay, (1 + step) / (10 + step))
//! ```
//!
//! # Rotation quaternions are a special case
//!
//! The linear blend above is applied component-wise to positions, scales,
//! opacities, and SH coefficients, but **not** to rotation quaternions as-is.
//! Because `q` and `-q` represent the same rotation (the unit-quaternion
//! double cover), a naive component-wise blend can average two quaternions
//! that sit on opposite sides of that cover, interpolating *through the
//! origin* and collapsing the shadow's norm instead of nlerp-ing between two
//! nearby rotations. Rotations therefore use a sign-aligned normalized
//! lerp (nlerp) instead: the incoming quaternion's sign is flipped to match
//! the shadow's hemisphere (`dot(shadow, current) < 0`) *before* the same
//! linear blend, and the blended result is renormalized back to unit length
//! afterward so [`GaussianEma::apply_to`] always writes a valid rotation.
//!
//! # Usage
//!
//! ```rust,ignore
//! use oxigaf_trainer::ema::GaussianEma;
//!
//! let mut ema = GaussianEma::new(&model, 0.999);
//! // During training:
//! ema.update(&model);
//! // For evaluation/inference:
//! let mut eval_model = model.clone();
//! ema.apply_to(&mut eval_model);
//! ```

use oxigaf_render::gaussian::GaussianModel;

// ---------------------------------------------------------------------------
// GaussianEma
// ---------------------------------------------------------------------------

/// Exponential Moving Average shadow copy of [`GaussianModel`] parameters.
///
/// See the [module-level documentation](self) for the update rule and bias
/// correction formula.
#[derive(Debug, Clone)]
pub struct GaussianEma {
    /// Nominal EMA decay factor (ρ).  Effective decay is additionally bounded
    /// by the bias-correction schedule during early training.
    decay: f32,

    /// Shadow positions — layout `N × 3`.
    positions: Vec<f32>,
    /// Shadow rotation quaternions — layout `N × 4`.
    rotations: Vec<f32>,
    /// Shadow log-scales — layout `N × 3`.
    scales: Vec<f32>,
    /// Shadow inverse-sigmoid opacities — layout `N × 1`.
    opacities: Vec<f32>,
    /// Shadow spherical-harmonics coefficients — layout `N × C`.
    sh_coeffs: Vec<f32>,

    /// Number of EMA update steps applied so far.
    step: u64,
}

impl GaussianEma {
    /// Create a new EMA tracker initialised to the **current** model parameters.
    ///
    /// # Arguments
    ///
    /// * `model` — The model whose parameters will be tracked.
    /// * `decay` — Nominal EMA decay in (0, 1).  Typical value: 0.999.
    pub fn new(model: &GaussianModel, decay: f32) -> Self {
        let n = model.len();
        let positions = extract_positions(model, n);
        let rotations = extract_rotations(model, n);
        let scales = extract_scales(model, n);
        let opacities = extract_opacities(model, n);
        let sh_coeffs = model.sh_coeffs.clone();

        Self {
            decay,
            positions,
            rotations,
            scales,
            opacities,
            sh_coeffs,
            step: 0,
        }
    }

    // ---- public API --------------------------------------------------------

    /// Apply one EMA update from the **current** model parameters.
    ///
    /// If the model size has changed (after density control), the shadow
    /// buffers are silently **re-initialised** to the current model to keep
    /// sizes in sync.  The step counter continues from where it left off.
    pub fn update(&mut self, model: &GaussianModel) {
        self.step += 1;
        let d = self.effective_decay();
        let one_minus_d = 1.0 - d;

        let n = model.len();

        // If sizes diverge (density control resized the model), re-sync.
        if self.positions.len() != n * 3 {
            self.positions = extract_positions(model, n);
            self.rotations = extract_rotations(model, n);
            self.scales = extract_scales(model, n);
            self.opacities = extract_opacities(model, n);
            self.sh_coeffs = model.sh_coeffs.clone();
            return; // Shadow already set to current; nothing more to do.
        }

        // Update positions.
        for i in 0..n {
            for j in 0..3 {
                let idx = i * 3 + j;
                self.positions[idx] =
                    d * self.positions[idx] + one_minus_d * model.gaussians[i].position[j];
            }
        }

        // Update rotations. Quaternions require special handling: `q` and
        // `-q` represent the same rotation (double cover), so a naive
        // component-wise blend can interpolate *through the origin* when the
        // optimized quaternion crosses the sign boundary between updates,
        // collapsing the shadow quaternion's norm toward zero. Align the
        // incoming quaternion's sign with the shadow before blending, then
        // renormalize the result so `apply_to` always writes a valid unit
        // quaternion.
        for i in 0..n {
            let base = i * 4;
            let shadow = [
                self.rotations[base],
                self.rotations[base + 1],
                self.rotations[base + 2],
                self.rotations[base + 3],
            ];
            let mut current = model.gaussians[i].rotation;

            let dot = shadow[0] * current[0]
                + shadow[1] * current[1]
                + shadow[2] * current[2]
                + shadow[3] * current[3];
            if dot < 0.0 {
                for c in current.iter_mut() {
                    *c = -*c;
                }
            }

            let mut blended = [0.0f32; 4];
            for j in 0..4 {
                blended[j] = d * shadow[j] + one_minus_d * current[j];
            }

            let norm_sq = blended[0] * blended[0]
                + blended[1] * blended[1]
                + blended[2] * blended[2]
                + blended[3] * blended[3];
            if norm_sq > 1e-16 {
                let inv_norm = norm_sq.sqrt().recip();
                for c in blended.iter_mut() {
                    *c *= inv_norm;
                }
            } else {
                // Degenerate blend (should not happen for sign-aligned unit
                // quaternions, but guard defensively) -- fall back to the
                // sign-aligned current rotation rather than a zero vector.
                blended = current;
            }

            self.rotations[base..(4 + base)].copy_from_slice(&blended);
        }

        // Update scales.
        for i in 0..n {
            for j in 0..3 {
                let idx = i * 3 + j;
                self.scales[idx] = d * self.scales[idx] + one_minus_d * model.gaussians[i].scale[j];
            }
        }

        // Update opacities.
        for i in 0..n {
            self.opacities[i] = d * self.opacities[i] + one_minus_d * model.gaussians[i].opacity;
        }

        // Update SH coefficients.
        if self.sh_coeffs.len() == model.sh_coeffs.len() {
            for (shadow, &current) in self.sh_coeffs.iter_mut().zip(model.sh_coeffs.iter()) {
                *shadow = d * *shadow + one_minus_d * current;
            }
        } else {
            // SH channels changed — re-sync.
            self.sh_coeffs = model.sh_coeffs.clone();
        }
    }

    /// Copy the EMA shadow parameters **into** the provided model.
    ///
    /// The model's learnable fields are overwritten with the shadow values.
    /// Call this before inference / evaluation to get the smoothed model.
    ///
    /// If the shadow and model sizes differ, the method does nothing (caller
    /// must ensure the model was not resized after the last `update`).
    pub fn apply_to(&self, model: &mut GaussianModel) {
        let n = model.len();
        if self.positions.len() != n * 3 {
            return;
        }

        for i in 0..n {
            for j in 0..3 {
                model.gaussians[i].position[j] = self.positions[i * 3 + j];
            }
            for j in 0..4 {
                model.gaussians[i].rotation[j] = self.rotations[i * 4 + j];
            }
            for j in 0..3 {
                model.gaussians[i].scale[j] = self.scales[i * 3 + j];
            }
            model.gaussians[i].opacity = self.opacities[i];
        }

        if model.sh_coeffs.len() == self.sh_coeffs.len() {
            model.sh_coeffs.clone_from(&self.sh_coeffs);
        }
    }

    /// Effective EMA decay at the current step.
    ///
    /// Applies bias correction so the shadow copy starts close to the current
    /// parameters rather than zero:
    ///
    /// ```text
    /// effective_decay = min(decay, (1 + step) / (10 + step))
    /// ```
    pub fn effective_decay(&self) -> f32 {
        let bias_correction = (1.0 + self.step as f32) / (10.0 + self.step as f32);
        self.decay.min(bias_correction)
    }

    /// Number of EMA update steps applied so far.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Return the current nominal decay value.
    pub fn decay(&self) -> f32 {
        self.decay
    }

    /// Read-only access to the shadow position buffer (`N × 3`).
    pub fn shadow_positions(&self) -> &[f32] {
        &self.positions
    }

    /// Read-only access to the shadow rotation buffer (`N × 4`).
    pub fn shadow_rotations(&self) -> &[f32] {
        &self.rotations
    }

    /// Read-only access to the shadow scale buffer (`N × 3`).
    pub fn shadow_scales(&self) -> &[f32] {
        &self.scales
    }

    /// Read-only access to the shadow opacity buffer (`N × 1`).
    pub fn shadow_opacities(&self) -> &[f32] {
        &self.opacities
    }

    /// Read-only access to the shadow SH-coefficient buffer.
    pub fn shadow_sh_coeffs(&self) -> &[f32] {
        &self.sh_coeffs
    }
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

fn extract_positions(model: &GaussianModel, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 3);
    for g in &model.gaussians {
        out.extend_from_slice(&g.position);
    }
    out
}

fn extract_rotations(model: &GaussianModel, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 4);
    for g in &model.gaussians {
        out.extend_from_slice(&g.rotation);
    }
    out
}

fn extract_scales(model: &GaussianModel, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 3);
    for g in &model.gaussians {
        out.extend_from_slice(&g.scale);
    }
    out
}

fn extract_opacities(model: &GaussianModel, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for g in &model.gaussians {
        out.push(g.opacity);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

    // ----- test helpers -----------------------------------------------------

    fn make_model(n: usize, fill: f32) -> GaussianModel {
        let attr = GaussianAttributes {
            position: [fill; 3],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, fill],
            scale: [fill; 3],
            opacity: fill,
        };
        let sh_degree = 0_u32;
        let sh_channels = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        GaussianModel {
            gaussians: vec![attr; n],
            sh_coeffs: vec![fill; n * sh_channels],
            sh_degree,
            face_indices: vec![0; n],
            barycentric: vec![[1.0, 0.0, 0.0]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        }
    }

    // ----- bias correction --------------------------------------------------

    #[test]
    fn effective_decay_bias_correction_at_step_zero() {
        let model = make_model(2, 0.5);
        // Decay = 0.999, step = 0 → bias correction = 1/10 = 0.1, min → 0.1.
        let ema = GaussianEma::new(&model, 0.999);
        let d = ema.effective_decay();
        assert!(
            (d - 0.1).abs() < 1e-6,
            "at step=0 effective decay should be 0.1, got {d}"
        );
    }

    #[test]
    fn effective_decay_converges_to_decay() {
        let model = make_model(1, 0.5);
        let mut ema = GaussianEma::new(&model, 0.99);
        // After enough steps the bias correction = (1+k)/(10+k) → 1, so effective
        // decay → min(0.99, 1) = 0.99.
        for _ in 0..1000 {
            ema.update(&model);
        }
        let d = ema.effective_decay();
        assert!(
            (d - 0.99).abs() < 1e-4,
            "after many steps effective decay should ≈ 0.99, got {d}"
        );
    }

    #[test]
    fn step_counter_increments() {
        let model = make_model(2, 1.0);
        let mut ema = GaussianEma::new(&model, 0.9);
        assert_eq!(ema.step(), 0);
        ema.update(&model);
        assert_eq!(ema.step(), 1);
        ema.update(&model);
        assert_eq!(ema.step(), 2);
    }

    // ----- convergence to current value -------------------------------------

    #[test]
    fn ema_approaches_current_value_after_many_updates() {
        // Model A has positions=1.0, model B has positions=0.0.
        // EMA starts from A, then we repeatedly update with B.
        // After many updates the shadow should be close to 0.0.
        let model_b = make_model(1, 0.0);
        // Start shadow at 1.0.
        let model_a = make_model(1, 1.0);
        let mut ema = GaussianEma::new(&model_a, 0.9);

        // After enough updates the shadow position should converge near 0.0.
        for _ in 0..2000 {
            ema.update(&model_b);
        }

        // The EMA approaches 0; at decay=0.9 and 2000 steps the residual is tiny.
        let shadow_pos = ema.shadow_positions();
        for &v in shadow_pos {
            assert!(
                v.abs() < 0.01,
                "shadow position {v} did not converge to 0.0"
            );
        }

        // Apply to a fresh model and verify the values were copied.
        let mut eval = make_model(1, 99.0);
        ema.apply_to(&mut eval);
        assert!(
            eval.gaussians[0].position[0].abs() < 0.01,
            "apply_to position not converged"
        );
    }

    // ----- apply_to overwrites only the right fields -----------------------

    #[test]
    fn apply_to_overwrites_positions_and_rotations() {
        let model_target = make_model(3, 2.0);
        let mut ema = GaussianEma::new(&model_target, 0.5);
        // Run a few updates with the same model to stabilise quickly.
        for _ in 0..20 {
            ema.update(&model_target);
        }

        let mut apply_target = make_model(3, 99.0);
        ema.apply_to(&mut apply_target);

        // Positions must be close to 2.0.
        for g in &apply_target.gaussians {
            for &p in &g.position {
                assert!((p - 2.0).abs() < 0.01, "position {p} not close to 2.0");
            }
        }
    }

    // ----- apply_to no-op on size mismatch ---------------------------------

    #[test]
    fn apply_to_no_op_when_sizes_differ() {
        let model_small = make_model(2, 1.0);
        let ema = GaussianEma::new(&model_small, 0.9);

        // Apply to a model of different size.
        let mut model_large = make_model(5, 42.0);
        ema.apply_to(&mut model_large);

        // Values must be unchanged.
        for g in &model_large.gaussians {
            assert_eq!(g.position[0], 42.0);
        }
    }

    // ----- initialise from model -------------------------------------------

    #[test]
    fn new_shadow_equals_model() {
        let model = make_model(4, 3.1);
        let ema = GaussianEma::new(&model, 0.99);

        let pos = ema.shadow_positions();
        for &v in pos {
            assert!((v - 3.1).abs() < 1e-6, "initial shadow position {v} != 3.1");
        }
    }

    // ----- decay accessor ---------------------------------------------------

    #[test]
    fn decay_accessor_returns_nominal() {
        let model = make_model(1, 0.0);
        let ema = GaussianEma::new(&model, 0.9999);
        assert!((ema.decay() - 0.9999).abs() < 1e-7);
    }

    // ----- quaternion double-cover handling ---------------------------------

    fn model_with_rotation(rotation: [f32; 4]) -> GaussianModel {
        let attr = GaussianAttributes {
            position: [0.0; 3],
            _pad0: 0.0,
            rotation,
            scale: [0.0; 3],
            opacity: 0.0,
        };
        GaussianModel {
            gaussians: vec![attr],
            sh_coeffs: vec![0.0; 3],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0, 0.0, 0.0]],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![false],
        }
    }

    #[test]
    fn rotation_ema_sign_alignment_prevents_norm_collapse() {
        // Shadow starts at q = (0, 0, 0, 1).
        let model_pos = model_with_rotation([0.0, 0.0, 0.0, 1.0]);
        let mut ema = GaussianEma::new(&model_pos, 0.5);

        // Update with -q = (0, 0, 0, -1): the SAME rotation via the double
        // cover. A naive component-wise blend would average q with -q and
        // the shadow's norm would collapse toward 0; with sign alignment the
        // blend should stay a valid, ~unit-norm quaternion pointing the same
        // direction as before.
        let model_neg = model_with_rotation([0.0, 0.0, 0.0, -1.0]);
        ema.update(&model_neg);

        let shadow = ema.shadow_rotations();
        let norm = (shadow[0] * shadow[0]
            + shadow[1] * shadow[1]
            + shadow[2] * shadow[2]
            + shadow[3] * shadow[3])
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "shadow quaternion norm should stay ~1.0 across a sign flip, got {norm}"
        );
        assert!(
            (shadow[3] - 1.0).abs() < 1e-4,
            "sign-aligned blend of q with itself should stay at w=1.0, got w={}",
            shadow[3]
        );
    }

    #[test]
    fn rotation_ema_apply_to_writes_unit_quaternion_across_sign_flips() {
        let model_pos = model_with_rotation([0.0, 0.0, 0.0, 1.0]);
        let mut ema = GaussianEma::new(&model_pos, 0.5);

        // Alternate feeding +q and -q across several updates -- every
        // resulting shadow quaternion written via `apply_to` must remain
        // (numerically) a unit quaternion.
        let signs = [-1.0f32, 1.0, -1.0, 1.0, -1.0];
        for &s in &signs {
            ema.update(&model_with_rotation([0.0, 0.0, 0.0, s]));
        }

        let mut eval = model_with_rotation([9.0, 9.0, 9.0, 9.0]);
        ema.apply_to(&mut eval);
        let r = eval.gaussians[0].rotation;
        let norm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "apply_to must write a unit-norm quaternion, got norm={norm}"
        );
    }
}
