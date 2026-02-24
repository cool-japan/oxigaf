//! Per-parameter Adam optimiser with group-wise learning rates.
//!
//! Each learnable parameter group (position, rotation, scale, opacity, SH,
//! offset) gets its own learning rate and independent Adam first/second moment
//! estimates.  Position learning rate follows an exponential decay schedule.

use oxigaf_render::gaussian::GaussianModel;

use crate::config::OptimizerConfig;

// ---------------------------------------------------------------------------
// Parameter groups
// ---------------------------------------------------------------------------

/// Identifies a learnable parameter group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterGroup {
    Position,
    Rotation,
    Scale,
    Opacity,
    Sh,
    Offset,
}

impl ParameterGroup {
    /// Number of scalar elements per Gaussian for this group.
    pub fn elem_size(self) -> usize {
        match self {
            Self::Position => 3,
            Self::Rotation => 4,
            Self::Scale => 3,
            Self::Opacity => 1,
            Self::Sh => 0, // variable — determined at runtime
            Self::Offset => 3,
        }
    }

    /// Human-readable name (used in checkpoint serialisation).
    pub fn name(self) -> &'static str {
        match self {
            Self::Position => "position",
            Self::Rotation => "rotation",
            Self::Scale => "scale",
            Self::Opacity => "opacity",
            Self::Sh => "sh",
            Self::Offset => "offset",
        }
    }
}

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

/// Gradients for every learnable parameter, stored as flat `Vec<f32>`.
///
/// The layout within each vector is `[N × elem_size]` where `N` is the number
/// of Gaussians.
#[derive(Debug, Clone)]
pub struct Gradients {
    /// `N × 3` — ∂L/∂position.
    pub position: Vec<f32>,
    /// `N × 4` — ∂L/∂rotation.
    pub rotation: Vec<f32>,
    /// `N × 3` — ∂L/∂log_scale.
    pub scale: Vec<f32>,
    /// `N × 1` — ∂L/∂inverse_sigmoid_opacity.
    pub opacity: Vec<f32>,
    /// `N × C` — ∂L/∂sh_coefficients  (C = (degree+1)² × 3).
    pub sh: Vec<f32>,
    /// `N × 3` — ∂L/∂local_offset.
    pub offset: Vec<f32>,
}

impl Gradients {
    /// Create an all-zeros gradient buffer for `n` Gaussians with `sh_channels`
    /// SH coefficients per Gaussian.
    pub fn zeros(n: usize, sh_channels: usize) -> Self {
        Self {
            position: vec![0.0; n * 3],
            rotation: vec![0.0; n * 4],
            scale: vec![0.0; n * 3],
            opacity: vec![0.0; n],
            sh: vec![0.0; n * sh_channels],
            offset: vec![0.0; n * 3],
        }
    }

    /// Number of Gaussians this gradient buffer was sized for (derived from
    /// `position` length).
    pub fn num_gaussians(&self) -> usize {
        self.position.len() / 3
    }
}

// ---------------------------------------------------------------------------
// Adam state (per-group)
// ---------------------------------------------------------------------------

/// First- and second-moment accumulators for one parameter group.
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment (mean of gradients).
    pub m: Vec<f32>,
    /// Second moment (mean of squared gradients).
    pub v: Vec<f32>,
    /// Number of update steps performed.
    pub t: u32,
}

impl AdamState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            t: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// GaussianOptimizer
// ---------------------------------------------------------------------------

/// Per-parameter Adam optimiser for a [`GaussianModel`].
#[derive(Debug, Clone)]
pub struct GaussianOptimizer {
    config: OptimizerConfig,
    /// State for each parameter group (one per [`ParameterGroup`] variant).
    pub position: AdamState,
    pub rotation: AdamState,
    pub scale: AdamState,
    pub opacity: AdamState,
    pub sh: AdamState,
    pub offset: AdamState,
    /// Number of SH coefficients per Gaussian.
    sh_channels: usize,
}

impl GaussianOptimizer {
    /// Allocate optimiser state matching the current `model` size.
    pub fn new(config: &OptimizerConfig, model: &GaussianModel) -> Self {
        let n = model.len();
        let sh_channels = ((model.sh_degree + 1) * (model.sh_degree + 1) * 3) as usize;
        Self {
            config: config.clone(),
            position: AdamState::new(n * 3),
            rotation: AdamState::new(n * 4),
            scale: AdamState::new(n * 3),
            opacity: AdamState::new(n),
            sh: AdamState::new(n * sh_channels),
            offset: AdamState::new(n * 3),
            sh_channels,
        }
    }

    // ----- learning-rate schedule -------------------------------------------

    /// Exponentially-decayed learning rate for position parameters.
    pub fn position_lr(&self, iteration: u32) -> f32 {
        let t = (iteration as f32) / (self.config.position_lr_decay_steps as f32).max(1.0);
        let t = t.min(1.0);
        let log_start = self.config.lr_position.ln();
        let log_end = self.config.lr_position_final.ln();
        ((1.0 - t) * log_start + t * log_end).exp()
    }

    // ----- optimiser step ---------------------------------------------------

    /// Perform one Adam update for **all** parameter groups.
    pub fn step(&mut self, model: &mut GaussianModel, gradients: &Gradients, iteration: u32) {
        let n = model.len();
        let cfg = &self.config;

        // Position (with exponential LR decay)
        {
            let lr = self.position_lr(iteration);
            let s = &mut self.position;
            s.t += 1;
            let bc1 = 1.0 - cfg.beta1.powi(s.t as i32);
            let bc2 = 1.0 - cfg.beta2.powi(s.t as i32);
            for i in 0..n {
                for j in 0..3 {
                    let idx = i * 3 + j;
                    adam_scalar(
                        &mut model.gaussians[i].position[j],
                        gradients.position[idx],
                        &mut s.m[idx],
                        &mut s.v[idx],
                        lr,
                        bc1,
                        bc2,
                        cfg.beta1,
                        cfg.beta2,
                        cfg.epsilon,
                    );
                }
            }
        }

        // Rotation
        {
            let lr = cfg.lr_rotation;
            let s = &mut self.rotation;
            s.t += 1;
            let bc1 = 1.0 - cfg.beta1.powi(s.t as i32);
            let bc2 = 1.0 - cfg.beta2.powi(s.t as i32);
            for i in 0..n {
                for j in 0..4 {
                    let idx = i * 4 + j;
                    adam_scalar(
                        &mut model.gaussians[i].rotation[j],
                        gradients.rotation[idx],
                        &mut s.m[idx],
                        &mut s.v[idx],
                        lr,
                        bc1,
                        bc2,
                        cfg.beta1,
                        cfg.beta2,
                        cfg.epsilon,
                    );
                }
            }
        }

        // Scale
        {
            let lr = cfg.lr_scale;
            let s = &mut self.scale;
            s.t += 1;
            let bc1 = 1.0 - cfg.beta1.powi(s.t as i32);
            let bc2 = 1.0 - cfg.beta2.powi(s.t as i32);
            for i in 0..n {
                for j in 0..3 {
                    let idx = i * 3 + j;
                    adam_scalar(
                        &mut model.gaussians[i].scale[j],
                        gradients.scale[idx],
                        &mut s.m[idx],
                        &mut s.v[idx],
                        lr,
                        bc1,
                        bc2,
                        cfg.beta1,
                        cfg.beta2,
                        cfg.epsilon,
                    );
                }
            }
        }

        // Opacity
        {
            let lr = cfg.lr_opacity;
            let s = &mut self.opacity;
            s.t += 1;
            let bc1 = 1.0 - cfg.beta1.powi(s.t as i32);
            let bc2 = 1.0 - cfg.beta2.powi(s.t as i32);
            for i in 0..n {
                adam_scalar(
                    &mut model.gaussians[i].opacity,
                    gradients.opacity[i],
                    &mut s.m[i],
                    &mut s.v[i],
                    lr,
                    bc1,
                    bc2,
                    cfg.beta1,
                    cfg.beta2,
                    cfg.epsilon,
                );
            }
        }

        // Spherical-Harmonics coefficients
        {
            let lr = cfg.lr_sh;
            let s = &mut self.sh;
            s.t += 1;
            let bc1 = 1.0 - cfg.beta1.powi(s.t as i32);
            let bc2 = 1.0 - cfg.beta2.powi(s.t as i32);
            for idx in 0..model.sh_coeffs.len() {
                adam_scalar(
                    &mut model.sh_coeffs[idx],
                    gradients.sh[idx],
                    &mut s.m[idx],
                    &mut s.v[idx],
                    lr,
                    bc1,
                    bc2,
                    cfg.beta1,
                    cfg.beta2,
                    cfg.epsilon,
                );
            }
        }

        // Local offsets
        {
            let lr = cfg.lr_offset;
            let s = &mut self.offset;
            s.t += 1;
            let bc1 = 1.0 - cfg.beta1.powi(s.t as i32);
            let bc2 = 1.0 - cfg.beta2.powi(s.t as i32);
            for i in 0..n {
                for j in 0..3 {
                    let idx = i * 3 + j;
                    adam_scalar(
                        &mut model.local_offsets[i][j],
                        gradients.offset[idx],
                        &mut s.m[idx],
                        &mut s.v[idx],
                        lr,
                        bc1,
                        bc2,
                        cfg.beta1,
                        cfg.beta2,
                        cfg.epsilon,
                    );
                }
            }
        }
    }

    // ----- density-control bookkeeping --------------------------------------

    /// Adjust internal buffers after density control has modified the model.
    ///
    /// * `keep_mask` — length = old number of Gaussians; `true` = kept.
    /// * `num_added` — number of new Gaussians appended **after** compaction.
    pub fn handle_densify(&mut self, keep_mask: &[bool], num_added: usize) {
        compact_and_extend(&mut self.position, keep_mask, num_added, 3);
        compact_and_extend(&mut self.rotation, keep_mask, num_added, 4);
        compact_and_extend(&mut self.scale, keep_mask, num_added, 3);
        compact_and_extend(&mut self.opacity, keep_mask, num_added, 1);
        compact_and_extend(&mut self.sh, keep_mask, num_added, self.sh_channels);
        compact_and_extend(&mut self.offset, keep_mask, num_added, 3);
    }

    // ----- checkpoint helpers -----------------------------------------------

    /// Serialisable snapshot of every group's Adam state.
    pub fn checkpoint_states(&self) -> Vec<(String, Vec<f32>, Vec<f32>, u32)> {
        vec![
            (
                "position".into(),
                self.position.m.clone(),
                self.position.v.clone(),
                self.position.t,
            ),
            (
                "rotation".into(),
                self.rotation.m.clone(),
                self.rotation.v.clone(),
                self.rotation.t,
            ),
            (
                "scale".into(),
                self.scale.m.clone(),
                self.scale.v.clone(),
                self.scale.t,
            ),
            (
                "opacity".into(),
                self.opacity.m.clone(),
                self.opacity.v.clone(),
                self.opacity.t,
            ),
            ("sh".into(), self.sh.m.clone(), self.sh.v.clone(), self.sh.t),
            (
                "offset".into(),
                self.offset.m.clone(),
                self.offset.v.clone(),
                self.offset.t,
            ),
        ]
    }

    /// Restore Adam state from a checkpoint.
    pub fn restore_states(&mut self, states: &[(String, Vec<f32>, Vec<f32>, u32)]) {
        for (name, m, v, t) in states {
            let target = match name.as_str() {
                "position" => &mut self.position,
                "rotation" => &mut self.rotation,
                "scale" => &mut self.scale,
                "opacity" => &mut self.opacity,
                "sh" => &mut self.sh,
                "offset" => &mut self.offset,
                other => {
                    tracing::warn!("Unknown optimizer group in checkpoint: {other}");
                    continue;
                }
            };
            target.m = m.clone();
            target.v = v.clone();
            target.t = *t;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Single-scalar Adam update (bias-corrected).
#[inline]
#[allow(clippy::too_many_arguments)]
fn adam_scalar(
    param: &mut f32,
    grad: f32,
    m: &mut f32,
    v: &mut f32,
    lr: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) {
    *m = beta1 * *m + (1.0 - beta1) * grad;
    *v = beta2 * *v + (1.0 - beta2) * grad * grad;
    let m_hat = *m / bias_correction1;
    let v_hat = *v / bias_correction2;
    *param -= lr * m_hat / (v_hat.sqrt() + epsilon);
}

/// Compact an [`AdamState`] to keep only the entries where `keep[i]` is true,
/// then extend with zeros for `num_added` new Gaussians.
fn compact_and_extend(state: &mut AdamState, keep: &[bool], num_added: usize, elem_size: usize) {
    let mut new_m =
        Vec::with_capacity((keep.iter().filter(|&&k| k).count() + num_added) * elem_size);
    let mut new_v = Vec::with_capacity(new_m.capacity());

    for (i, &k) in keep.iter().enumerate() {
        if k {
            let start = i * elem_size;
            let end = start + elem_size;
            if end <= state.m.len() {
                new_m.extend_from_slice(&state.m[start..end]);
                new_v.extend_from_slice(&state.v[start..end]);
            } else {
                // Guard: if state is shorter than expected, pad with zeros.
                new_m.extend(std::iter::repeat_n(0.0f32, elem_size));
                new_v.extend(std::iter::repeat_n(0.0f32, elem_size));
            }
        }
    }

    // Zeros for newly added Gaussians.
    new_m.resize(new_m.len() + num_added * elem_size, 0.0);
    new_v.resize(new_v.len() + num_added * elem_size, 0.0);

    state.m = new_m;
    state.v = new_v;
    // t is *not* reset — it continues counting.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_scalar_moves_toward_minimum() {
        // Minimise f(x) = x² → grad = 2x
        let mut x = 5.0_f32;
        let mut m = 0.0_f32;
        let mut v = 0.0_f32;
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps: f32 = 1e-8;
        let lr: f32 = 0.1;

        for t in 1..=200 {
            let grad = 2.0 * x;
            let bc1 = 1.0 - beta1.powi(t);
            let bc2 = 1.0 - beta2.powi(t);
            adam_scalar(
                &mut x, grad, &mut m, &mut v, lr, bc1, bc2, beta1, beta2, eps,
            );
        }

        assert!(x.abs() < 0.1, "expected x ≈ 0, got {x}");
    }

    #[test]
    fn compact_and_extend_works() {
        let mut state = AdamState {
            m: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            v: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            t: 5,
        };
        let keep = [true, false, true]; // elem_size = 2 → keep [1,2] and [5,6]
        compact_and_extend(&mut state, &keep, 1, 2);

        assert_eq!(state.m, vec![1.0, 2.0, 5.0, 6.0, 0.0, 0.0]);
        assert_eq!(state.v, vec![10.0, 20.0, 50.0, 60.0, 0.0, 0.0]);
        assert_eq!(state.t, 5); // unchanged
    }
}
