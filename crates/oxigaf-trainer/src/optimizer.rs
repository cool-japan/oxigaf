//! Per-parameter Adam optimiser with group-wise learning rates.
//!
//! Each learnable parameter group (position, rotation, scale, opacity, SH,
//! offset) gets its own learning rate and independent Adam first/second moment
//! estimates.  Position learning rate follows an exponential decay schedule.

use oxigaf_render::gaussian::GaussianModel;

use crate::config::OptimizerConfig;
use crate::TrainerError;

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

    /// The six parameter groups as separate flat vectors, in
    /// [`ParameterGroup`] order (position, rotation, scale, opacity, SH,
    /// offset).
    ///
    /// This is the shape every group-wise utility in the crate speaks —
    /// [`crate::gradient_clipping::GradientClipper::step`],
    /// [`crate::gradient_accumulation::GradientAccumulator::accumulate`] and
    /// [`crate::data_parallel::GradientAggregator::submit_gradients`] all take
    /// `&[Vec<f32>]` — so it is the single conversion point rather than six
    /// ad-hoc `vec![...]` literals scattered across call sites.
    pub fn to_group_vecs(&self) -> Vec<Vec<f32>> {
        vec![
            self.position.clone(),
            self.rotation.clone(),
            self.scale.clone(),
            self.opacity.clone(),
            self.sh.clone(),
            self.offset.clone(),
        ]
    }

    /// Element counts of the six groups, in [`to_group_vecs`](Self::to_group_vecs)
    /// order — what [`crate::gradient_accumulation::GradientAccumulator::initialize`]
    /// wants.
    pub fn group_sizes(&self) -> [usize; 6] {
        [
            self.position.len(),
            self.rotation.len(),
            self.scale.len(),
            self.opacity.len(),
            self.sh.len(),
            self.offset.len(),
        ]
    }

    /// Overwrite the six groups from vectors laid out as
    /// [`to_group_vecs`](Self::to_group_vecs) produced them.
    ///
    /// # Errors
    ///
    /// [`TrainerError::GradientSizeMismatch`] if the slice does not hold
    /// exactly six groups, or if any group's length differs from this buffer's.
    /// Silently truncating instead would drop the tail of a parameter group's
    /// gradient — a corrupted step that no later check could catch.
    pub fn set_from_group_vecs(&mut self, groups: &[Vec<f32>]) -> Result<(), TrainerError> {
        if groups.len() != 6 {
            return Err(TrainerError::GradientSizeMismatch {
                expected: 6,
                actual: groups.len(),
            });
        }
        let expected = self.group_sizes();
        for (idx, group) in groups.iter().enumerate() {
            if group.len() != expected[idx] {
                return Err(TrainerError::GradientSizeMismatch {
                    expected: expected[idx],
                    actual: group.len(),
                });
            }
        }
        self.position.copy_from_slice(&groups[0]);
        self.rotation.copy_from_slice(&groups[1]);
        self.scale.copy_from_slice(&groups[2]);
        self.opacity.copy_from_slice(&groups[3]);
        self.sh.copy_from_slice(&groups[4]);
        self.offset.copy_from_slice(&groups[5]);
        Ok(())
    }

    /// Accumulate (`+=`) each element from `other` element-wise.
    ///
    /// Fields are zipped so mismatched lengths silently skip extra elements
    /// of the longer slice — callers should ensure both buffers are sized
    /// identically.
    pub fn accumulate_from(&mut self, other: &Gradients) {
        add_elementwise(&mut self.position, &other.position);
        add_elementwise(&mut self.rotation, &other.rotation);
        add_elementwise(&mut self.scale, &other.scale);
        add_elementwise(&mut self.opacity, &other.opacity);
        add_elementwise(&mut self.sh, &other.sh);
        add_elementwise(&mut self.offset, &other.offset);
    }

    /// Scale every element by `factor` (in-place).
    pub fn scale(&mut self, factor: f32) {
        for x in self.position.iter_mut() {
            *x *= factor;
        }
        for x in self.rotation.iter_mut() {
            *x *= factor;
        }
        for x in self.scale.iter_mut() {
            *x *= factor;
        }
        for x in self.opacity.iter_mut() {
            *x *= factor;
        }
        for x in self.sh.iter_mut() {
            *x *= factor;
        }
        for x in self.offset.iter_mut() {
            *x *= factor;
        }
    }
}

/// Add `src` into `dst` element-wise (zip stops at the shorter slice).
#[inline]
fn add_elementwise(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
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
    /// Running sum of gradients across micro-batches for gradient accumulation.
    accumulated_gradients: Option<Gradients>,
    /// Number of micro-batches accumulated so far since the last optimizer step.
    accumulation_step: u32,
    /// Multiplier applied to **every** group's learning rate.
    ///
    /// This is where an external schedule ([`crate::lr_scheduler`]) enters the
    /// update; `1.0` reproduces the configured rates exactly.  Kept private so
    /// it can only be set through the validating [`GaussianOptimizer::set_lr_scale`].
    lr_scale: f32,
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
            accumulated_gradients: None,
            accumulation_step: 0,
            lr_scale: 1.0,
        }
    }

    // ----- learning-rate schedule -------------------------------------------

    /// Exponentially-decayed learning rate for position parameters, including
    /// the current [`lr_scale`](Self::lr_scale).
    pub fn position_lr(&self, iteration: u32) -> f32 {
        let t = (iteration as f32) / (self.config.position_lr_decay_steps as f32).max(1.0);
        let t = t.min(1.0);
        let log_start = self.config.lr_position.ln();
        let log_end = self.config.lr_position_final.ln();
        ((1.0 - t) * log_start + t * log_end).exp() * self.lr_scale
    }

    /// The multiplier currently applied to every group's learning rate.
    pub fn lr_scale(&self) -> f32 {
        self.lr_scale
    }

    /// Set the multiplier applied to every group's learning rate.
    ///
    /// This is how an [`crate::lr_scheduler::LrScheduler`] reaches the update:
    /// build the schedule with `base_lr = 1.0` and feed its value here once per
    /// iteration.  It composes with — rather than replaces — the optimizer's own
    /// exponential position decay.
    ///
    /// # Errors
    ///
    /// [`TrainerError::ParameterOutOfRange`] for a non-finite or negative
    /// scale: either would silently turn the whole update into `NaN` or into
    /// gradient *ascent*, which is far worse than refusing it here.
    pub fn set_lr_scale(&mut self, scale: f32) -> Result<(), TrainerError> {
        if !scale.is_finite() || scale < 0.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "lr_scale".into(),
                value: format!("{scale}"),
                expected: ">= 0 and finite".into(),
            });
        }
        self.lr_scale = scale;
        Ok(())
    }

    // ----- optimiser step ---------------------------------------------------

    /// Validate that `gradients` and every Adam-state buffer match the
    /// current model size (`n` Gaussians, `sh_len` total SH coefficients).
    ///
    /// # Errors
    /// [`TrainerError::GradientSizeMismatch`] naming the first mismatched
    /// buffer's actual/expected element counts.
    fn validate_step_shapes(
        &self,
        gradients: &Gradients,
        n: usize,
        sh_len: usize,
    ) -> Result<(), TrainerError> {
        // Each Adam state's `m`/`v` buffers are always allocated together
        // (see `AdamState::new`) and kept in sync by `handle_densify`, so
        // checking `m` alone is sufficient to catch a desync.
        let buffers: [(usize, usize); 12] = [
            (gradients.position.len(), n * 3), // position gradient
            (self.position.m.len(), n * 3),    // position Adam state
            (gradients.rotation.len(), n * 4), // rotation gradient
            (self.rotation.m.len(), n * 4),    // rotation Adam state
            (gradients.scale.len(), n * 3),    // scale gradient
            (self.scale.m.len(), n * 3),       // scale Adam state
            (gradients.opacity.len(), n),      // opacity gradient
            (self.opacity.m.len(), n),         // opacity Adam state
            (gradients.sh.len(), sh_len),      // SH gradient
            (self.sh.m.len(), sh_len),         // SH Adam state
            (gradients.offset.len(), n * 3),   // offset gradient
            (self.offset.m.len(), n * 3),      // offset Adam state
        ];
        for (actual, expected) in buffers {
            if actual != expected {
                return Err(TrainerError::GradientSizeMismatch { expected, actual });
            }
        }
        Ok(())
    }

    /// Perform one Adam update for **all** parameter groups.
    ///
    /// # Errors
    /// [`TrainerError::GradientSizeMismatch`] if `gradients` (or this
    /// optimiser's own Adam-state buffers) do not match `model`'s current
    /// size — e.g. a `Gradients` built for a different Gaussian count, or a
    /// `model.sh_degree` that changed since this optimiser was constructed
    /// without a corresponding resize of its SH state.
    pub fn step(
        &mut self,
        model: &mut GaussianModel,
        gradients: &Gradients,
        iteration: u32,
    ) -> Result<(), TrainerError> {
        let n = model.len();
        self.validate_step_shapes(gradients, n, model.sh_coeffs.len())?;
        let lr_scale = self.lr_scale;
        let cfg = &self.config;

        // Position (with exponential LR decay)
        {
            // `position_lr` already folds in `lr_scale`.
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
            let lr = cfg.lr_rotation * lr_scale;
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
            let lr = cfg.lr_scale * lr_scale;
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
            let lr = cfg.lr_opacity * lr_scale;
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
            let lr = cfg.lr_sh * lr_scale;
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
            let lr = cfg.lr_offset * lr_scale;
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

        Ok(())
    }

    // ----- gradient clipping ------------------------------------------------

    /// Clip gradients by global L2 norm across all parameter groups.
    ///
    /// Computes the global L2 norm of all gradient tensors (position, rotation,
    /// scale, opacity, sh, offset). If the norm exceeds `max_norm`, all gradients
    /// are scaled uniformly by `max_norm / norm`.
    ///
    /// Returns the norm **before** any clipping was applied.
    ///
    /// # Errors
    /// [`TrainerError::NanDetected`] / [`TrainerError::InfDetected`] if any
    /// gradient element is non-finite, naming the offending parameter group
    /// and index. Previously a NaN silently defeated the `norm > max_norm`
    /// check (NaN compares `false` to everything) and passed straight
    /// through unclipped, while an Inf silently zeroed every gradient
    /// (`max_norm / Inf == 0`) without reporting anything. On `Err`,
    /// `gradients` is left **unmodified** — unlike the old Inf behaviour,
    /// callers must not assume any clipping happened.
    pub fn clip_grad_norm(
        &self,
        gradients: &mut Gradients,
        max_norm: f32,
    ) -> Result<f32, TrainerError> {
        let groups: [(&str, &Vec<f32>); 6] = [
            ("position", &gradients.position),
            ("rotation", &gradients.rotation),
            ("scale", &gradients.scale),
            ("opacity", &gradients.opacity),
            ("sh", &gradients.sh),
            ("offset", &gradients.offset),
        ];

        // Single pass: accumulate the squared sum while also recording the
        // first non-finite element found (if any), so this costs no more
        // than the original unguarded fold.
        let mut sum_sq = 0.0_f32;
        let mut non_finite: Option<(&str, usize, f32)> = None;
        for (name, buf) in groups {
            for (idx, &x) in buf.iter().enumerate() {
                if non_finite.is_none() && !x.is_finite() {
                    non_finite = Some((name, idx, x));
                }
                sum_sq += x * x;
            }
        }

        if let Some((parameter, index, value)) = non_finite {
            return if value.is_nan() {
                Err(TrainerError::NanDetected {
                    parameter: parameter.to_string(),
                    index,
                })
            } else {
                Err(TrainerError::InfDetected {
                    parameter: parameter.to_string(),
                    index,
                })
            };
        }

        let norm = sum_sq.sqrt();

        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for g in [
                &mut gradients.position,
                &mut gradients.rotation,
                &mut gradients.scale,
                &mut gradients.opacity,
                &mut gradients.sh,
                &mut gradients.offset,
            ] {
                for x in g.iter_mut() {
                    *x *= scale;
                }
            }
        }

        Ok(norm)
    }

    /// Clip each gradient value per-parameter group to `[-max_value, max_value]`.
    ///
    /// Every element of every gradient tensor is clamped independently.
    pub fn clip_grad_value(&self, gradients: &mut Gradients, max_value: f32) {
        for g in [
            &mut gradients.position,
            &mut gradients.rotation,
            &mut gradients.scale,
            &mut gradients.opacity,
            &mut gradients.sh,
            &mut gradients.offset,
        ] {
            for x in g.iter_mut() {
                *x = x.clamp(-max_value, max_value);
            }
        }
    }

    // ----- gradient accumulation --------------------------------------------

    /// Accumulate gradients from one micro-batch.
    ///
    /// Call this method once per micro-batch.  After `accumulation_steps`
    /// calls, invoke [`Self::step_accumulated`] to apply the averaged update
    /// and reset the accumulation buffer.
    ///
    /// If the accumulation buffer has not been initialised yet (or was just
    /// reset after a step), it is created as all-zeros matching `gradients`'
    /// shape on the first call.
    pub fn accumulate_gradients(&mut self, gradients: &Gradients) {
        match self.accumulated_gradients {
            None => {
                // Initialise buffer as zeros with the same shape as `gradients`.
                let n = gradients.num_gaussians();
                let sh_len = gradients.sh.len();
                let sh_channels = sh_len.checked_div(n).unwrap_or(0);
                let mut buf = Gradients::zeros(n, sh_channels);
                buf.accumulate_from(gradients);
                self.accumulated_gradients = Some(buf);
            }
            Some(ref mut buf) => {
                buf.accumulate_from(gradients);
            }
        }
        self.accumulation_step += 1;
    }

    /// Apply an Adam update using the mean of all accumulated micro-batch
    /// gradients, then reset the accumulation state.
    ///
    /// # Parameters
    /// * `model` — model whose parameters will be updated.
    /// * `accumulation_steps` — total number of micro-batches to wait for
    ///   before applying the update.
    /// * `iteration` — global training iteration (used for LR scheduling).
    ///
    /// # Returns
    /// * `Ok(Some(n))` — the update was applied; `n` is the number of
    ///   micro-batches that were averaged together.
    /// * `Ok(None)` — not enough micro-batches have been accumulated yet;
    ///   the caller should continue calling `accumulate_gradients`.
    ///
    /// # Errors
    /// Propagates [`TrainerError::GradientSizeMismatch`] from
    /// [`Self::step`] if the accumulated buffer no longer matches `model`'s
    /// size.
    pub fn step_accumulated(
        &mut self,
        model: &mut GaussianModel,
        accumulation_steps: u32,
        iteration: u32,
    ) -> Result<Option<u32>, TrainerError> {
        if self.accumulation_step < accumulation_steps {
            return Ok(None);
        }

        let n_steps = self.accumulation_step;

        // Take ownership of the accumulated buffer to release the borrow on
        // `self.accumulated_gradients` before calling `self.step()`.
        let mut averaged = match self.accumulated_gradients.take() {
            Some(g) => g,
            None => return Ok(None),
        };

        // Average in-place.
        if n_steps > 1 {
            averaged.scale(1.0 / n_steps as f32);
        }

        // Reset counter before calling step (step may error; we still reset).
        self.accumulation_step = 0;

        self.step(model, &averaged, iteration)?;

        Ok(Some(n_steps))
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

        // Also compact the accumulation buffer if one is in-flight.
        if let Some(ref mut buf) = self.accumulated_gradients {
            compact_and_extend_vec(&mut buf.position, keep_mask, num_added, 3);
            compact_and_extend_vec(&mut buf.rotation, keep_mask, num_added, 4);
            compact_and_extend_vec(&mut buf.scale, keep_mask, num_added, 3);
            compact_and_extend_vec(&mut buf.opacity, keep_mask, num_added, 1);
            compact_and_extend_vec(&mut buf.sh, keep_mask, num_added, self.sh_channels);
            compact_and_extend_vec(&mut buf.offset, keep_mask, num_added, 3);
        }
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

/// Compact a flat gradient `Vec<f32>` to keep only the entries where
/// `keep[i]` is true, then append zeros for `num_added` new Gaussians.
fn compact_and_extend_vec(buf: &mut Vec<f32>, keep: &[bool], num_added: usize, elem_size: usize) {
    let kept_count = keep.iter().filter(|&&k| k).count();
    let mut new_buf = Vec::with_capacity((kept_count + num_added) * elem_size);

    for (i, &k) in keep.iter().enumerate() {
        if k {
            let start = i * elem_size;
            let end = start + elem_size;
            if end <= buf.len() {
                new_buf.extend_from_slice(&buf[start..end]);
            } else {
                new_buf.extend(std::iter::repeat_n(0.0_f32, elem_size));
            }
        }
    }
    new_buf.resize(new_buf.len() + num_added * elem_size, 0.0);
    *buf = new_buf;
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
    fn clip_grad_norm_reduces_norm_correctly() {
        // Build a tiny GaussianModel and optimizer purely to call the method.
        // We only need the GaussianOptimizer value for the receiver; the method
        // only reads `self.config` and `self.sh_channels` (not used in clipping).
        let config = crate::config::OptimizerConfig::default();

        // We can't trivially create a GaussianModel here without a large set-up,
        // so test the core logic through the Gradients struct directly, calling
        // the method via a synthetic optimizer with sh_channels = 0.
        let dummy_model = make_tiny_model(2, 0);
        let opt = GaussianOptimizer::new(&config, &dummy_model);

        // Gradient buffer: N=2, sh_channels=0.
        // position: 2×3 = 6 elements, set to 1.0 each.
        // All others zero to keep the math simple.
        let mut grads = Gradients {
            position: vec![1.0_f32; 6],
            rotation: vec![0.0_f32; 8],
            scale: vec![0.0_f32; 6],
            opacity: vec![0.0_f32; 2],
            sh: vec![],
            offset: vec![0.0_f32; 6],
        };

        // Global L2 norm = sqrt(6 * 1²) = sqrt(6) ≈ 2.449
        let expected_norm = (6.0_f32).sqrt();
        let max_norm = 1.0_f32;
        let returned_norm = opt
            .clip_grad_norm(&mut grads, max_norm)
            .expect("all-finite gradients must not error");

        // Returned norm must equal norm *before* clipping.
        assert!(
            (returned_norm - expected_norm).abs() < 1e-5,
            "returned norm {returned_norm} != expected {expected_norm}"
        );

        // After clipping, every position element should be scaled by 1.0 / sqrt(6).
        let expected_val = 1.0 / expected_norm;
        for &v in &grads.position {
            assert!(
                (v - expected_val).abs() < 1e-5,
                "position element {v} != expected {expected_val}"
            );
        }
    }

    #[test]
    fn clip_grad_norm_no_op_when_within_bound() {
        let config = crate::config::OptimizerConfig::default();
        let dummy_model = make_tiny_model(1, 0);
        let opt = GaussianOptimizer::new(&config, &dummy_model);

        let mut grads = Gradients {
            position: vec![0.1_f32; 3],
            rotation: vec![0.0_f32; 4],
            scale: vec![0.0_f32; 3],
            opacity: vec![0.0_f32; 1],
            sh: vec![],
            offset: vec![0.0_f32; 3],
        };
        // norm = sqrt(3 * 0.01) = sqrt(0.03) ≈ 0.173, well within max_norm=1.0
        let norm = opt
            .clip_grad_norm(&mut grads, 1.0)
            .expect("all-finite gradients must not error");
        assert!(norm < 1.0, "norm should be under max_norm");
        // Values must be unchanged.
        for &v in &grads.position {
            assert!((v - 0.1).abs() < 1e-7, "should be unchanged, got {v}");
        }
    }

    #[test]
    fn clip_grad_norm_nan_returns_err_and_leaves_gradients_unmodified() {
        // Regression: a NaN used to poison `norm` (NaN compares false to
        // everything), so `norm > max_norm` was false and no clipping
        // happened — the poisoned gradients passed straight through
        // silently instead of being reported.
        let config = crate::config::OptimizerConfig::default();
        let dummy_model = make_tiny_model(1, 0);
        let opt = GaussianOptimizer::new(&config, &dummy_model);

        let mut grads = Gradients {
            position: vec![1.0_f32, f32::NAN, 1.0],
            rotation: vec![0.0_f32; 4],
            scale: vec![0.0_f32; 3],
            opacity: vec![0.0_f32; 1],
            sh: vec![],
            offset: vec![0.0_f32; 3],
        };
        let before = grads.position.clone();
        let result = opt.clip_grad_norm(&mut grads, 1.0);
        assert!(
            matches!(result, Err(TrainerError::NanDetected { .. })),
            "expected NanDetected, got {result:?}"
        );
        // Gradients must be left untouched on error.
        assert_eq!(before[0], grads.position[0]);
        assert!(grads.position[1].is_nan());
        assert_eq!(before[2], grads.position[2]);
    }

    #[test]
    fn clip_grad_norm_inf_returns_err_and_leaves_gradients_unmodified() {
        // Regression: `Inf > max_norm` is true, so the old code computed
        // `scale = max_norm / Inf = 0.0` and silently zeroed every
        // gradient instead of reporting the overflow.
        let config = crate::config::OptimizerConfig::default();
        let dummy_model = make_tiny_model(1, 0);
        let opt = GaussianOptimizer::new(&config, &dummy_model);

        let mut grads = Gradients {
            position: vec![1.0_f32, f32::INFINITY, 1.0],
            rotation: vec![0.0_f32; 4],
            scale: vec![0.0_f32; 3],
            opacity: vec![0.0_f32; 1],
            sh: vec![],
            offset: vec![0.0_f32; 3],
        };
        let result = opt.clip_grad_norm(&mut grads, 1.0);
        assert!(
            matches!(result, Err(TrainerError::InfDetected { .. })),
            "expected InfDetected, got {result:?}"
        );
        // Gradients must not have been silently zeroed.
        assert_eq!(grads.position[0], 1.0);
        assert!(grads.position[1].is_infinite());
    }

    #[test]
    fn clip_grad_value_clamps_all_elements() {
        let config = crate::config::OptimizerConfig::default();
        let dummy_model = make_tiny_model(2, 0);
        let opt = GaussianOptimizer::new(&config, &dummy_model);

        let mut grads = Gradients {
            position: vec![5.0, -7.0, 2.0, -3.0, 0.5, -0.5],
            rotation: vec![10.0; 8],
            scale: vec![-10.0; 6],
            opacity: vec![0.0_f32; 2],
            sh: vec![],
            offset: vec![1.5, -2.5, 3.0, -4.0, 0.0, 0.0],
        };
        opt.clip_grad_value(&mut grads, 2.0);

        // Every value must be in [-2.0, 2.0].
        for (field_name, field) in [
            ("position", &grads.position),
            ("rotation", &grads.rotation),
            ("scale", &grads.scale),
            ("opacity", &grads.opacity),
            ("offset", &grads.offset),
        ] {
            for &v in field.iter() {
                assert!(
                    (-2.0..=2.0).contains(&v),
                    "{field_name} value {v} out of [-2.0, 2.0]"
                );
            }
        }
    }

    /// Build a minimal GaussianModel with `n` Gaussians and the given `sh_degree`.
    fn make_tiny_model(n: usize, sh_degree: u32) -> oxigaf_render::gaussian::GaussianModel {
        use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
        let attr = GaussianAttributes {
            position: [0.0; 3],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            opacity: 0.5,
        };
        let sh_channels = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        GaussianModel {
            gaussians: vec![attr; n],
            sh_coeffs: vec![0.0; n * sh_channels],
            sh_degree,
            face_indices: vec![0; n],
            barycentric: vec![[1.0, 0.0, 0.0]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        }
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

    // -----------------------------------------------------------------------
    // Gradient accumulation tests
    // -----------------------------------------------------------------------

    /// Build a tiny Gradients buffer for `n` Gaussians with SH degree 0.
    ///
    /// SH degree 0 → `(0+1)² × 3 = 3` channels per Gaussian.
    fn make_tiny_gradients(n: usize, fill: f32) -> Gradients {
        // sh_degree=0 → sh_channels = (0+1)*(0+1)*3 = 3
        let sh_channels = 3_usize;
        Gradients {
            position: vec![fill; n * 3],
            rotation: vec![fill; n * 4],
            scale: vec![fill; n * 3],
            opacity: vec![fill; n],
            sh: vec![fill; n * sh_channels],
            offset: vec![fill; n * 3],
        }
    }

    #[test]
    fn step_rejects_mismatched_gradient_size_instead_of_panicking() {
        // Regression: `step` used to index `gradients.position[idx]` etc.
        // with no size validation, panicking with an index-out-of-bounds
        // for any caller whose `Gradients` was built for a different
        // Gaussian count than `model`.
        let config = crate::config::OptimizerConfig::default();
        let mut model = make_tiny_model(2, 0); // n=2 Gaussians
        let mut opt = GaussianOptimizer::new(&config, &model);

        // Gradients sized for only 1 Gaussian instead of 2.
        let mismatched = make_tiny_gradients(1, 0.1);
        let result = opt.step(&mut model, &mismatched, 0);
        assert!(
            matches!(result, Err(TrainerError::GradientSizeMismatch { .. })),
            "expected GradientSizeMismatch, got {result:?}"
        );
    }

    #[test]
    fn accumulate_once_and_step_applies_update() {
        let config = crate::config::OptimizerConfig::default();
        let mut model = make_tiny_model(2, 0);
        let mut opt = GaussianOptimizer::new(&config, &model);

        // Record original position values.
        let before: Vec<f32> = model.gaussians.iter().map(|g| g.position[0]).collect();

        let grads = make_tiny_gradients(2, 0.1);
        opt.accumulate_gradients(&grads);

        // With accumulation_steps = 1, step_accumulated should fire immediately.
        let result = opt
            .step_accumulated(&mut model, 1, 0)
            .expect("valid gradients");
        assert_eq!(result, Some(1), "expected step to fire");

        // Position should have changed (Adam update with non-zero gradient).
        let after: Vec<f32> = model.gaussians.iter().map(|g| g.position[0]).collect();
        assert!(
            before
                .iter()
                .zip(after.iter())
                .any(|(b, a)| (b - a).abs() > 1e-8),
            "model parameters should have changed after step"
        );
    }

    #[test]
    fn step_accumulated_returns_none_when_not_enough_microbatches() {
        let config = crate::config::OptimizerConfig::default();
        let mut model = make_tiny_model(2, 0);
        let mut opt = GaussianOptimizer::new(&config, &model);

        let grads = make_tiny_gradients(2, 0.1);
        opt.accumulate_gradients(&grads);

        // Require 4 micro-batches; only 1 accumulated — should return None.
        let result = opt
            .step_accumulated(&mut model, 4, 0)
            .expect("valid gradients");
        assert_eq!(
            result, None,
            "should not fire with only 1 of 4 micro-batches"
        );
        assert_eq!(opt.accumulation_step, 1);
    }

    #[test]
    fn accumulate_and_step_resets_state() {
        let config = crate::config::OptimizerConfig::default();
        let mut model = make_tiny_model(2, 0);
        let mut opt = GaussianOptimizer::new(&config, &model);

        let grads = make_tiny_gradients(2, 0.5);
        opt.accumulate_gradients(&grads);
        opt.step_accumulated(&mut model, 1, 0)
            .expect("valid gradients");

        // After step, accumulation_step must be reset.
        assert_eq!(
            opt.accumulation_step, 0,
            "step counter should reset after step"
        );
        assert!(
            opt.accumulated_gradients.is_none(),
            "gradient buffer should be cleared after step"
        );
    }

    #[test]
    fn gradient_averaging_correctness() {
        // Accumulate 4 identical micro-batch gradients of 1.0.
        // The averaged gradient should equal 1.0 (not 4.0).
        // We verify this indirectly by checking that the model update is the
        // same as a direct step with a single batch of 1.0 gradients.
        let config = crate::config::OptimizerConfig::default();

        let mut model_direct = make_tiny_model(1, 0);
        let mut opt_direct = GaussianOptimizer::new(&config, &model_direct);
        let single_grad = make_tiny_gradients(1, 1.0);
        opt_direct
            .step(&mut model_direct, &single_grad, 0)
            .expect("valid gradients");

        let mut model_accum = make_tiny_model(1, 0);
        let mut opt_accum = GaussianOptimizer::new(&config, &model_accum);
        let micro = make_tiny_gradients(1, 1.0);
        for _ in 0..4 {
            opt_accum.accumulate_gradients(&micro);
        }
        opt_accum
            .step_accumulated(&mut model_accum, 4, 0)
            .expect("valid gradients");

        // Both should result in the same parameter values.
        for i in 0..model_direct.gaussians.len() {
            for j in 0..3 {
                let d = model_direct.gaussians[i].position[j];
                let a = model_accum.gaussians[i].position[j];
                assert!(
                    (d - a).abs() < 1e-6,
                    "position[{i}][{j}]: direct={d}, accum={a}"
                );
            }
        }
    }

    #[test]
    fn multiple_accumulate_steps_before_fire() {
        let config = crate::config::OptimizerConfig::default();
        let mut model = make_tiny_model(2, 0);
        let mut opt = GaussianOptimizer::new(&config, &model);

        let grads = make_tiny_gradients(2, 0.2);
        // Accumulate 3 times; require 4.
        for _ in 0..3 {
            opt.accumulate_gradients(&grads);
        }
        assert_eq!(
            opt.step_accumulated(&mut model, 4, 0)
                .expect("valid gradients"),
            None,
            "should not fire after 3 of 4"
        );
        assert_eq!(opt.accumulation_step, 3);

        // Fourth micro-batch — should now fire.
        opt.accumulate_gradients(&grads);
        let result = opt
            .step_accumulated(&mut model, 4, 0)
            .expect("valid gradients");
        assert_eq!(result, Some(4), "should fire after 4 micro-batches");
    }

    #[test]
    fn accumulate_from_adds_elementwise() {
        let mut base = make_tiny_gradients(2, 1.0);
        let extra = make_tiny_gradients(2, 3.0);
        base.accumulate_from(&extra);

        // All elements should now be 4.0.
        for &v in base
            .position
            .iter()
            .chain(base.rotation.iter())
            .chain(base.scale.iter())
            .chain(base.opacity.iter())
            .chain(base.offset.iter())
        {
            assert!(
                (v - 4.0).abs() < 1e-6,
                "expected 4.0 after accumulate, got {v}"
            );
        }
    }

    #[test]
    fn gradient_scale_method() {
        let mut grads = make_tiny_gradients(2, 4.0);
        grads.scale(0.25);

        for &v in grads
            .position
            .iter()
            .chain(grads.rotation.iter())
            .chain(grads.scale.iter())
            .chain(grads.opacity.iter())
            .chain(grads.offset.iter())
        {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "expected 1.0 after scale by 0.25, got {v}"
            );
        }
    }

    // ---- LR scale + group-vector conversion --------------------------------

    #[test]
    fn lr_scale_multiplies_every_group_and_is_validated() {
        let model = make_tiny_model(2, 0);
        let config = OptimizerConfig::default();
        let mut optimizer = GaussianOptimizer::new(&config, &model);
        assert_eq!(optimizer.lr_scale(), 1.0);

        let base_position_lr = optimizer.position_lr(0);
        optimizer
            .set_lr_scale(0.5)
            .expect("0.5 is a valid multiplier");
        assert_eq!(optimizer.lr_scale(), 0.5);
        assert!((optimizer.position_lr(0) - base_position_lr * 0.5).abs() < 1e-12);

        // A halved schedule must move the parameters half as far.
        let mut grads = Gradients::zeros(model.len(), 3);
        grads.opacity.iter_mut().for_each(|g| *g = 1.0);

        let mut full = model.clone();
        let mut full_opt = GaussianOptimizer::new(&config, &model);
        full_opt
            .step(&mut full, &grads, 1)
            .expect("shapes match the model");

        let mut halved = model.clone();
        let mut half_opt = GaussianOptimizer::new(&config, &model);
        half_opt.set_lr_scale(0.5).expect("valid multiplier");
        half_opt
            .step(&mut halved, &grads, 1)
            .expect("shapes match the model");

        let full_delta = model.gaussians[0].opacity - full.gaussians[0].opacity;
        let half_delta = model.gaussians[0].opacity - halved.gaussians[0].opacity;
        assert!(full_delta.abs() > 0.0, "the full step must move opacity");
        assert!(
            (half_delta - full_delta * 0.5).abs() < 1e-6,
            "half scale should halve the step: {half_delta} vs {full_delta}"
        );

        // Non-finite / negative multipliers are refused, not silently applied.
        assert!(optimizer.set_lr_scale(f32::NAN).is_err());
        assert!(optimizer.set_lr_scale(-0.1).is_err());
        assert_eq!(optimizer.lr_scale(), 0.5, "a rejected scale must not stick");
    }

    #[test]
    fn gradients_round_trip_through_group_vectors() {
        let mut grads = Gradients::zeros(3, 4);
        for (i, g) in grads.position.iter_mut().enumerate() {
            *g = i as f32;
        }
        grads.opacity[1] = -2.5;

        let groups = grads.to_group_vecs();
        assert_eq!(groups.len(), 6);
        assert_eq!(
            groups.iter().map(Vec::len).collect::<Vec<_>>(),
            grads.group_sizes().to_vec()
        );

        let mut restored = Gradients::zeros(3, 4);
        restored
            .set_from_group_vecs(&groups)
            .expect("matching shapes round-trip");
        assert_eq!(restored.position, grads.position);
        assert_eq!(restored.opacity, grads.opacity);

        // A wrong group count or a wrong length is an error, never a silent
        // truncation that would drop the tail of a parameter group.
        assert!(restored.set_from_group_vecs(&groups[..5]).is_err());
        let mut short = groups.clone();
        short[0].pop();
        assert!(restored.set_from_group_vecs(&short).is_err());
    }
}
