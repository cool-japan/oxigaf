//! # Consistency Distillation Loss
//!
//! Implements consistency distillation and progressive distillation loss
//! computation as described in the Consistency Models paper (Song et al., 2023)
//! and the Progressive Distillation paper (Salimans & Ho, 2022).
//!
//! Consistency distillation trains a student model to predict the same
//! denoised output regardless of the noise level, enabling few-step
//! (even single-step) generation.
//!
//! ## What each [`DistillationMode`] compares
//!
//! All three modes start from the same inputs - a teacher/target prediction and
//! a student prediction at the noisy level `t`, plus `ᾱ_t` and `ᾱ_{t−n}` - but
//! they build genuinely different quantities:
//!
//! * [`DistillationMode::ConsistencyDistillation`] - both sides take one
//!   probability-flow (DDIM) step down to `t − n` and the loss is the resulting
//!   *trajectory discrepancy* `‖x̂ᵀ_{t−n} − x̂ˢ_{t−n}‖`. The teacher's step uses
//!   the teacher's ε, the student's step uses its own. This assumes both were
//!   evaluated on the *same* `x_t`, which is what distillation feeds them.
//! * [`DistillationMode::ConsistencyTraining`] - no teacher and no ODE solve:
//!   the student's `x₀` estimate at `t` is compared against the EMA target
//!   network's `x₀` estimate at the cleaner level `t − n` (self-consistency,
//!   loss weight `λ ≡ 1` as in the paper).
//! * [`DistillationMode::Progressive`] - the teacher's `stride =
//!   teacher_steps / student_steps` sub-steps are collapsed into the single
//!   `x₀` target that makes *one* student DDIM step land on the teacher's
//!   endpoint (the Salimans & Ho reparameterisation). Supply
//!   [`TeacherStudentPair::with_teacher_mid`] to get the true two-step teacher
//!   solve; without it the exact one-step solve is used.
//!
//! Every mode is zero exactly when the student agrees with the teacher - a
//! perfect student is never left with a residual floor.
//!
//! ## Scale of the consistency-distillation loss
//!
//! Sharing `x_t`, the CD discrepancy is `(α_s − (σ_s/σ_t)·α_t)·Δx₀`, and that
//! factor equals `α_s·(σ̃_t − σ̃_s)/σ̃_t` in EDM noise levels: the CD loss carries
//! a `(σ̃_t − σ̃_s)²` factor and therefore *shrinks quadratically as the two
//! timesteps get closer* (ᾱ 0.70 → 0.71 already scales it by ~4e-4). With
//! adjacent timesteps, compensate through
//! [`DistillationConfig::loss_weight`] - this is exactly what the
//! `1/(σ − σ_next)` weighting of [`crate::consistency_model::cm_loss_weight`]
//! exists for. It is not applied automatically here because it is unbounded and
//! would make the reported components incomparable across a mini-batch.
//!
//! Note also that the EDM preconditioning of [`crate::consistency_model`] is
//! deliberately *not* applied: `predicted_x0` already is a denoised estimate,
//! and applying `c_skip`/`c_out` at two different noise levels would leave the
//! loss with a floor that a perfect student could not reach.

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// DistillationMode
// ---------------------------------------------------------------------------

/// Type of distillation approach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillationMode {
    /// Progressive distillation: student matches teacher with 2x fewer steps.
    Progressive,
    /// Consistency Training (CT): self-consistency without teacher.
    ConsistencyTraining,
    /// Consistency Distillation (CD): student matches teacher's deterministic trajectory.
    ConsistencyDistillation,
}

// ---------------------------------------------------------------------------
// LatentDims
// ---------------------------------------------------------------------------

/// Spatial layout (`C × H × W`, channel-major) of the flat `x0` buffers.
///
/// Required by the LPIPS proxy term: a diffusion latent is a stack of planes,
/// not a square single-channel image, so the patch metric has to be evaluated
/// per channel plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentDims {
    /// Number of channel planes (e.g. 4 for a Stable-Diffusion-style latent).
    pub channels: usize,
    /// Plane height in latent pixels.
    pub height: usize,
    /// Plane width in latent pixels.
    pub width: usize,
}

impl LatentDims {
    /// Create a `channels × height × width` layout.
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
        }
    }

    /// Number of elements the layout describes (`channels * height * width`).
    pub fn len(&self) -> usize {
        self.channels
            .saturating_mul(self.height)
            .saturating_mul(self.width)
    }

    /// `true` when any dimension is zero, i.e. the layout describes no data.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// DistillationConfig
// ---------------------------------------------------------------------------

/// Configuration for consistency distillation.
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// The distillation mode (progressive, CT, or CD).
    pub mode: DistillationMode,
    /// Number of steps used by the original teacher model (default: 50).
    pub teacher_steps: usize,
    /// Target number of steps for fast inference by the student (default: 4).
    pub student_steps: usize,
    /// Weight for the overall distillation loss (default: 1.0).
    pub loss_weight: f32,
    /// EMA decay rate for maintaining a shadow copy of teacher weights (default: 0.999).
    pub ema_decay: f32,
    /// Weight for the consistency regularisation term (default: 0.1).
    pub consistency_loss_weight: f32,
    /// Weight for the perceptual (LPIPS proxy) loss component (default: 0.0 = disabled).
    pub lpips_proxy_weight: f32,
    /// Layout of the flat `x0` buffers (default: `None`).
    ///
    /// Must be supplied whenever `lpips_proxy_weight > 0`: the patch metric is
    /// undefined without knowing how the flat buffer maps onto channel planes.
    pub latent_dims: Option<LatentDims>,
}

impl DistillationConfig {
    /// Create a progressive distillation config where student uses half as many
    /// steps as the teacher.
    pub fn progressive(teacher_steps: usize) -> Self {
        let student_steps = (teacher_steps / 2).max(1);
        Self {
            mode: DistillationMode::Progressive,
            teacher_steps,
            student_steps,
            loss_weight: 1.0,
            ema_decay: 0.999,
            consistency_loss_weight: 0.1,
            lpips_proxy_weight: 0.0,
            latent_dims: None,
        }
    }

    /// Create a consistency distillation config with explicit step counts.
    pub fn consistency_distillation(teacher_steps: usize, student_steps: usize) -> Self {
        Self {
            mode: DistillationMode::ConsistencyDistillation,
            teacher_steps,
            student_steps,
            loss_weight: 1.0,
            ema_decay: 0.999,
            consistency_loss_weight: 0.1,
            lpips_proxy_weight: 0.0,
            latent_dims: None,
        }
    }

    /// Create a consistency *training* (teacher-free) config.
    pub fn consistency_training(teacher_steps: usize, student_steps: usize) -> Self {
        Self {
            mode: DistillationMode::ConsistencyTraining,
            ..Self::consistency_distillation(teacher_steps, student_steps)
        }
    }

    /// Enable the LPIPS proxy term with the given weight and latent layout.
    pub fn with_lpips_proxy(mut self, weight: f32, dims: LatentDims) -> Self {
        self.lpips_proxy_weight = weight;
        self.latent_dims = Some(dims);
        self
    }

    /// Validate that all configuration parameters are in legal ranges.
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if self.teacher_steps == 0 {
            return Err(DiffusionError::InvalidConfig(
                "teacher_steps must be > 0".to_string(),
            ));
        }
        if self.student_steps == 0 {
            return Err(DiffusionError::InvalidConfig(
                "student_steps must be > 0".to_string(),
            ));
        }
        if self.student_steps > self.teacher_steps {
            return Err(DiffusionError::InvalidConfig(format!(
                "student_steps ({}) must be <= teacher_steps ({})",
                self.student_steps, self.teacher_steps
            )));
        }
        if !(0.0..1.0).contains(&self.ema_decay) {
            return Err(DiffusionError::InvalidConfig(format!(
                "ema_decay ({}) must be in [0, 1)",
                self.ema_decay
            )));
        }
        if self.loss_weight < 0.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "loss_weight ({}) must be >= 0",
                self.loss_weight
            )));
        }
        if self.consistency_loss_weight < 0.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "consistency_loss_weight ({}) must be >= 0",
                self.consistency_loss_weight
            )));
        }
        if self.lpips_proxy_weight < 0.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "lpips_proxy_weight ({}) must be >= 0",
                self.lpips_proxy_weight
            )));
        }
        if let Some(dims) = self.latent_dims {
            if dims.is_empty() {
                return Err(DiffusionError::InvalidConfig(format!(
                    "latent_dims {}x{}x{} must have non-zero channels/height/width",
                    dims.channels, dims.height, dims.width
                )));
            }
        } else if self.lpips_proxy_weight > 0.0 {
            return Err(DiffusionError::InvalidConfig(
                "lpips_proxy_weight > 0 requires latent_dims to describe the x0 layout".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for DistillationConfig {
    /// Default: Consistency Distillation, 50 teacher steps → 4 student steps.
    fn default() -> Self {
        Self::consistency_distillation(50, 4)
    }
}

// ---------------------------------------------------------------------------
// NoisePrediction
// ---------------------------------------------------------------------------

/// A (noise, denoised) prediction pair from teacher or student.
#[derive(Debug, Clone)]
pub struct NoisePrediction {
    /// Diffusion timestep index.
    pub timestep: u32,
    /// Noisy input sample x_t.
    pub noisy_input: Vec<f32>,
    /// Raw network output: epsilon when `is_v_prediction` is `false`, the
    /// velocity `v` when it is `true`.
    pub predicted_noise: Vec<f32>,
    /// Predicted clean sample x_0.
    pub predicted_x0: Vec<f32>,
    /// Whether the network uses v-prediction parameterisation.
    ///
    /// Read by [`NoisePrediction::epsilon_at`], which every ODE solve in this
    /// module goes through, so a v-parameterised prediction is never treated as
    /// an epsilon one.
    pub is_v_prediction: bool,
}

impl NoisePrediction {
    /// Compute the clean sample x_0 from a noisy sample and a noise prediction
    /// using the epsilon-mode formula:
    ///
    ///   x_0 = (x_t - sqrt(1 - alpha_cumprod) * pred_noise) / sqrt(alpha_cumprod)
    ///
    /// Both `noisy_input` and `predicted_noise` must have the same length.
    /// If they differ in length the result is an empty `Vec`.
    pub fn compute_x0(
        noisy_input: &[f32],
        predicted_noise: &[f32],
        alpha_t: f32,
        sigma_t: f32,
    ) -> Vec<f32> {
        if noisy_input.len() != predicted_noise.len() {
            return Vec::new();
        }
        let inv_alpha = if alpha_t.abs() < f32::EPSILON {
            f32::NAN
        } else {
            1.0 / alpha_t
        };
        noisy_input
            .iter()
            .zip(predicted_noise.iter())
            .map(|(&x_t, &eps)| (x_t - sigma_t * eps) * inv_alpha)
            .collect()
    }

    /// Construct a `NoisePrediction` from an epsilon-parameterised prediction.
    ///
    /// `alpha_cumprod` is ᾱ_t (= α_t²). The square-root is taken internally:
    ///   - alpha_t  = sqrt(alpha_cumprod)
    ///   - sigma_t  = sqrt(1 - alpha_cumprod)
    pub fn from_epsilon(
        timestep: u32,
        noisy_input: Vec<f32>,
        predicted_noise: Vec<f32>,
        alpha_cumprod: f32,
    ) -> Self {
        let alpha_t = alpha_cumprod.sqrt();
        let sigma_t = (1.0 - alpha_cumprod).max(0.0).sqrt();
        let predicted_x0 = Self::compute_x0(&noisy_input, &predicted_noise, alpha_t, sigma_t);
        Self {
            timestep,
            noisy_input,
            predicted_noise,
            predicted_x0,
            is_v_prediction: false,
        }
    }

    /// Compute the clean sample x_0 from a v-parameterised network output:
    ///
    ///   x_0 = sqrt(alpha_cumprod) * x_t - sqrt(1 - alpha_cumprod) * v
    ///
    /// Returns an empty `Vec` when the slices differ in length.
    pub fn compute_x0_from_v(
        noisy_input: &[f32],
        predicted_v: &[f32],
        alpha_t: f32,
        sigma_t: f32,
    ) -> Vec<f32> {
        if noisy_input.len() != predicted_v.len() {
            return Vec::new();
        }
        noisy_input
            .iter()
            .zip(predicted_v.iter())
            .map(|(&x_t, &v)| alpha_t * x_t - sigma_t * v)
            .collect()
    }

    /// Construct a `NoisePrediction` from a v-parameterised prediction.
    ///
    /// `alpha_cumprod` is ᾱ_t; with `alpha_t = sqrt(ᾱ_t)` and
    /// `sigma_t = sqrt(1 - ᾱ_t)` the standard identities are
    ///
    ///   x_0 = alpha_t * x_t - sigma_t * v      and      eps = sigma_t * x_t + alpha_t * v
    ///
    /// `predicted_noise` keeps the raw `v` output and `is_v_prediction` is set,
    /// so [`Self::epsilon_at`] recovers epsilon on demand.
    pub fn from_v_prediction(
        timestep: u32,
        noisy_input: Vec<f32>,
        predicted_v: Vec<f32>,
        alpha_cumprod: f32,
    ) -> Self {
        let alpha_t = alpha_cumprod.clamp(0.0, 1.0).sqrt();
        let sigma_t = (1.0 - alpha_cumprod).clamp(0.0, 1.0).sqrt();
        let predicted_x0 = Self::compute_x0_from_v(&noisy_input, &predicted_v, alpha_t, sigma_t);
        Self {
            timestep,
            noisy_input,
            predicted_noise: predicted_v,
            predicted_x0,
            is_v_prediction: true,
        }
    }

    /// Epsilon prediction of this network output at noise level `alpha_cumprod`.
    ///
    /// For an epsilon-parameterised prediction this is `predicted_noise` itself;
    /// for a v-parameterised one it is `sigma_t * x_t + alpha_t * v`.
    ///
    /// Returns an empty `Vec` when a v-prediction has a `noisy_input` whose
    /// length does not match `predicted_noise`.
    pub fn epsilon_at(&self, alpha_cumprod: f32) -> Vec<f32> {
        if !self.is_v_prediction {
            return self.predicted_noise.clone();
        }
        if self.noisy_input.len() != self.predicted_noise.len() {
            return Vec::new();
        }
        let alpha_t = alpha_cumprod.clamp(0.0, 1.0).sqrt();
        let sigma_t = (1.0 - alpha_cumprod).clamp(0.0, 1.0).sqrt();
        self.noisy_input
            .iter()
            .zip(self.predicted_noise.iter())
            .map(|(&x_t, &v)| sigma_t * x_t + alpha_t * v)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TeacherStudentPair
// ---------------------------------------------------------------------------

/// Teacher-student prediction pair for one distillation step.
#[derive(Debug, Clone)]
pub struct TeacherStudentPair {
    /// Teacher model prediction at the current timestep.
    ///
    /// For [`DistillationMode::ConsistencyTraining`] this slot holds the EMA
    /// *target* copy of the student, evaluated at the cleaner level
    /// `alpha_cumprod_t_minus_n`; the other modes evaluate the teacher at
    /// `alpha_cumprod_t` on the same `x_t` as the student.
    pub teacher: NoisePrediction,
    /// Student model prediction at the current timestep.
    pub student: NoisePrediction,
    /// Cumulative product of alphas at the current (noisier) timestep.
    pub alpha_cumprod_t: f32,
    /// Cumulative product of alphas at the target (fewer-step / cleaner) timestep.
    ///
    /// ᾱ grows as the sample gets cleaner, so this must be `>= alpha_cumprod_t`.
    pub alpha_cumprod_t_minus_n: f32,
    /// Optional teacher evaluation at the intermediate level of a two-step
    /// solve, used by [`DistillationMode::Progressive`].
    ///
    /// It must be produced by running the teacher on
    /// [`Self::intermediate_sample`] at [`Self::intermediate_alpha_cumprod`].
    /// Without it, progressive distillation falls back to the exact one-step
    /// solve (which is what a frozen-epsilon teacher yields for any number of
    /// sub-steps).
    pub teacher_mid: Option<NoisePrediction>,
}

impl TeacherStudentPair {
    /// Create a pair without an intermediate teacher evaluation.
    pub fn new(
        teacher: NoisePrediction,
        student: NoisePrediction,
        alpha_cumprod_t: f32,
        alpha_cumprod_t_minus_n: f32,
    ) -> Self {
        Self {
            teacher,
            student,
            alpha_cumprod_t,
            alpha_cumprod_t_minus_n,
            teacher_mid: None,
        }
    }

    /// Attach the teacher's intermediate-step evaluation (see [`Self::teacher_mid`]).
    pub fn with_teacher_mid(mut self, teacher_mid: NoisePrediction) -> Self {
        self.teacher_mid = Some(teacher_mid);
        self
    }

    /// ᾱ at the log-SNR midpoint of `[t, t − n]`, i.e. the level at which
    /// [`Self::teacher_mid`] has to be evaluated.
    pub fn intermediate_alpha_cumprod(&self) -> f32 {
        midpoint_alpha_cumprod(self.alpha_cumprod_t, self.alpha_cumprod_t_minus_n)
    }

    /// The sample the teacher's first DDIM sub-step lands on, i.e. the input the
    /// teacher has to be run on to produce [`Self::teacher_mid`].
    ///
    /// Returns an empty `Vec` when the teacher's buffers are inconsistent.
    pub fn intermediate_sample(&self) -> Vec<f32> {
        let eps = self.teacher.epsilon_at(self.alpha_cumprod_t);
        ddim_sample_at(
            &self.teacher.predicted_x0,
            &eps,
            self.intermediate_alpha_cumprod(),
        )
    }
}

// ---------------------------------------------------------------------------
// Noise-schedule helpers
// ---------------------------------------------------------------------------

/// Smallest ᾱ used when converting to an EDM noise level (keeps σ finite).
const MIN_ALPHA_CUMPROD: f32 = 1e-8;

/// Smallest `sigma_t` / denominator treated as non-degenerate.
const MIN_DENOMINATOR: f32 = 1e-6;

/// Variance-preserving coefficients `(alpha, sigma) = (sqrt(ᾱ), sqrt(1 − ᾱ))`.
pub fn vp_coefficients(alpha_cumprod: f32) -> (f32, f32) {
    let a = alpha_cumprod.clamp(0.0, 1.0);
    (a.sqrt(), (1.0 - a).clamp(0.0, 1.0).sqrt())
}

/// EDM (variance-exploding) noise level `σ = sqrt((1 − ᾱ) / ᾱ)` of a
/// variance-preserving level ᾱ.
///
/// ᾱ is clamped away from zero so σ stays finite.
pub fn edm_sigma(alpha_cumprod: f32) -> f32 {
    let a = alpha_cumprod.clamp(MIN_ALPHA_CUMPROD, 1.0);
    ((1.0 - a) / a).max(0.0).sqrt()
}

/// ᾱ halfway (in log-SNR) between two noise levels.
///
/// `σ_mid = sqrt(σ_t · σ_s)` and `ᾱ_mid = 1 / (1 + σ_mid²)`, so the result
/// always lies between the two inputs.
pub fn midpoint_alpha_cumprod(alpha_cumprod_t: f32, alpha_cumprod_t_minus_n: f32) -> f32 {
    let sigma_t = edm_sigma(alpha_cumprod_t);
    let sigma_s = edm_sigma(alpha_cumprod_t_minus_n);
    let sigma_mid = (sigma_t * sigma_s).max(0.0).sqrt();
    1.0 / (1.0 + sigma_mid * sigma_mid)
}

/// One deterministic DDIM / probability-flow-ODE step:
///
///   `x_target = sqrt(ᾱ_target) * x0 + sqrt(1 − ᾱ_target) * eps`
///
/// Returns an empty `Vec` when `x0` and `eps` differ in length.
pub fn ddim_sample_at(x0: &[f32], eps: &[f32], alpha_cumprod_target: f32) -> Vec<f32> {
    if x0.len() != eps.len() {
        return Vec::new();
    }
    let (alpha, sigma) = vp_coefficients(alpha_cumprod_target);
    x0.iter()
        .zip(eps.iter())
        .map(|(&x, &e)| alpha * x + sigma * e)
        .collect()
}

// ---------------------------------------------------------------------------
// Free loss functions
// ---------------------------------------------------------------------------

/// Mean squared error between two equally sized slices.
///
/// Returns `f32::NAN` on a length mismatch, `0.0` for empty inputs.
fn l2_slice(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::NAN;
    }
    if a.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum();
    sum_sq / a.len() as f32
}

/// Compute the L2 (MSE) loss between the clean-sample predictions of the
/// teacher and student.
///
/// Returns `f32::NAN` when the prediction vectors have different lengths.
pub fn l2_distillation_loss(teacher: &NoisePrediction, student: &NoisePrediction) -> f32 {
    l2_slice(&teacher.predicted_x0, &student.predicted_x0)
}

/// Pseudo-LPIPS spatial-coherence proxy loss.
///
/// Divides the image (given as a flat slice in row-major order) into
/// non-overlapping `patch_size × patch_size` patches and computes the mean
/// per-patch L1 distance between `teacher_x0` and `student_x0`.
///
/// Returns `0.0` for identical inputs, `f32::NAN` when lengths mismatch or
/// when `patch_size` is 0.
///
/// # Degenerate geometry
///
/// When `width * height` does not equal the slice length, or when a patch is
/// larger than the image, this falls back to the plain mean L1 over the whole
/// slice - a different metric from the documented per-patch score. Multi-channel
/// latents should therefore go through [`lpips_proxy_loss_planar`], which
/// validates the layout instead of degrading silently.
pub fn lpips_proxy_loss(
    teacher_x0: &[f32],
    student_x0: &[f32],
    patch_size: usize,
    width: usize,
    height: usize,
) -> f32 {
    if teacher_x0.len() != student_x0.len() {
        return f32::NAN;
    }
    if patch_size == 0 {
        return f32::NAN;
    }
    if teacher_x0.is_empty() {
        return 0.0;
    }

    let expected_len = width * height;
    // If width/height don't match the slice length, fall back to whole-slice L1 mean.
    if expected_len == 0 || expected_len != teacher_x0.len() {
        // Degenerate case: return mean L1 over the whole slice.
        let sum: f32 = teacher_x0
            .iter()
            .zip(student_x0.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        return sum / teacher_x0.len() as f32;
    }

    let patches_x = width / patch_size;
    let patches_y = height / patch_size;

    if patches_x == 0 || patches_y == 0 {
        // Patches are larger than the image; fall back to whole-image L1.
        let sum: f32 = teacher_x0
            .iter()
            .zip(student_x0.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        return sum / teacher_x0.len() as f32;
    }

    let mut total_patch_loss: f32 = 0.0;
    let mut patch_count: usize = 0;

    for py in 0..patches_y {
        for px in 0..patches_x {
            let mut patch_sum: f32 = 0.0;
            let mut pixel_count: usize = 0;
            for dy in 0..patch_size {
                let row = py * patch_size + dy;
                for dx in 0..patch_size {
                    let col = px * patch_size + dx;
                    let idx = row * width + col;
                    if idx < teacher_x0.len() {
                        patch_sum += (teacher_x0[idx] - student_x0[idx]).abs();
                        pixel_count += 1;
                    }
                }
            }
            if pixel_count > 0 {
                total_patch_loss += patch_sum / pixel_count as f32;
                patch_count += 1;
            }
        }
    }

    if patch_count == 0 {
        return 0.0;
    }
    total_patch_loss / patch_count as f32
}

/// Pseudo-LPIPS proxy loss over a `C × H × W` latent.
///
/// The patch metric is evaluated on each channel plane separately and averaged,
/// so channels are never mixed into one bogus square "image".
///
/// # Errors
///
/// Returns an error when the two buffers differ in length, when `patch_size` is
/// zero, or when `dims` does not describe exactly `teacher_x0.len()` elements -
/// the layout is never guessed.
pub fn lpips_proxy_loss_planar(
    teacher_x0: &[f32],
    student_x0: &[f32],
    patch_size: usize,
    dims: LatentDims,
) -> Result<f32, DiffusionError> {
    if teacher_x0.len() != student_x0.len() {
        return Err(DiffusionError::ShapeMismatch {
            op: "lpips_proxy_loss_planar".to_string(),
            expected: vec![teacher_x0.len()],
            got: vec![student_x0.len()],
        });
    }
    if patch_size == 0 {
        return Err(DiffusionError::InvalidConfig(
            "lpips patch_size must be > 0".to_string(),
        ));
    }
    if dims.is_empty() || dims.len() != teacher_x0.len() {
        return Err(DiffusionError::ShapeMismatch {
            op: "lpips_proxy_loss_planar".to_string(),
            expected: vec![dims.channels, dims.height, dims.width],
            got: vec![teacher_x0.len()],
        });
    }

    let plane = dims.height * dims.width;
    let mut total: f32 = 0.0;
    for c in 0..dims.channels {
        let start = c * plane;
        let end = start + plane;
        total += lpips_proxy_loss(
            &teacher_x0[start..end],
            &student_x0[start..end],
            patch_size,
            dims.width,
            dims.height,
        );
    }
    Ok(total / dims.channels as f32)
}

/// Huber loss (smooth L1) between two prediction vectors.
///
/// For each element:
/// - If |diff| <= delta: 0.5 * diff² / delta
/// - Otherwise:          |diff| - 0.5 * delta
///
/// Returns the mean over all elements. Returns `f32::NAN` for different lengths.
pub fn huber_loss(teacher_x0: &[f32], student_x0: &[f32], delta: f32) -> f32 {
    if teacher_x0.len() != student_x0.len() {
        return f32::NAN;
    }
    if teacher_x0.is_empty() {
        return 0.0;
    }
    let sum: f32 = teacher_x0
        .iter()
        .zip(student_x0.iter())
        .map(|(&a, &b)| {
            let diff = (a - b).abs();
            if diff <= delta {
                0.5 * diff * diff / delta.max(f32::EPSILON)
            } else {
                diff - 0.5 * delta
            }
        })
        .sum();
    sum / teacher_x0.len() as f32
}

// ---------------------------------------------------------------------------
// DistillationLossResult
// ---------------------------------------------------------------------------

/// Result of a full distillation loss computation.
#[derive(Debug, Clone)]
pub struct DistillationLossResult {
    /// Weighted sum of all loss components.
    pub total_loss: f32,
    /// Raw L2 (MSE) loss between the mode's target and the student quantity.
    pub l2_loss: f32,
    /// Raw Huber loss between the mode's target and the student quantity.
    pub huber_loss: f32,
    /// Raw pseudo-LPIPS proxy loss (0 when disabled).
    pub lpips_proxy: f32,
    /// Consistency weight applied to the Huber term.
    pub consistency_weight: f32,
    /// Distillation mode the components were computed for.
    pub mode: DistillationMode,
}

impl DistillationLossResult {
    /// Human-readable summary of the loss breakdown.
    pub fn format(&self) -> String {
        format!(
            "DistillationLossResult {{ mode={:?}, total={:.6}, l2={:.6}, huber={:.6}, lpips_proxy={:.6}, consistency_weight={:.6} }}",
            self.mode,
            self.total_loss,
            self.l2_loss,
            self.huber_loss,
            self.lpips_proxy,
            self.consistency_weight,
        )
    }

    /// Returns `true` when all loss components are finite (not NaN or Inf).
    pub fn is_finite(&self) -> bool {
        self.total_loss.is_finite()
            && self.l2_loss.is_finite()
            && self.huber_loss.is_finite()
            && self.lpips_proxy.is_finite()
            && self.consistency_weight.is_finite()
    }
}

// ---------------------------------------------------------------------------
// compute_distillation_loss
// ---------------------------------------------------------------------------

const HUBER_DELTA: f32 = 1.0;
const LPIPS_PATCH_SIZE: usize = 8;

/// Validate that a cumulative alpha product is a usable noise level.
fn check_alpha_cumprod(name: &str, alpha_cumprod: f32) -> Result<(), DiffusionError> {
    if !alpha_cumprod.is_finite() || !(0.0..=1.0).contains(&alpha_cumprod) {
        return Err(DiffusionError::InvalidConfig(format!(
            "{name} ({alpha_cumprod}) must be a finite value in [0, 1]"
        )));
    }
    Ok(())
}

/// Fetch the epsilon prediction of `pred` at `alpha_cumprod`, checking its length.
fn epsilon_checked(
    pred: &NoisePrediction,
    alpha_cumprod: f32,
    expected_len: usize,
    what: &str,
) -> Result<Vec<f32>, DiffusionError> {
    let eps = pred.epsilon_at(alpha_cumprod);
    if eps.len() != expected_len {
        return Err(DiffusionError::ShapeMismatch {
            op: format!("compute_distillation_loss/{what}"),
            expected: vec![expected_len],
            got: vec![eps.len()],
        });
    }
    Ok(eps)
}

/// Consistency Distillation: compare the two probability-flow-ODE endpoints at
/// `t − n`, one solved with the teacher's epsilon and one with the student's.
///
/// # Precondition
///
/// Teacher and student must have been evaluated on the same `x_t` (guaranteed by
/// the distillation loop that produces the pair). Under that precondition the
/// difference reduces to `(alpha_s − sigma_s * alpha_t / sigma_t) * (x0_s − x0_t)`,
/// so it vanishes exactly when the student matches the teacher - and it shrinks
/// with the gap between the two noise levels (see the module docs). The buffers
/// are not compared elementwise: doing so would cost a full pass per call and
/// reject legitimate rounding differences.
fn consistency_distillation_pair(
    pair: &TeacherStudentPair,
) -> Result<(Vec<f32>, Vec<f32>), DiffusionError> {
    let n = pair.teacher.predicted_x0.len();
    let eps_teacher = epsilon_checked(&pair.teacher, pair.alpha_cumprod_t, n, "teacher_epsilon")?;
    let eps_student = epsilon_checked(&pair.student, pair.alpha_cumprod_t, n, "student_epsilon")?;

    let target = ddim_sample_at(
        &pair.teacher.predicted_x0,
        &eps_teacher,
        pair.alpha_cumprod_t_minus_n,
    );
    let prediction = ddim_sample_at(
        &pair.student.predicted_x0,
        &eps_student,
        pair.alpha_cumprod_t_minus_n,
    );
    Ok((target, prediction))
}

/// Consistency Training: no teacher and no ODE solve - the student's `x0`
/// estimate at `t` must match the EMA target network's estimate at `t − n`.
fn consistency_training_pair(
    pair: &TeacherStudentPair,
) -> Result<(Vec<f32>, Vec<f32>), DiffusionError> {
    Ok((
        pair.teacher.predicted_x0.clone(),
        pair.student.predicted_x0.clone(),
    ))
}

/// Progressive distillation: collapse the teacher's sub-steps into the single
/// `x0` target that makes one student DDIM step land on the teacher's endpoint.
fn progressive_pair(
    pair: &TeacherStudentPair,
    config: &DistillationConfig,
) -> Result<(Vec<f32>, Vec<f32>), DiffusionError> {
    let n = pair.teacher.predicted_x0.len();
    if pair.teacher.noisy_input.len() != n {
        return Err(DiffusionError::ShapeMismatch {
            op: "compute_distillation_loss/teacher_noisy_input".to_string(),
            expected: vec![n],
            got: vec![pair.teacher.noisy_input.len()],
        });
    }
    let eps_teacher = epsilon_checked(&pair.teacher, pair.alpha_cumprod_t, n, "teacher_epsilon")?;

    // How many teacher sub-steps one student step has to cover.
    let stride = config.teacher_steps / config.student_steps.max(1);

    let x_target = match (&pair.teacher_mid, stride >= 2) {
        (Some(mid), true) => {
            // True two-step solve: the second step starts from the teacher's
            // evaluation at the intermediate level.
            let alpha_mid = pair.intermediate_alpha_cumprod();
            if mid.predicted_x0.len() != n {
                return Err(DiffusionError::ShapeMismatch {
                    op: "compute_distillation_loss/teacher_mid_x0".to_string(),
                    expected: vec![n],
                    got: vec![mid.predicted_x0.len()],
                });
            }
            let eps_mid = epsilon_checked(mid, alpha_mid, n, "teacher_mid_epsilon")?;
            ddim_sample_at(&mid.predicted_x0, &eps_mid, pair.alpha_cumprod_t_minus_n)
        }
        _ => ddim_sample_at(
            &pair.teacher.predicted_x0,
            &eps_teacher,
            pair.alpha_cumprod_t_minus_n,
        ),
    };

    // Salimans & Ho reparameterisation, with r = sigma_s / sigma_t:
    //   x0_target = (x_{t−n} − r * x_t) / (alpha_s − r * alpha_t)
    let (alpha_t, sigma_t) = vp_coefficients(pair.alpha_cumprod_t);
    let (alpha_s, sigma_s) = vp_coefficients(pair.alpha_cumprod_t_minus_n);
    let ratio = if sigma_t <= MIN_DENOMINATOR {
        0.0
    } else {
        sigma_s / sigma_t
    };
    let denom = alpha_s - ratio * alpha_t;

    let target = if sigma_t <= MIN_DENOMINATOR || denom.abs() <= MIN_DENOMINATOR {
        // Degenerate geometry (x_t carries no noise, or the two levels have the
        // same signal-to-noise ratio): the teacher's own x0 estimate is the
        // only well-defined target.
        pair.teacher.predicted_x0.clone()
    } else {
        x_target
            .iter()
            .zip(pair.teacher.noisy_input.iter())
            .map(|(&x_s, &x_t)| (x_s - ratio * x_t) / denom)
            .collect()
    };

    Ok((target, pair.student.predicted_x0.clone()))
}

/// Compute the full distillation loss for a single teacher-student pair.
///
/// The comparison depends on [`DistillationConfig::mode`] (see the module docs);
/// the weighted combination of the resulting components is always
///
///   total = loss_weight * l2 + consistency_loss_weight * huber + lpips_proxy_weight * lpips
///
/// # Errors
///
/// Returns an error when the teacher and student buffers disagree in length,
/// when a mode needs an epsilon/`x_t` buffer that is missing or mis-sized, when
/// the noise levels are outside `[0, 1]` or ordered the wrong way round, or when
/// the LPIPS proxy is enabled without a matching [`DistillationConfig::latent_dims`].
pub fn compute_distillation_loss(
    pair: &TeacherStudentPair,
    config: &DistillationConfig,
) -> Result<DistillationLossResult, DiffusionError> {
    let n = pair.teacher.predicted_x0.len();
    if pair.student.predicted_x0.len() != n {
        return Err(DiffusionError::ShapeMismatch {
            op: "compute_distillation_loss/predicted_x0".to_string(),
            expected: vec![n],
            got: vec![pair.student.predicted_x0.len()],
        });
    }
    check_alpha_cumprod("alpha_cumprod_t", pair.alpha_cumprod_t)?;
    check_alpha_cumprod("alpha_cumprod_t_minus_n", pair.alpha_cumprod_t_minus_n)?;
    if pair.alpha_cumprod_t_minus_n < pair.alpha_cumprod_t {
        return Err(DiffusionError::InvalidConfig(format!(
            "alpha_cumprod_t_minus_n ({}) must be >= alpha_cumprod_t ({}): the cumulative alpha \
             product grows as the sample gets cleaner",
            pair.alpha_cumprod_t_minus_n, pair.alpha_cumprod_t
        )));
    }

    // Mode-specific (target, student) quantities.
    let (target, prediction) = match config.mode {
        DistillationMode::Progressive => progressive_pair(pair, config)?,
        DistillationMode::ConsistencyTraining => consistency_training_pair(pair)?,
        DistillationMode::ConsistencyDistillation => consistency_distillation_pair(pair)?,
    };

    let l2 = l2_slice(&target, &prediction);
    let huber = huber_loss(&target, &prediction, HUBER_DELTA);

    let lpips = if config.lpips_proxy_weight > 0.0 && !target.is_empty() {
        let dims = config.latent_dims.ok_or_else(|| {
            DiffusionError::InvalidConfig(
                "lpips_proxy_weight > 0 requires DistillationConfig::latent_dims describing the \
                 channel/height/width layout of the x0 buffers"
                    .to_string(),
            )
        })?;
        lpips_proxy_loss_planar(&target, &prediction, LPIPS_PATCH_SIZE, dims)?
    } else {
        0.0
    };

    let l2_finite = if l2.is_finite() { l2 } else { 0.0 };
    let huber_finite = if huber.is_finite() { huber } else { 0.0 };
    let lpips_finite = if lpips.is_finite() { lpips } else { 0.0 };

    let total = config.loss_weight * l2_finite
        + config.consistency_loss_weight * huber_finite
        + config.lpips_proxy_weight * lpips_finite;

    Ok(DistillationLossResult {
        total_loss: total,
        l2_loss: l2,
        huber_loss: huber,
        lpips_proxy: lpips,
        consistency_weight: config.consistency_loss_weight,
        mode: config.mode,
    })
}

// ---------------------------------------------------------------------------
// EmaTeacher
// ---------------------------------------------------------------------------

/// Exponential moving average shadow copy of the teacher model weights.
///
/// The EMA is updated after each student optimiser step so the teacher
/// parameters lag behind the student at a controlled rate.
#[derive(Debug, Clone)]
pub struct EmaTeacher {
    /// Configured EMA decay rate.
    pub decay: f32,
    /// Number of completed update steps.
    step: u64,
    /// Shadow weight tensors, indexed as [param_idx][value_idx].
    weights: Vec<Vec<f32>>,
}

impl EmaTeacher {
    /// Initialise the EMA teacher with `num_params` *empty* shadow tensors.
    ///
    /// The shadow tensors carry no values until the first [`Self::update`],
    /// which sizes each of them from the corresponding student tensor (and
    /// fills it with zeros first). Use [`Self::new_with_shapes`] when the shadow
    /// weights have to be inspectable before training starts.
    pub fn new(decay: f32, num_params: usize) -> Self {
        Self {
            decay,
            step: 0,
            weights: vec![Vec::new(); num_params],
        }
    }

    /// Initialise the EMA teacher with one all-zero shadow tensor per entry of
    /// `shapes`, where each entry is that tensor's element count.
    pub fn new_with_shapes(decay: f32, shapes: &[usize]) -> Self {
        Self {
            decay,
            step: 0,
            weights: shapes.iter().map(|&len| vec![0.0_f32; len]).collect(),
        }
    }

    /// Apply one EMA update step using the current student weights.
    ///
    /// The bias-corrected decay `d = effective_decay()` is used:
    ///   ema_w\[i\] = d * ema_w\[i\] + (1 - d) * student_w\[i\]
    ///
    /// If the shadow weight vector for a parameter is shorter than the student
    /// vector (e.g., on the first update), it is extended with zeros first.
    ///
    /// Returns `Err` when the number of parameter tensors does not match.
    pub fn update(&mut self, student_weights: &[Vec<f32>]) -> Result<(), DiffusionError> {
        if student_weights.len() != self.weights.len() {
            return Err(DiffusionError::InvalidConfig(format!(
                "EmaTeacher weight count mismatch: expected {}, got {}",
                self.weights.len(),
                student_weights.len()
            )));
        }
        let d = self.effective_decay();
        let one_minus_d = 1.0 - d;
        for (ema_w, student_w) in self.weights.iter_mut().zip(student_weights.iter()) {
            // Extend shadow if this is the first real update.
            if ema_w.len() < student_w.len() {
                ema_w.resize(student_w.len(), 0.0);
            } else if ema_w.len() != student_w.len() {
                return Err(DiffusionError::InvalidConfig(format!(
                    "EmaTeacher inner weight length mismatch: shadow has {}, student has {}",
                    ema_w.len(),
                    student_w.len()
                )));
            }
            for (e, &s) in ema_w.iter_mut().zip(student_w.iter()) {
                *e = d * (*e) + one_minus_d * s;
            }
        }
        self.step += 1;
        Ok(())
    }

    /// Return a reference to the shadow weight tensors.
    pub fn weights(&self) -> &[Vec<f32>] {
        &self.weights
    }

    /// Bias-corrected effective decay:
    ///   min(decay, (1 + step) / (10 + step))
    pub fn effective_decay(&self) -> f32 {
        let correction = (1 + self.step) as f32 / (10 + self.step) as f32;
        self.decay.min(correction)
    }

    /// Return the number of completed update steps.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Reset the shadow weights to the provided tensors and zero the step
    /// counter.
    pub fn reset(&mut self, new_weights: Vec<Vec<f32>>) {
        self.weights = new_weights;
        self.step = 0;
    }
}

// ---------------------------------------------------------------------------
// DistillationStep
// ---------------------------------------------------------------------------

/// Represents one full distillation training iteration, potentially covering
/// multiple teacher-student pairs within a mini-batch.
#[derive(Debug, Clone)]
pub struct DistillationStep {
    /// Training iteration index.
    pub iteration: u32,
    /// All teacher-student prediction pairs in this batch.
    pub teacher_student_pairs: Vec<TeacherStudentPair>,
    /// Aggregated loss result across all pairs.
    pub loss_result: DistillationLossResult,
}

impl DistillationStep {
    /// Compute and aggregate losses across all pairs, returning a mean result.
    ///
    /// If `pairs` is empty, returns a zeroed `DistillationLossResult`.
    ///
    /// # Errors
    ///
    /// Propagates the first error reported by [`compute_distillation_loss`].
    pub fn aggregate_losses(
        pairs: &[TeacherStudentPair],
        config: &DistillationConfig,
    ) -> Result<DistillationLossResult, DiffusionError> {
        if pairs.is_empty() {
            return Ok(DistillationLossResult {
                total_loss: 0.0,
                l2_loss: 0.0,
                huber_loss: 0.0,
                lpips_proxy: 0.0,
                consistency_weight: config.consistency_loss_weight,
                mode: config.mode,
            });
        }

        let n = pairs.len() as f32;
        let mut sum_total: f32 = 0.0;
        let mut sum_l2: f32 = 0.0;
        let mut sum_huber: f32 = 0.0;
        let mut sum_lpips: f32 = 0.0;

        for pair in pairs {
            let r = compute_distillation_loss(pair, config)?;
            sum_total += r.total_loss;
            sum_l2 += r.l2_loss;
            sum_huber += r.huber_loss;
            sum_lpips += r.lpips_proxy;
        }

        Ok(DistillationLossResult {
            total_loss: sum_total / n,
            l2_loss: sum_l2 / n,
            huber_loss: sum_huber / n,
            lpips_proxy: sum_lpips / n,
            consistency_weight: config.consistency_loss_weight,
            mode: config.mode,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper factories
    // -----------------------------------------------------------------------

    fn make_prediction(timestep: u32, len: usize, value: f32, noise_value: f32) -> NoisePrediction {
        NoisePrediction {
            timestep,
            noisy_input: vec![1.0; len],
            predicted_noise: vec![noise_value; len],
            predicted_x0: vec![value; len],
            is_v_prediction: false,
        }
    }

    /// ᾱ grows as the sample gets cleaner, hence `t` < `t − n`.
    const FIXTURE_ALPHA_T: f32 = 0.7;
    const FIXTURE_ALPHA_S: f32 = 0.9;

    /// Build a *self-consistent* prediction at `FIXTURE_ALPHA_T`: teacher and
    /// student share the noisy sample `x_t = 1.0` and reach the requested `x0`
    /// through their own epsilon, exactly like a real network evaluation.
    fn make_consistent_prediction(timestep: u32, len: usize, x0_value: f32) -> NoisePrediction {
        let (alpha_t, sigma_t) = vp_coefficients(FIXTURE_ALPHA_T);
        let eps = (1.0 - alpha_t * x0_value) / sigma_t;
        NoisePrediction::from_epsilon(timestep, vec![1.0; len], vec![eps; len], FIXTURE_ALPHA_T)
    }

    fn make_pair(len: usize, teacher_val: f32, student_val: f32) -> TeacherStudentPair {
        TeacherStudentPair::new(
            make_consistent_prediction(10, len, teacher_val),
            make_consistent_prediction(10, len, student_val),
            FIXTURE_ALPHA_T,
            FIXTURE_ALPHA_S,
        )
    }

    // -----------------------------------------------------------------------
    // DistillationConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_progressive() {
        let cfg = DistillationConfig::progressive(50);
        assert_eq!(cfg.mode, DistillationMode::Progressive);
        assert_eq!(cfg.teacher_steps, 50);
        assert_eq!(cfg.student_steps, 25);
        assert!((cfg.loss_weight - 1.0).abs() < f32::EPSILON);
        assert!((cfg.ema_decay - 0.999).abs() < 1e-5);
    }

    #[test]
    fn test_config_consistency_distillation() {
        let cfg = DistillationConfig::consistency_distillation(50, 4);
        assert_eq!(cfg.mode, DistillationMode::ConsistencyDistillation);
        assert_eq!(cfg.teacher_steps, 50);
        assert_eq!(cfg.student_steps, 4);
    }

    #[test]
    fn test_config_default() {
        let cfg = DistillationConfig::default();
        assert_eq!(cfg.mode, DistillationMode::ConsistencyDistillation);
        assert_eq!(cfg.teacher_steps, 50);
        assert_eq!(cfg.student_steps, 4);
    }

    #[test]
    fn test_config_validate_valid() {
        let cfg = DistillationConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate() {
        // teacher_steps == 0
        let cfg = DistillationConfig {
            teacher_steps: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        // student_steps > teacher_steps
        let cfg = DistillationConfig {
            student_steps: 100,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        // ema_decay >= 1.0
        let cfg = DistillationConfig {
            ema_decay: 1.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        // negative loss_weight
        let cfg = DistillationConfig {
            loss_weight: -0.5,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // NoisePrediction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_noise_prediction_compute_x0() {
        // alpha_cumprod = 0.64 → alpha_t = 0.8, sigma_t = 0.6
        let alpha_cumprod: f32 = 0.64;
        let alpha_t = alpha_cumprod.sqrt(); // 0.8
        let sigma_t = (1.0 - alpha_cumprod).sqrt(); // 0.6
        let x_t = vec![1.0_f32; 4];
        let eps = vec![0.5_f32; 4];
        let x0 = NoisePrediction::compute_x0(&x_t, &eps, alpha_t, sigma_t);
        // x0 = (1.0 - 0.6 * 0.5) / 0.8 = (1.0 - 0.3) / 0.8 = 0.7 / 0.8 = 0.875
        assert_eq!(x0.len(), 4);
        for &v in &x0 {
            assert!((v - 0.875).abs() < 1e-5, "expected 0.875, got {v}");
        }
    }

    #[test]
    fn test_noise_prediction_compute_x0_length_mismatch() {
        let x0 = NoisePrediction::compute_x0(&[1.0, 2.0], &[1.0], 0.8, 0.6);
        assert!(x0.is_empty());
    }

    #[test]
    fn test_noise_prediction_from_epsilon() {
        let alpha_cumprod: f32 = 0.64; // alpha_t = 0.8, sigma_t = 0.6
        let noisy = vec![1.0_f32; 4];
        let eps = vec![0.5_f32; 4];
        let pred = NoisePrediction::from_epsilon(5, noisy.clone(), eps, alpha_cumprod);
        assert_eq!(pred.timestep, 5);
        assert!(!pred.is_v_prediction);
        // expected x0 = 0.875
        for &v in &pred.predicted_x0 {
            assert!((v - 0.875).abs() < 1e-5, "expected 0.875, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // L2 loss tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_l2_loss_identical() {
        let p = make_prediction(0, 8, 0.5, 0.0);
        let loss = l2_distillation_loss(&p, &p);
        assert!(loss.abs() < f32::EPSILON, "expected 0, got {loss}");
    }

    #[test]
    fn test_l2_loss_simple() {
        let teacher = make_prediction(0, 4, 1.0, 0.0);
        let student = make_prediction(0, 4, 0.0, 0.0);
        // MSE = mean((1-0)^2) = 1.0
        let loss = l2_distillation_loss(&teacher, &student);
        assert!((loss - 1.0).abs() < 1e-5, "expected 1.0, got {loss}");
    }

    #[test]
    fn test_l2_loss_different_lengths() {
        let teacher = make_prediction(0, 4, 1.0, 0.0);
        let mut student = make_prediction(0, 5, 0.0, 0.0);
        student.predicted_x0 = vec![0.0; 5];
        let loss = l2_distillation_loss(&teacher, &student);
        assert!(loss.is_nan(), "expected NAN, got {loss}");
    }

    // -----------------------------------------------------------------------
    // Huber loss tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_huber_loss_small_diff() {
        // |diff| = 0.5 <= delta = 1.0 → 0.5 * 0.5^2 / 1.0 = 0.125
        let teacher = vec![1.0_f32; 4];
        let student = vec![0.5_f32; 4];
        let loss = huber_loss(&teacher, &student, 1.0);
        assert!((loss - 0.125).abs() < 1e-5, "expected 0.125, got {loss}");
    }

    #[test]
    fn test_huber_loss_large_diff() {
        // |diff| = 3.0 > delta = 1.0 → 3.0 - 0.5 * 1.0 = 2.5
        let teacher = vec![4.0_f32; 4];
        let student = vec![1.0_f32; 4];
        let loss = huber_loss(&teacher, &student, 1.0);
        assert!((loss - 2.5).abs() < 1e-5, "expected 2.5, got {loss}");
    }

    // -----------------------------------------------------------------------
    // LPIPS proxy loss tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lpips_proxy_loss_identical() {
        let img = vec![0.5_f32; 64]; // 8×8 image
        let loss = lpips_proxy_loss(&img, &img, 4, 8, 8);
        assert!(loss.abs() < f32::EPSILON, "expected 0, got {loss}");
    }

    #[test]
    fn test_lpips_proxy_loss_different() {
        let teacher = vec![1.0_f32; 64];
        let student = vec![0.0_f32; 64];
        let loss = lpips_proxy_loss(&teacher, &student, 4, 8, 8);
        // Every pixel difference = 1.0 → mean patch L1 = 1.0
        assert!((loss - 1.0).abs() < 1e-5, "expected 1.0, got {loss}");
    }

    #[test]
    fn test_lpips_proxy_loss_length_mismatch() {
        let teacher = vec![1.0_f32; 64];
        let student = vec![0.0_f32; 32];
        let loss = lpips_proxy_loss(&teacher, &student, 4, 8, 8);
        assert!(loss.is_nan());
    }

    // -----------------------------------------------------------------------
    // compute_distillation_loss tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_distillation_loss() {
        let cfg = DistillationConfig::default();
        let pair = make_pair(16, 1.0, 0.0);
        let result = compute_distillation_loss(&pair, &cfg).expect("loss should be computable");
        assert_eq!(result.mode, DistillationMode::ConsistencyDistillation);
        assert!(result.total_loss > 0.0);
        assert!(result.l2_loss > 0.0);
        assert!(result.is_finite());
    }

    /// Every mode must read `config.mode` and produce its own quantity.
    #[test]
    fn test_compute_distillation_loss_honours_mode() {
        let pair = make_pair(16, 1.0, 0.0);

        let cfg_cd = DistillationConfig::consistency_distillation(50, 4);
        let cfg_ct = DistillationConfig::consistency_training(50, 4);
        let cfg_pd = DistillationConfig::progressive(50);

        let cd = compute_distillation_loss(&pair, &cfg_cd).expect("cd loss");
        let ct = compute_distillation_loss(&pair, &cfg_ct).expect("ct loss");
        let pd = compute_distillation_loss(&pair, &cfg_pd).expect("pd loss");

        assert_eq!(cd.mode, DistillationMode::ConsistencyDistillation);
        assert_eq!(ct.mode, DistillationMode::ConsistencyTraining);
        assert_eq!(pd.mode, DistillationMode::Progressive);

        // CT compares x0 directly: MSE of a constant 1.0 offset.
        assert!((ct.l2_loss - 1.0).abs() < 1e-4, "ct l2 {}", ct.l2_loss);

        // A single-step progressive target is the teacher's own x0, so it must
        // agree with the plain x0 distance here.
        assert!((pd.l2_loss - 1.0).abs() < 1e-4, "pd l2 {}", pd.l2_loss);

        // CD compares the two ODE endpoints at t-n; sharing x_t, their gap is
        // (alpha_s - sigma_s * alpha_t / sigma_t) * delta_x0.
        let (alpha_t, sigma_t) = vp_coefficients(FIXTURE_ALPHA_T);
        let (alpha_s, sigma_s) = vp_coefficients(FIXTURE_ALPHA_S);
        let gap = alpha_s - (sigma_s / sigma_t) * alpha_t;
        let expected_cd = gap * gap;
        assert!(
            (cd.l2_loss - expected_cd).abs() < 1e-4,
            "cd l2 {} vs expected {expected_cd}",
            cd.l2_loss
        );

        // CD must not collapse onto the mode-agnostic x0 distance.
        assert!((cd.l2_loss - ct.l2_loss).abs() > 1e-3);
    }

    /// A perfect student must score exactly zero in every mode.
    #[test]
    fn test_zero_loss_at_optimum_for_every_mode() {
        for mode in [
            DistillationMode::Progressive,
            DistillationMode::ConsistencyTraining,
            DistillationMode::ConsistencyDistillation,
        ] {
            let cfg = DistillationConfig {
                mode,
                ..Default::default()
            };
            let pair = make_pair(16, 0.75, 0.75);
            let r = compute_distillation_loss(&pair, &cfg).expect("loss should be computable");
            assert!(
                r.l2_loss.abs() < 1e-6,
                "{mode:?} l2 should vanish at the optimum, got {}",
                r.l2_loss
            );
            assert!(
                r.total_loss.abs() < 1e-6,
                "{mode:?} total should vanish at the optimum, got {}",
                r.total_loss
            );
        }
    }

    /// Progressive distillation with a genuine two-step teacher solve must
    /// differ from the single-step fallback.
    #[test]
    fn test_progressive_two_step_teacher_target() {
        let cfg = DistillationConfig::progressive(8); // stride = 8 / 4 = 2
        let base = make_pair(4, 1.0, 0.0);

        let single = compute_distillation_loss(&base, &cfg).expect("single-step loss");

        // Teacher's intermediate evaluation: same x_mid, but a different x0.
        let alpha_mid = base.intermediate_alpha_cumprod();
        assert!(alpha_mid > base.alpha_cumprod_t && alpha_mid < base.alpha_cumprod_t_minus_n);

        let x_mid = base.intermediate_sample();
        assert_eq!(x_mid.len(), 4);

        let mid = NoisePrediction::from_epsilon(5, x_mid, vec![0.35_f32; 4], alpha_mid);
        let two_step = compute_distillation_loss(&base.clone().with_teacher_mid(mid), &cfg)
            .expect("two-step loss");

        assert!(two_step.is_finite());
        assert!(
            (two_step.l2_loss - single.l2_loss).abs() > 1e-4,
            "two-step teacher target must differ from the one-step solve ({} vs {})",
            two_step.l2_loss,
            single.l2_loss
        );
    }

    #[test]
    fn test_compute_distillation_loss_rejects_bad_inputs() {
        let cfg = DistillationConfig::default();

        // Mismatched x0 lengths.
        let mut pair = make_pair(8, 1.0, 0.0);
        pair.student.predicted_x0 = vec![0.0; 4];
        assert!(compute_distillation_loss(&pair, &cfg).is_err());

        // Inverted noise levels.
        let mut pair = make_pair(8, 1.0, 0.0);
        pair.alpha_cumprod_t = 0.9;
        pair.alpha_cumprod_t_minus_n = 0.7;
        assert!(compute_distillation_loss(&pair, &cfg).is_err());

        // Out-of-range noise level.
        let mut pair = make_pair(8, 1.0, 0.0);
        pair.alpha_cumprod_t = -0.1;
        assert!(compute_distillation_loss(&pair, &cfg).is_err());

        // Missing epsilon buffer for a mode that needs an ODE solve.
        let mut pair = make_pair(8, 1.0, 0.0);
        pair.teacher.predicted_noise = Vec::new();
        assert!(compute_distillation_loss(&pair, &cfg).is_err());
    }

    /// LPIPS must not guess a square single-channel geometry any more.
    #[test]
    fn test_lpips_requires_latent_dims() {
        let pair = make_pair(4 * 32 * 32, 1.0, 0.0);

        let cfg_without = DistillationConfig {
            lpips_proxy_weight: 0.5,
            ..Default::default()
        };
        assert!(compute_distillation_loss(&pair, &cfg_without).is_err());
        assert!(cfg_without.validate().is_err());

        let cfg_with =
            DistillationConfig::default().with_lpips_proxy(0.5, LatentDims::new(4, 32, 32));
        assert!(cfg_with.validate().is_ok());
        let r = compute_distillation_loss(&pair, &cfg_with).expect("planar lpips");
        assert!(r.lpips_proxy > 0.0);
        assert!(r.is_finite());

        // A layout that does not match the buffer is an error, not a fallback.
        let cfg_wrong =
            DistillationConfig::default().with_lpips_proxy(0.5, LatentDims::new(4, 16, 16));
        assert!(compute_distillation_loss(&pair, &cfg_wrong).is_err());
    }

    // -----------------------------------------------------------------------
    // DistillationLossResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_loss_result_is_finite() {
        let r = DistillationLossResult {
            total_loss: 1.0,
            l2_loss: 0.5,
            huber_loss: 0.3,
            lpips_proxy: 0.0,
            consistency_weight: 0.1,
            mode: DistillationMode::ConsistencyDistillation,
        };
        assert!(r.is_finite());

        let r_nan = DistillationLossResult {
            total_loss: f32::NAN,
            l2_loss: 0.5,
            huber_loss: 0.3,
            lpips_proxy: 0.0,
            consistency_weight: 0.1,
            mode: DistillationMode::ConsistencyDistillation,
        };
        assert!(!r_nan.is_finite());
    }

    #[test]
    fn test_loss_result_format() {
        let r = DistillationLossResult {
            total_loss: 1.23456,
            l2_loss: 0.5,
            huber_loss: 0.3,
            lpips_proxy: 0.0,
            consistency_weight: 0.1,
            mode: DistillationMode::Progressive,
        };
        let s = r.format();
        assert!(s.contains("total="));
        assert!(s.contains("l2="));
        assert!(s.contains("huber="));
        assert!(s.contains("lpips_proxy="));
        assert!(s.contains("mode=Progressive"));
    }

    // -----------------------------------------------------------------------
    // EmaTeacher tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_teacher_new() {
        let ema = EmaTeacher::new(0.999, 3);
        assert_eq!(ema.weights().len(), 3);
        assert_eq!(ema.step(), 0);
        assert!((ema.decay - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_ema_teacher_update() {
        let mut ema = EmaTeacher::new(0.9, 2);
        let student = vec![vec![1.0_f32; 4], vec![2.0_f32; 4]];
        ema.update(&student).expect("update should succeed");
        assert_eq!(ema.step(), 1);
        // After first update, effective_decay = min(0.9, 2/11) ≈ 0.1818
        // ema_w = 0.1818 * 0.0 + (1 - 0.1818) * 1.0 ≈ 0.8182
        let w = &ema.weights()[0];
        assert_eq!(w.len(), 4);
        for &v in w {
            assert!(v > 0.5 && v < 1.0, "unexpected weight {v}");
        }
    }

    #[test]
    fn test_ema_teacher_update_mismatch() {
        let mut ema = EmaTeacher::new(0.9, 2);
        let student = vec![vec![1.0_f32; 4]]; // wrong count
        assert!(ema.update(&student).is_err());
    }

    #[test]
    fn test_ema_teacher_effective_decay() {
        let ema = EmaTeacher::new(0.999, 0);
        // step = 0 → (1+0)/(10+0) = 0.1 → effective = min(0.999, 0.1) = 0.1
        assert!((ema.effective_decay() - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_ema_teacher_reset() {
        let mut ema = EmaTeacher::new(0.9, 2);
        let student = vec![vec![1.0_f32; 4], vec![2.0_f32; 4]];
        ema.update(&student).expect("update should succeed");
        assert_eq!(ema.step(), 1);

        let new_weights = vec![vec![0.0_f32; 4], vec![0.0_f32; 4]];
        ema.reset(new_weights);
        assert_eq!(ema.step(), 0);
        for w in ema.weights() {
            for &v in w {
                assert!((v - 0.0).abs() < f32::EPSILON);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Aggregate losses test
    // -----------------------------------------------------------------------

    #[test]
    fn test_aggregate_losses() {
        let cfg = DistillationConfig::default();
        let pairs = vec![make_pair(8, 1.0, 0.0), make_pair(8, 2.0, 0.0)];
        let result = DistillationStep::aggregate_losses(&pairs, &cfg).expect("aggregate");
        assert!(result.is_finite());
        assert!(result.total_loss > 0.0);
        assert_eq!(result.mode, cfg.mode);

        // Empty pairs → zeroed result
        let empty_result = DistillationStep::aggregate_losses(&[], &cfg).expect("aggregate empty");
        assert!((empty_result.total_loss).abs() < f32::EPSILON);

        // A malformed pair propagates the error instead of silently averaging.
        let mut bad = make_pair(8, 1.0, 0.0);
        bad.student.predicted_x0 = vec![0.0; 3];
        assert!(DistillationStep::aggregate_losses(&[bad], &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // v-prediction / helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_v_prediction_round_trip() {
        // Pick x0 and eps, build x_t and v, then check both are recovered.
        let alpha_cumprod: f32 = 0.64; // alpha_t = 0.8, sigma_t = 0.6
        let (alpha_t, sigma_t) = vp_coefficients(alpha_cumprod);
        let x0 = 0.5_f32;
        let eps = -0.25_f32;
        let x_t = alpha_t * x0 + sigma_t * eps;
        let v = alpha_t * eps - sigma_t * x0;

        let pred = NoisePrediction::from_v_prediction(7, vec![x_t; 4], vec![v; 4], alpha_cumprod);
        assert!(pred.is_v_prediction);
        assert_eq!(pred.predicted_x0.len(), 4);
        for &value in &pred.predicted_x0 {
            assert!((value - x0).abs() < 1e-5, "expected {x0}, got {value}");
        }

        // The raw output stays v, but epsilon_at recovers epsilon.
        for &value in &pred.predicted_noise {
            assert!((value - v).abs() < 1e-6);
        }
        for value in pred.epsilon_at(alpha_cumprod) {
            assert!((value - eps).abs() < 1e-5, "expected {eps}, got {value}");
        }
    }

    /// A v-parameterised pair must not be treated as an epsilon one.
    #[test]
    fn test_v_prediction_pair_uses_v_arithmetic() {
        let (alpha_t, sigma_t) = vp_coefficients(FIXTURE_ALPHA_T);
        let x0 = 0.6_f32;
        let eps = (1.0 - alpha_t * x0) / sigma_t; // so that x_t == 1.0
        let v = alpha_t * eps - sigma_t * x0;

        let v_pred =
            NoisePrediction::from_v_prediction(10, vec![1.0; 8], vec![v; 8], FIXTURE_ALPHA_T);
        let eps_pred = make_consistent_prediction(10, 8, x0);

        // Both parameterisations describe the same network output.
        for (a, b) in v_pred
            .epsilon_at(FIXTURE_ALPHA_T)
            .iter()
            .zip(eps_pred.predicted_noise.iter())
        {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }

        // ... so the distillation loss against the same student must match.
        let cfg = DistillationConfig::default();
        let student = make_consistent_prediction(10, 8, 0.1);
        let v_loss = compute_distillation_loss(
            &TeacherStudentPair::new(v_pred, student.clone(), FIXTURE_ALPHA_T, FIXTURE_ALPHA_S),
            &cfg,
        )
        .expect("v-prediction loss");
        let eps_loss = compute_distillation_loss(
            &TeacherStudentPair::new(eps_pred, student, FIXTURE_ALPHA_T, FIXTURE_ALPHA_S),
            &cfg,
        )
        .expect("epsilon loss");

        assert!(
            (v_loss.l2_loss - eps_loss.l2_loss).abs() < 1e-5,
            "{} vs {}",
            v_loss.l2_loss,
            eps_loss.l2_loss
        );
    }

    #[test]
    fn test_midpoint_alpha_cumprod_is_between() {
        let mid = midpoint_alpha_cumprod(0.2, 0.95);
        assert!(mid > 0.2 && mid < 0.95, "mid = {mid}");
        // Symmetric in log-SNR: the midpoint of a level with itself is itself.
        assert!((midpoint_alpha_cumprod(0.5, 0.5) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_lpips_proxy_loss_planar() {
        let dims = LatentDims::new(2, 8, 8);
        assert_eq!(dims.len(), 128);
        assert!(!dims.is_empty());

        let teacher = vec![1.0_f32; 128];
        let student = vec![0.0_f32; 128];
        let loss = lpips_proxy_loss_planar(&teacher, &student, 4, dims).expect("planar loss");
        assert!((loss - 1.0).abs() < 1e-5, "expected 1.0, got {loss}");

        // Identical inputs → zero.
        let zero = lpips_proxy_loss_planar(&teacher, &teacher, 4, dims).expect("planar loss");
        assert!(zero.abs() < f32::EPSILON);

        // Layout mismatches are errors, not silent whole-slice L1.
        assert!(lpips_proxy_loss_planar(&teacher, &student, 4, LatentDims::new(3, 8, 8)).is_err());
        assert!(lpips_proxy_loss_planar(&teacher, &student, 0, dims).is_err());
        assert!(lpips_proxy_loss_planar(&teacher, &student[..64], 4, dims).is_err());
    }

    #[test]
    fn test_ema_teacher_new_with_shapes() {
        let ema = EmaTeacher::new_with_shapes(0.99, &[3, 5]);
        assert_eq!(ema.weights().len(), 2);
        assert_eq!(ema.weights()[0].len(), 3);
        assert_eq!(ema.weights()[1].len(), 5);
        for tensor in ema.weights() {
            for &value in tensor {
                assert!(value.abs() < f32::EPSILON);
            }
        }
        assert_eq!(ema.step(), 0);

        // `new` documents empty shadow tensors instead.
        assert!(EmaTeacher::new(0.99, 2)
            .weights()
            .iter()
            .all(|w| w.is_empty()));
    }
}
