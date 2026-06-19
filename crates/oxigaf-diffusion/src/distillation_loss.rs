//! # Consistency Distillation Loss
//!
//! Implements consistency distillation and progressive distillation loss
//! computation as described in the Consistency Models paper.
//!
//! Consistency distillation trains a student model to predict the same
//! denoised output regardless of the noise level, enabling few-step
//! (even single-step) generation.

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
        }
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
    /// Raw network noise prediction (epsilon or v-parameterisation).
    pub predicted_noise: Vec<f32>,
    /// Predicted clean sample x_0.
    pub predicted_x0: Vec<f32>,
    /// Whether the network uses v-prediction parameterisation.
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
}

// ---------------------------------------------------------------------------
// TeacherStudentPair
// ---------------------------------------------------------------------------

/// Teacher-student prediction pair for one distillation step.
#[derive(Debug, Clone)]
pub struct TeacherStudentPair {
    /// Teacher model prediction at the current timestep.
    pub teacher: NoisePrediction,
    /// Student model prediction at the current timestep.
    pub student: NoisePrediction,
    /// Cumulative product of alphas at the current (noisier) timestep.
    pub alpha_cumprod_t: f32,
    /// Cumulative product of alphas at the target (fewer-step / cleaner) timestep.
    pub alpha_cumprod_t_minus_n: f32,
}

// ---------------------------------------------------------------------------
// Free loss functions
// ---------------------------------------------------------------------------

/// Compute the L2 (MSE) loss between the clean-sample predictions of the
/// teacher and student.
///
/// Returns `f32::NAN` when the prediction vectors have different lengths.
pub fn l2_distillation_loss(teacher: &NoisePrediction, student: &NoisePrediction) -> f32 {
    let t = &teacher.predicted_x0;
    let s = &student.predicted_x0;
    if t.len() != s.len() {
        return f32::NAN;
    }
    if t.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = t
        .iter()
        .zip(s.iter())
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .sum();
    sum_sq / t.len() as f32
}

/// Pseudo-LPIPS spatial-coherence proxy loss.
///
/// Divides the image (given as a flat slice in row-major order) into
/// non-overlapping `patch_size × patch_size` patches and computes the mean
/// per-patch L1 distance between `teacher_x0` and `student_x0`.
///
/// Returns `0.0` for identical inputs, `f32::NAN` when lengths mismatch or
/// when `patch_size` is 0.
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
    /// Raw L2 (MSE) loss between teacher and student x0 predictions.
    pub l2_loss: f32,
    /// Raw Huber loss between teacher and student x0 predictions.
    pub huber_loss: f32,
    /// Raw pseudo-LPIPS proxy loss (0 when disabled).
    pub lpips_proxy: f32,
    /// Consistency weight applied to the Huber term.
    pub consistency_weight: f32,
}

impl DistillationLossResult {
    /// Human-readable summary of the loss breakdown.
    pub fn format(&self) -> String {
        format!(
            "DistillationLossResult {{ total={:.6}, l2={:.6}, huber={:.6}, lpips_proxy={:.6}, consistency_weight={:.6} }}",
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

/// Compute the full distillation loss for a single teacher-student pair.
///
/// The total loss is:
///   total = loss_weight * l2 + consistency_loss_weight * huber + lpips_proxy_weight * lpips
pub fn compute_distillation_loss(
    pair: &TeacherStudentPair,
    config: &DistillationConfig,
) -> DistillationLossResult {
    let l2 = l2_distillation_loss(&pair.teacher, &pair.student);
    let huber = huber_loss(
        &pair.teacher.predicted_x0,
        &pair.student.predicted_x0,
        HUBER_DELTA,
    );

    // LPIPS proxy: attempt to infer spatial dimensions from the slice length.
    // We use a square assumption as a reasonable fallback.
    let n = pair.teacher.predicted_x0.len();
    let side = (n as f32).sqrt() as usize;
    let lpips = if config.lpips_proxy_weight > 0.0 && n > 0 {
        lpips_proxy_loss(
            &pair.teacher.predicted_x0,
            &pair.student.predicted_x0,
            LPIPS_PATCH_SIZE,
            side,
            side,
        )
    } else {
        0.0
    };

    let l2_finite = if l2.is_finite() { l2 } else { 0.0 };
    let huber_finite = if huber.is_finite() { huber } else { 0.0 };
    let lpips_finite = if lpips.is_finite() { lpips } else { 0.0 };

    let total = config.loss_weight * l2_finite
        + config.consistency_loss_weight * huber_finite
        + config.lpips_proxy_weight * lpips_finite;

    DistillationLossResult {
        total_loss: total,
        l2_loss: l2,
        huber_loss: huber,
        lpips_proxy: lpips,
        consistency_weight: config.consistency_loss_weight,
    }
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
    /// Initialise the EMA teacher with `num_params` all-zero parameter tensors.
    pub fn new(decay: f32, num_params: usize) -> Self {
        Self {
            decay,
            step: 0,
            weights: vec![Vec::new(); num_params],
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
    pub fn aggregate_losses(
        pairs: &[TeacherStudentPair],
        config: &DistillationConfig,
    ) -> DistillationLossResult {
        if pairs.is_empty() {
            return DistillationLossResult {
                total_loss: 0.0,
                l2_loss: 0.0,
                huber_loss: 0.0,
                lpips_proxy: 0.0,
                consistency_weight: config.consistency_loss_weight,
            };
        }

        let n = pairs.len() as f32;
        let mut sum_total: f32 = 0.0;
        let mut sum_l2: f32 = 0.0;
        let mut sum_huber: f32 = 0.0;
        let mut sum_lpips: f32 = 0.0;

        for pair in pairs {
            let r = compute_distillation_loss(pair, config);
            sum_total += r.total_loss;
            sum_l2 += r.l2_loss;
            sum_huber += r.huber_loss;
            sum_lpips += r.lpips_proxy;
        }

        DistillationLossResult {
            total_loss: sum_total / n,
            l2_loss: sum_l2 / n,
            huber_loss: sum_huber / n,
            lpips_proxy: sum_lpips / n,
            consistency_weight: config.consistency_loss_weight,
        }
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

    fn make_pair(len: usize, teacher_val: f32, student_val: f32) -> TeacherStudentPair {
        TeacherStudentPair {
            teacher: make_prediction(10, len, teacher_val, 0.0),
            student: make_prediction(10, len, student_val, 0.0),
            alpha_cumprod_t: 0.9,
            alpha_cumprod_t_minus_n: 0.7,
        }
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
        let result = compute_distillation_loss(&pair, &cfg);
        // l2 = 1.0, huber = 2.5 (diff=1.0, delta=1.0 → 1.0 - 0.5 = 0.5)
        // total = 1.0 * 1.0 + 0.1 * 0.5 = 1.05
        assert!(result.total_loss > 0.0);
        assert!(result.l2_loss > 0.0);
        assert!(result.is_finite());
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
        };
        assert!(r.is_finite());

        let r_nan = DistillationLossResult {
            total_loss: f32::NAN,
            l2_loss: 0.5,
            huber_loss: 0.3,
            lpips_proxy: 0.0,
            consistency_weight: 0.1,
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
        };
        let s = r.format();
        assert!(s.contains("total="));
        assert!(s.contains("l2="));
        assert!(s.contains("huber="));
        assert!(s.contains("lpips_proxy="));
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
        let result = DistillationStep::aggregate_losses(&pairs, &cfg);
        assert!(result.is_finite());
        assert!(result.total_loss > 0.0);

        // Empty pairs → zeroed result
        let empty_result = DistillationStep::aggregate_losses(&[], &cfg);
        assert!((empty_result.total_loss).abs() < f32::EPSILON);
    }
}
