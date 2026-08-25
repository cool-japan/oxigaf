//! Regularization losses and penalties for 3D Gaussian Splatting training.
//!
//! Provides:
//! - L1 and L2 weight-decay penalties with per-parameter gradients
//! - Opacity regularization (sparsity, binary, entropy) operating in logit space
//! - Scale regularization (volume, anisotropy, max-scale, combined)
//! - Positional regularization (L1/L2 distance from origin or initial positions)
//! - A composite regularizer that combines all terms with configurable weights

use thiserror::Error;

// ──────────────────────────────────────────────────────────────────────────────
// Errors
// ──────────────────────────────────────────────────────────────────────────────

/// Errors produced by regularization computations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum RegularizationError {
    /// Input slice lengths did not match the expected size.
    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// Input slice was empty when at least one element was required.
    #[error("empty input")]
    EmptyInput,

    /// Configuration value is invalid (e.g. negative weight).
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper functions
// ──────────────────────────────────────────────────────────────────────────────

/// Sigmoid with numerical stability clamp.
///
/// Clamps the input to `[-20, 20]` before computing `1 / (1 + exp(-x))`
/// to prevent overflow/underflow.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    let clamped = x.clamp(-20.0, 20.0);
    1.0 / (1.0 + (-clamped).exp())
}

/// Soft L1 (Huber-like) loss for a single value: `sqrt(x² + eps²) - eps`.
///
/// This smoothed variant avoids the non-differentiable kink at zero.
#[inline]
pub fn soft_l1(x: f32, eps: f32) -> f32 {
    (x * x + eps * eps).sqrt() - eps
}

/// Gradient of `soft_l1` with respect to `x`: `x / sqrt(x² + eps²)`.
#[inline]
pub fn soft_l1_grad(x: f32, eps: f32) -> f32 {
    x / (x * x + eps * eps).sqrt()
}

/// Root-mean-square of a slice. Returns `0.0` for an empty slice.
pub fn rms(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mean_sq = vals.iter().map(|&v| v * v).sum::<f32>() / vals.len() as f32;
    mean_sq.sqrt()
}

// ──────────────────────────────────────────────────────────────────────────────
// L1 Regularization
// ──────────────────────────────────────────────────────────────────────────────

/// Computes L1 penalty: `weight * Σ |param_i|`.
#[derive(Debug, Clone)]
pub struct L1Regularization {
    /// Non-negative regularization strength.
    pub weight: f32,
}

impl L1Regularization {
    /// Create a new L1 regularizer.
    ///
    /// # Errors
    /// Returns [`RegularizationError::InvalidConfig`] if `weight < 0.0`.
    pub fn new(weight: f32) -> Result<Self, RegularizationError> {
        if weight < 0.0 {
            return Err(RegularizationError::InvalidConfig(format!(
                "weight must be non-negative, got {weight}"
            )));
        }
        Ok(Self { weight })
    }

    /// Compute the L1 penalty: `weight * Σ |param_i|`.
    pub fn loss(&self, params: &[f32]) -> f32 {
        self.weight * params.iter().map(|&p| p.abs()).sum::<f32>()
    }

    /// Per-element gradient: `weight * sign(param_i)`.
    ///
    /// The subgradient at zero is taken to be zero.
    pub fn gradient(&self, params: &[f32]) -> Vec<f32> {
        params
            .iter()
            .map(|&p| {
                let sign = if p > 0.0 {
                    1.0_f32
                } else if p < 0.0 {
                    -1.0_f32
                } else {
                    0.0_f32
                };
                self.weight * sign
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// L2 Regularization
// ──────────────────────────────────────────────────────────────────────────────

/// Computes L2 penalty (weight decay): `0.5 * weight * Σ param_i²`.
#[derive(Debug, Clone)]
pub struct L2Regularization {
    /// Non-negative regularization strength.
    pub weight: f32,
}

impl L2Regularization {
    /// Create a new L2 regularizer.
    ///
    /// # Errors
    /// Returns [`RegularizationError::InvalidConfig`] if `weight < 0.0`.
    pub fn new(weight: f32) -> Result<Self, RegularizationError> {
        if weight < 0.0 {
            return Err(RegularizationError::InvalidConfig(format!(
                "weight must be non-negative, got {weight}"
            )));
        }
        Ok(Self { weight })
    }

    /// Compute the L2 penalty: `0.5 * weight * Σ param_i²`.
    pub fn loss(&self, params: &[f32]) -> f32 {
        0.5 * self.weight * params.iter().map(|&p| p * p).sum::<f32>()
    }

    /// Per-element gradient: `weight * param_i`.
    pub fn gradient(&self, params: &[f32]) -> Vec<f32> {
        params.iter().map(|&p| self.weight * p).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Opacity Regularization
// ──────────────────────────────────────────────────────────────────────────────

/// How to penalize Gaussian opacities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpacityRegMode {
    /// Sparsity: penalize non-zero opacities.
    ///
    /// `loss = weight * Σ sigmoid(logit_i)`
    Sparsity,

    /// Binary: push opacities toward 0 or 1.
    ///
    /// `loss = weight * Σ sigmoid(logit_i) * (1 - sigmoid(logit_i))`
    Binary,

    /// Entropy: penalizes uncertain (mid-range) opacities.
    ///
    /// `loss = weight * Σ -[σ·log(σ) + (1-σ)·log(1-σ)]` where `σ = sigmoid(logit_i)`
    Entropy,
}

/// Regularizer that operates on per-Gaussian opacity logits.
///
/// The actual opacity is `σ = sigmoid(logit)`.  All gradients are computed
/// with the chain rule through the sigmoid.
#[derive(Debug, Clone)]
pub struct OpacityRegularization {
    /// Non-negative regularization strength.
    pub weight: f32,
    /// Which opacity penalty to apply.
    pub mode: OpacityRegMode,
}

impl OpacityRegularization {
    /// Create a new opacity regularizer.
    ///
    /// # Errors
    /// Returns [`RegularizationError::InvalidConfig`] if `weight < 0.0`.
    pub fn new(weight: f32, mode: OpacityRegMode) -> Result<Self, RegularizationError> {
        if weight < 0.0 {
            return Err(RegularizationError::InvalidConfig(format!(
                "weight must be non-negative, got {weight}"
            )));
        }
        Ok(Self { weight, mode })
    }

    /// Compute the opacity penalty over all logits.
    pub fn loss(&self, logits: &[f32]) -> f32 {
        let sum: f32 = logits
            .iter()
            .map(|&l| {
                let s = sigmoid(l);
                match self.mode {
                    OpacityRegMode::Sparsity => s,
                    OpacityRegMode::Binary => s * (1.0 - s),
                    OpacityRegMode::Entropy => {
                        let s_safe = s.clamp(1e-7, 1.0 - 1e-7);
                        let one_minus = 1.0 - s_safe;
                        -(s_safe * s_safe.ln() + one_minus * one_minus.ln())
                    }
                }
            })
            .sum();
        self.weight * sum
    }

    /// Gradient of the opacity penalty w.r.t. each `logit_i`.
    ///
    /// All modes apply the chain rule through `sigmoid`:
    /// - **Sparsity**: `weight * σ*(1-σ)`
    /// - **Binary**: `weight * (1-2σ) * σ*(1-σ)`
    /// - **Entropy**: `weight * σ*(1-σ) * (-logit)`  (= `-weight * σ*(1-σ) * log(σ/(1-σ))`)
    pub fn gradient(&self, logits: &[f32]) -> Vec<f32> {
        logits
            .iter()
            .map(|&l| {
                let s = sigmoid(l);
                let ds_dl = s * (1.0 - s); // dσ/d(logit)
                let dloss_ds = match self.mode {
                    OpacityRegMode::Sparsity => 1.0,
                    OpacityRegMode::Binary => 1.0 - 2.0 * s,
                    OpacityRegMode::Entropy => {
                        // H(σ) = -[σ·ln σ + (1-σ)·ln(1-σ)]
                        // dH/dσ = -(ln σ - ln(1-σ)) = -logit
                        // d(loss)/d(logit) = weight * σ*(1-σ) * (-logit)
                        //
                        // Descending this gradient therefore pushes σ AWAY
                        // from 0.5 toward 0 or 1 (minimizing entropy /
                        // maximizing certainty), matching this mode's
                        // documented purpose of penalizing mid-range
                        // opacities. The previous `l` (no negation) did the
                        // opposite: it pushed every opacity toward σ=0.5.
                        -l
                    }
                };
                self.weight * dloss_ds * ds_dl
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Scale Regularization
// ──────────────────────────────────────────────────────────────────────────────

/// How to penalize Gaussian scales (stored as `log_scale`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleRegMode {
    /// Penalize large total volume.
    ///
    /// `loss = weight * Σ exp(log_sx + log_sy + log_sz)`
    Volume,

    /// Penalize anisotropy (difference between largest and smallest log-scale).
    ///
    /// `loss = weight * Σ (max(log_s) - min(log_s))` per Gaussian
    Anisotropy,

    /// Penalize the largest axis scale per Gaussian.
    ///
    /// `loss = weight * Σ max(exp(log_s))` per Gaussian
    MaxScale,

    /// Combined volume + anisotropy with independent sub-weights.
    Combined {
        volume_weight: f32,
        anisotropy_weight: f32,
    },
}

/// Regularizer acting on per-Gaussian log-scale triplets `[log_sx, log_sy, log_sz]`.
///
/// `log_scales` is a flat `Vec<f32>` of length `3 * n_gaussians`.
#[derive(Debug, Clone)]
pub struct ScaleRegularization {
    /// Non-negative outer regularization strength.
    pub weight: f32,
    /// Which scale penalty to apply.
    pub mode: ScaleRegMode,
}

impl ScaleRegularization {
    /// Create a new scale regularizer.
    ///
    /// # Errors
    /// Returns [`RegularizationError::InvalidConfig`] for any negative weight
    /// (including sub-weights inside `Combined`).
    pub fn new(weight: f32, mode: ScaleRegMode) -> Result<Self, RegularizationError> {
        if weight < 0.0 {
            return Err(RegularizationError::InvalidConfig(format!(
                "weight must be non-negative, got {weight}"
            )));
        }
        if let ScaleRegMode::Combined {
            volume_weight,
            anisotropy_weight,
        } = mode
        {
            if volume_weight < 0.0 {
                return Err(RegularizationError::InvalidConfig(format!(
                    "volume_weight must be non-negative, got {volume_weight}"
                )));
            }
            if anisotropy_weight < 0.0 {
                return Err(RegularizationError::InvalidConfig(format!(
                    "anisotropy_weight must be non-negative, got {anisotropy_weight}"
                )));
            }
        }
        Ok(Self { weight, mode })
    }

    /// Compute the scale penalty.
    ///
    /// `log_scales` must have length `3 * n` (triplets per Gaussian).
    ///
    /// # Errors
    /// - [`RegularizationError::EmptyInput`] if `log_scales` is empty.
    /// - [`RegularizationError::LengthMismatch`] if length is not a multiple of 3.
    pub fn loss(&self, log_scales: &[f32]) -> Result<f32, RegularizationError> {
        let n = validate_triplets(log_scales)?;

        let sum: f32 = (0..n)
            .map(|i| {
                let lx = log_scales[3 * i];
                let ly = log_scales[3 * i + 1];
                let lz = log_scales[3 * i + 2];
                match self.mode {
                    ScaleRegMode::Volume => (lx + ly + lz).exp(),
                    ScaleRegMode::Anisotropy => {
                        let max_l = lx.max(ly).max(lz);
                        let min_l = lx.min(ly).min(lz);
                        max_l - min_l
                    }
                    ScaleRegMode::MaxScale => lx.exp().max(ly.exp()).max(lz.exp()),
                    ScaleRegMode::Combined {
                        volume_weight,
                        anisotropy_weight,
                    } => {
                        let vol = (lx + ly + lz).exp();
                        let max_l = lx.max(ly).max(lz);
                        let min_l = lx.min(ly).min(lz);
                        let aniso = max_l - min_l;
                        volume_weight * vol + anisotropy_weight * aniso
                    }
                }
            })
            .sum();

        Ok(self.weight * sum)
    }

    /// Compute the gradient of the scale penalty w.r.t. `log_scales`.
    ///
    /// Output has the same length as `log_scales`.
    ///
    /// # Errors
    /// Same conditions as [`Self::loss`].
    pub fn gradient(&self, log_scales: &[f32]) -> Result<Vec<f32>, RegularizationError> {
        let n = validate_triplets(log_scales)?;
        let mut grad = vec![0.0_f32; 3 * n];

        for i in 0..n {
            let lx = log_scales[3 * i];
            let ly = log_scales[3 * i + 1];
            let lz = log_scales[3 * i + 2];

            let (gx, gy, gz) = match self.mode {
                ScaleRegMode::Volume => {
                    // d/d(log_s) exp(lx+ly+lz) = exp(lx+ly+lz) for all three
                    let v = (lx + ly + lz).exp();
                    (v, v, v)
                }
                ScaleRegMode::Anisotropy => {
                    // Subgradient: +1 at max, -1 at min, 0 at middle
                    // Ties: first occurrence wins for max, last for min (consistent).
                    anisotropy_subgradient(lx, ly, lz)
                }
                ScaleRegMode::MaxScale => {
                    // Gradient of max(exp(lx), exp(ly), exp(lz)) w.r.t. log_s:
                    // d max/d log_s = exp(log_s_max) only at the winning index.
                    let ex = lx.exp();
                    let ey = ly.exp();
                    let ez = lz.exp();
                    let max_exp = ex.max(ey).max(ez);
                    let gx = if (ex - max_exp).abs() < f32::EPSILON {
                        ex
                    } else {
                        0.0
                    };
                    let gy = if (ey - max_exp).abs() < f32::EPSILON && gx == 0.0 {
                        ey
                    } else {
                        0.0
                    };
                    let gz = if (ez - max_exp).abs() < f32::EPSILON && gx == 0.0 && gy == 0.0 {
                        ez
                    } else {
                        0.0
                    };
                    (gx, gy, gz)
                }
                ScaleRegMode::Combined {
                    volume_weight,
                    anisotropy_weight,
                } => {
                    let v = (lx + ly + lz).exp();
                    let vol_grad = (v * volume_weight, v * volume_weight, v * volume_weight);
                    let (ax, ay, az) = anisotropy_subgradient(lx, ly, lz);
                    (
                        vol_grad.0 + anisotropy_weight * ax,
                        vol_grad.1 + anisotropy_weight * ay,
                        vol_grad.2 + anisotropy_weight * az,
                    )
                }
            };

            grad[3 * i] = self.weight * gx;
            grad[3 * i + 1] = self.weight * gy;
            grad[3 * i + 2] = self.weight * gz;
        }

        Ok(grad)
    }
}

/// Subgradient of `(max - min)` for three values `(a, b, c)`.
///
/// Returns `(ga, gb, gc)` where the max component gets `+1`, the min gets `-1`,
/// and the middle gets `0`.  Ties (e.g. all equal) produce `(0, 0, 0)`.
#[inline]
fn anisotropy_subgradient(a: f32, b: f32, c: f32) -> (f32, f32, f32) {
    let max_val = a.max(b).max(c);
    let min_val = a.min(b).min(c);

    // When all three are equal, max == min → loss is 0, gradient is 0.
    if (max_val - min_val).abs() < f32::EPSILON {
        return (0.0, 0.0, 0.0);
    }

    // Assign +1 to the first index that achieves max_val.
    let mut ga = 0.0_f32;
    let mut gb = 0.0_f32;
    let mut gc = 0.0_f32;

    if (a - max_val).abs() < f32::EPSILON {
        ga = 1.0;
    } else if (b - max_val).abs() < f32::EPSILON {
        gb = 1.0;
    } else {
        gc = 1.0;
    }

    // Assign -1 to the first index that achieves min_val.
    if (a - min_val).abs() < f32::EPSILON {
        ga -= 1.0;
    } else if (b - min_val).abs() < f32::EPSILON {
        gb -= 1.0;
    } else {
        gc -= 1.0;
    }

    (ga, gb, gc)
}

/// Validate that a flat slice represents triplets.  Returns the number of triplets.
fn validate_triplets(data: &[f32]) -> Result<usize, RegularizationError> {
    if data.is_empty() {
        return Err(RegularizationError::EmptyInput);
    }
    if !data.len().is_multiple_of(3) {
        return Err(RegularizationError::LengthMismatch {
            expected: (data.len() / 3) * 3,
            actual: data.len(),
        });
    }
    Ok(data.len() / 3)
}

// ──────────────────────────────────────────────────────────────────────────────
// Positional Regularization
// ──────────────────────────────────────────────────────────────────────────────

/// How to penalize Gaussian positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PosRegMode {
    /// L2 distance from origin: `weight * Σ (x² + y² + z²)`.
    L2FromOrigin,
    /// L1 (Manhattan) distance from origin: `weight * Σ (|x| + |y| + |z|)`.
    L1FromOrigin,
}

/// Regularizer that penalizes Gaussian positions.
///
/// `positions` is a flat `Vec<f32>` of length `3 * n_gaussians`.
#[derive(Debug, Clone)]
pub struct PositionalRegularization {
    /// Non-negative regularization strength.
    pub weight: f32,
    /// Which positional penalty to apply.
    pub mode: PosRegMode,
}

impl PositionalRegularization {
    /// Create a new positional regularizer.
    ///
    /// # Errors
    /// Returns [`RegularizationError::InvalidConfig`] if `weight < 0.0`.
    pub fn new(weight: f32, mode: PosRegMode) -> Result<Self, RegularizationError> {
        if weight < 0.0 {
            return Err(RegularizationError::InvalidConfig(format!(
                "weight must be non-negative, got {weight}"
            )));
        }
        Ok(Self { weight, mode })
    }

    /// Compute positional penalty from the origin.
    ///
    /// # Errors
    /// - [`RegularizationError::EmptyInput`] if `positions` is empty.
    /// - [`RegularizationError::LengthMismatch`] if not a multiple of 3.
    pub fn loss(&self, positions: &[f32]) -> Result<f32, RegularizationError> {
        let n = validate_triplets(positions)?;
        let sum: f32 = (0..n)
            .map(|i| {
                let x = positions[3 * i];
                let y = positions[3 * i + 1];
                let z = positions[3 * i + 2];
                match self.mode {
                    PosRegMode::L2FromOrigin => x * x + y * y + z * z,
                    PosRegMode::L1FromOrigin => x.abs() + y.abs() + z.abs(),
                }
            })
            .sum();
        Ok(self.weight * sum)
    }

    /// Gradient of the positional penalty w.r.t. each position coordinate.
    ///
    /// # Errors
    /// Same conditions as [`Self::loss`].
    pub fn gradient(&self, positions: &[f32]) -> Result<Vec<f32>, RegularizationError> {
        let n = validate_triplets(positions)?;
        let mut grad = vec![0.0_f32; 3 * n];
        for i in 0..n {
            let x = positions[3 * i];
            let y = positions[3 * i + 1];
            let z = positions[3 * i + 2];
            let (gx, gy, gz) = match self.mode {
                PosRegMode::L2FromOrigin => (2.0 * x, 2.0 * y, 2.0 * z),
                PosRegMode::L1FromOrigin => (sign_f32(x), sign_f32(y), sign_f32(z)),
            };
            grad[3 * i] = self.weight * gx;
            grad[3 * i + 1] = self.weight * gy;
            grad[3 * i + 2] = self.weight * gz;
        }
        Ok(grad)
    }

    /// Compute the penalty for drift from initial positions.
    ///
    /// `positions` and `initial` must have the same length.
    ///
    /// # Errors
    /// - Propagates validation errors from `positions`.
    /// - [`RegularizationError::LengthMismatch`] if lengths differ.
    pub fn drift_loss(
        &self,
        positions: &[f32],
        initial: &[f32],
    ) -> Result<f32, RegularizationError> {
        let n = validate_triplets(positions)?;
        if initial.len() != positions.len() {
            return Err(RegularizationError::LengthMismatch {
                expected: positions.len(),
                actual: initial.len(),
            });
        }
        let sum: f32 = (0..n)
            .map(|i| {
                let dx = positions[3 * i] - initial[3 * i];
                let dy = positions[3 * i + 1] - initial[3 * i + 1];
                let dz = positions[3 * i + 2] - initial[3 * i + 2];
                match self.mode {
                    PosRegMode::L2FromOrigin => dx * dx + dy * dy + dz * dz,
                    PosRegMode::L1FromOrigin => dx.abs() + dy.abs() + dz.abs(),
                }
            })
            .sum();
        Ok(self.weight * sum)
    }

    /// Gradient of the drift penalty w.r.t. current positions.
    ///
    /// # Errors
    /// Same conditions as [`Self::drift_loss`].
    pub fn drift_gradient(
        &self,
        positions: &[f32],
        initial: &[f32],
    ) -> Result<Vec<f32>, RegularizationError> {
        let n = validate_triplets(positions)?;
        if initial.len() != positions.len() {
            return Err(RegularizationError::LengthMismatch {
                expected: positions.len(),
                actual: initial.len(),
            });
        }
        let mut grad = vec![0.0_f32; 3 * n];
        for i in 0..n {
            let dx = positions[3 * i] - initial[3 * i];
            let dy = positions[3 * i + 1] - initial[3 * i + 1];
            let dz = positions[3 * i + 2] - initial[3 * i + 2];
            let (gx, gy, gz) = match self.mode {
                PosRegMode::L2FromOrigin => (2.0 * dx, 2.0 * dy, 2.0 * dz),
                PosRegMode::L1FromOrigin => (sign_f32(dx), sign_f32(dy), sign_f32(dz)),
            };
            grad[3 * i] = self.weight * gx;
            grad[3 * i + 1] = self.weight * gy;
            grad[3 * i + 2] = self.weight * gz;
        }
        Ok(grad)
    }
}

/// Returns the sign of `x` as `{-1.0, 0.0, 1.0}`.
#[inline]
fn sign_f32(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RegularizationConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the composite regularizer.
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// Weight for L2 (weight-decay) penalty on color/SH parameters.
    pub l2_weight: f32,
    /// Weight for opacity regularization.
    pub opacity_weight: f32,
    /// Opacity regularization mode.
    pub opacity_mode: OpacityRegMode,
    /// Weight for scale regularization.
    pub scale_weight: f32,
    /// Scale regularization mode.
    pub scale_mode: ScaleRegMode,
    /// Weight for positional regularization.
    pub positional_weight: f32,
    /// Positional regularization mode.
    pub positional_mode: PosRegMode,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l2_weight: 0.0,
            opacity_weight: 0.01,
            opacity_mode: OpacityRegMode::Entropy,
            scale_weight: 0.005,
            scale_mode: ScaleRegMode::Volume,
            positional_weight: 0.0,
            positional_mode: PosRegMode::L2FromOrigin,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CompositeRegularizer
// ──────────────────────────────────────────────────────────────────────────────

/// Combines multiple regularization terms into a single total loss.
pub struct CompositeRegularizer {
    /// Configuration controlling weights and modes for each term.
    pub config: RegularizationConfig,
}

impl CompositeRegularizer {
    /// Create a new composite regularizer with the given configuration.
    pub fn new(config: RegularizationConfig) -> Self {
        Self { config }
    }

    /// Total regularization loss for a Gaussian model.
    ///
    /// - `color_params`: SH coefficients (any length); receives L2 penalty.
    /// - `opacity_logits`: per-Gaussian opacity in logit space.
    /// - `log_scales`: flat `[sx, sy, sz, …]` of length `3 * n`.
    /// - `positions`: flat `[x, y, z, …]` of length `3 * n`.
    ///
    /// # Errors
    /// Propagates any [`RegularizationError`] from sub-regularizers.
    pub fn total_loss(
        &self,
        color_params: &[f32],
        opacity_logits: &[f32],
        log_scales: &[f32],
        positions: &[f32],
    ) -> Result<f32, RegularizationError> {
        let breakdown = self.loss_breakdown(color_params, opacity_logits, log_scales, positions)?;
        Ok(breakdown.total)
    }

    /// Per-component loss breakdown.
    ///
    /// # Errors
    /// Propagates any [`RegularizationError`] from sub-regularizers.
    pub fn loss_breakdown(
        &self,
        color_params: &[f32],
        opacity_logits: &[f32],
        log_scales: &[f32],
        positions: &[f32],
    ) -> Result<RegBreakdown, RegularizationError> {
        // L2 on color / SH parameters (zero if weight is zero).
        let l2_loss = if self.config.l2_weight > 0.0 {
            let l2 = L2Regularization::new(self.config.l2_weight)?;
            l2.loss(color_params)
        } else {
            0.0
        };

        // Opacity regularization.
        let opacity_loss = if self.config.opacity_weight > 0.0 {
            let op =
                OpacityRegularization::new(self.config.opacity_weight, self.config.opacity_mode)?;
            op.loss(opacity_logits)
        } else {
            0.0
        };

        // Scale regularization.
        let scale_loss = if self.config.scale_weight > 0.0 && !log_scales.is_empty() {
            let sr = ScaleRegularization::new(self.config.scale_weight, self.config.scale_mode)?;
            sr.loss(log_scales)?
        } else {
            0.0
        };

        // Positional regularization.
        let positional_loss = if self.config.positional_weight > 0.0 && !positions.is_empty() {
            let pr = PositionalRegularization::new(
                self.config.positional_weight,
                self.config.positional_mode,
            )?;
            pr.loss(positions)?
        } else {
            0.0
        };

        let total = l2_loss + opacity_loss + scale_loss + positional_loss;

        Ok(RegBreakdown {
            l2_loss,
            opacity_loss,
            scale_loss,
            positional_loss,
            total,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RegBreakdown
// ──────────────────────────────────────────────────────────────────────────────

/// Per-component breakdown of the composite regularization loss.
pub struct RegBreakdown {
    /// L2 weight-decay loss on color/SH parameters.
    pub l2_loss: f32,
    /// Opacity regularization loss.
    pub opacity_loss: f32,
    /// Scale regularization loss.
    pub scale_loss: f32,
    /// Positional regularization loss.
    pub positional_loss: f32,
    /// Sum of all components.
    pub total: f32,
}

impl RegBreakdown {
    /// Format the breakdown as a human-readable string.
    pub fn format(&self) -> String {
        format!(
            "RegBreakdown {{ l2={:.6}, opacity={:.6}, scale={:.6}, positional={:.6}, total={:.6} }}",
            self.l2_loss, self.opacity_loss, self.scale_loss, self.positional_loss, self.total
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── Test 1: L1 loss on positive params ──────────────────────────────────

    #[test]
    fn test_l1_loss_positive() {
        let reg = L1Regularization::new(2.0).unwrap();
        let params = [1.0_f32, 2.0, 3.0];
        let expected = 2.0 * (1.0 + 2.0 + 3.0);
        assert!(approx_eq(reg.loss(&params), expected, EPS));
    }

    // ── Test 2: L1 loss on mixed-sign params ────────────────────────────────

    #[test]
    fn test_l1_loss_mixed_signs() {
        let reg = L1Regularization::new(1.5).unwrap();
        let params = [-3.0_f32, 2.0, -1.0, 4.0];
        let expected = 1.5 * (3.0 + 2.0 + 1.0 + 4.0);
        assert!(approx_eq(reg.loss(&params), expected, EPS));
    }

    // ── Test 3: L1 gradient sign matches param sign ──────────────────────────

    #[test]
    fn test_l1_gradient_sign() {
        let reg = L1Regularization::new(1.0).unwrap();
        let params = [-2.0_f32, 0.0, 3.0];
        let grad = reg.gradient(&params);
        assert_eq!(grad.len(), 3);
        assert!(approx_eq(grad[0], -1.0, EPS)); // negative param
        assert!(approx_eq(grad[1], 0.0, EPS)); // zero param
        assert!(approx_eq(grad[2], 1.0, EPS)); // positive param
    }

    // ── Test 4: L2 loss = 0.5 * weight * sum(x²) ────────────────────────────

    #[test]
    fn test_l2_loss_formula() {
        let weight = 0.01_f32;
        let reg = L2Regularization::new(weight).unwrap();
        let params = [1.0_f32, 2.0, 3.0];
        let expected = 0.5 * weight * (1.0 + 4.0 + 9.0);
        assert!(approx_eq(reg.loss(&params), expected, EPS));
    }

    // ── Test 5: L2 gradient = weight * x ────────────────────────────────────

    #[test]
    fn test_l2_gradient() {
        let weight = 0.5_f32;
        let reg = L2Regularization::new(weight).unwrap();
        let params = [2.0_f32, -3.0, 0.5];
        let grad = reg.gradient(&params);
        for (g, &p) in grad.iter().zip(params.iter()) {
            assert!(approx_eq(*g, weight * p, EPS));
        }
    }

    // ── Test 6: L2 zero params → zero loss ──────────────────────────────────

    #[test]
    fn test_l2_zero_params() {
        let reg = L2Regularization::new(1.0).unwrap();
        let params = [0.0_f32, 0.0, 0.0];
        assert!(approx_eq(reg.loss(&params), 0.0, EPS));
    }

    // ── Test 7: Sparsity — high logit → high loss ────────────────────────────

    #[test]
    fn test_opacity_sparsity_high_logit() {
        let reg = OpacityRegularization::new(1.0, OpacityRegMode::Sparsity).unwrap();
        // logit = 10 → sigmoid ≈ 1.0 → loss ≈ 1.0
        let loss_high = reg.loss(&[10.0]);
        // logit = -10 → sigmoid ≈ 0.0 → loss ≈ 0.0
        let loss_low = reg.loss(&[-10.0]);
        assert!(loss_high > 0.9, "expected near 1.0, got {loss_high}");
        assert!(loss_low < 0.1, "expected near 0.0, got {loss_low}");
        assert!(loss_high > loss_low);
    }

    // ── Test 8: Sparsity — very negative logit → near-zero loss ─────────────

    #[test]
    fn test_opacity_sparsity_very_negative() {
        let reg = OpacityRegularization::new(1.0, OpacityRegMode::Sparsity).unwrap();
        let loss = reg.loss(&[-20.0]);
        assert!(loss < 1e-4, "expected near zero, got {loss}");
    }

    // ── Test 9: Binary gradient at logit≈0 pushes toward boundary ────────────

    #[test]
    fn test_opacity_binary_gradient_direction() {
        let reg = OpacityRegularization::new(1.0, OpacityRegMode::Binary).unwrap();
        // At logit=0, σ=0.5; binary loss = σ(1-σ)=0.25.
        // Gradient = (1-2σ)*σ(1-σ) = 0 at exactly σ=0.5 (it's a maximum).
        // But slightly positive logit should push toward σ=1, so gradient > 0.
        let grad_pos = reg.gradient(&[0.5]);
        let grad_neg = reg.gradient(&[-0.5]);
        // Positive logit → σ>0.5 → (1-2σ)<0, so gradient negative (loss decreasing going right).
        // Negative logit → σ<0.5 → (1-2σ)>0, so gradient positive.
        // The point is gradient changes sign around 0.
        assert!(
            grad_pos[0] < grad_neg[0],
            "binary gradient should change sign: pos={}, neg={}",
            grad_pos[0],
            grad_neg[0]
        );
    }

    // ── Test 10: Entropy maximum at logit=0 ──────────────────────────────────

    #[test]
    fn test_opacity_entropy_maximum_at_zero() {
        let reg = OpacityRegularization::new(1.0, OpacityRegMode::Entropy).unwrap();
        let loss_zero = reg.loss(&[0.0]); // σ=0.5 → max entropy
        let loss_pos = reg.loss(&[3.0]); // σ≈0.95 → lower entropy
        let loss_neg = reg.loss(&[-3.0]); // σ≈0.05 → lower entropy
        assert!(
            loss_zero > loss_pos,
            "entropy at logit=0 ({loss_zero}) should exceed logit=3 ({loss_pos})"
        );
        assert!(
            loss_zero > loss_neg,
            "entropy at logit=0 ({loss_zero}) should exceed logit=-3 ({loss_neg})"
        );
        // Maximum entropy = ln(2) ≈ 0.693
        assert!(
            approx_eq(loss_zero, std::f32::consts::LN_2, 1e-4),
            "max entropy should be ln(2)≈0.693, got {loss_zero}"
        );
    }

    // ── Test 10b: Entropy gradient matches finite-difference of loss, and
    // points AWAY from the σ=0.5 maximum (descent reduces entropy) ──────────

    #[test]
    fn test_opacity_entropy_gradient_matches_finite_difference() {
        // Regression: `gradient()` used to return `weight * σ*(1-σ) * logit`
        // (no negation), the exact opposite sign of dH/d(logit) = -logit,
        // which would push opacities TOWARD σ=0.5 under gradient descent
        // instead of away from it.
        let reg = OpacityRegularization::new(1.0, OpacityRegMode::Entropy).unwrap();
        let h = 1e-3_f32;
        for &logit in &[-2.5_f32, -0.5, 0.3, 1.5, 2.5] {
            let loss_plus = reg.loss(&[logit + h]);
            let loss_minus = reg.loss(&[logit - h]);
            let numerical = (loss_plus - loss_minus) / (2.0 * h);
            let analytic = reg.gradient(&[logit])[0];
            assert!(
                (numerical - analytic).abs() < 1e-2,
                "logit={logit}: analytic grad {analytic} != numerical grad {numerical}"
            );
        }
    }

    #[test]
    fn test_opacity_entropy_gradient_descent_reduces_entropy() {
        // A single gradient-descent step from a mid-range (high-entropy)
        // opacity must REDUCE entropy (move away from σ=0.5), not increase
        // it. This is the exact user-visible consequence of the sign bug.
        let reg = OpacityRegularization::new(1.0, OpacityRegMode::Entropy).unwrap();
        let logit = 0.5_f32; // moderately uncertain, not yet at the exact peak
        let loss_before = reg.loss(&[logit]);
        let grad = reg.gradient(&[logit])[0];
        let lr = 0.1_f32;
        let logit_after = logit - lr * grad;
        let loss_after = reg.loss(&[logit_after]);
        assert!(
            loss_after < loss_before,
            "descending the entropy gradient should reduce entropy: \
             before={loss_before}, after={loss_after} (logit {logit} -> {logit_after})"
        );
    }

    // ── Test 11: ScaleReg Volume scales with exp of sum of log_scales ─────────

    #[test]
    fn test_scale_volume_loss() {
        let reg = ScaleRegularization::new(1.0, ScaleRegMode::Volume).unwrap();
        // Single Gaussian with log_scales = [1, 2, 3]
        // Volume = exp(1+2+3) = exp(6)
        let log_scales = [1.0_f32, 2.0, 3.0];
        let loss = reg.loss(&log_scales).unwrap();
        let expected = 6.0_f32.exp();
        assert!(
            approx_eq(loss, expected, 1e-3),
            "volume loss: expected {expected}, got {loss}"
        );
    }

    // ── Test 12: Anisotropy — isotropic Gaussians → zero loss ────────────────

    #[test]
    fn test_scale_anisotropy_isotropic_zero() {
        let reg = ScaleRegularization::new(1.0, ScaleRegMode::Anisotropy).unwrap();
        // Three Gaussians, all isotropic
        let log_scales = [1.0_f32, 1.0, 1.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5];
        let loss = reg.loss(&log_scales).unwrap();
        assert!(
            approx_eq(loss, 0.0, EPS),
            "isotropic should give zero anisotropy, got {loss}"
        );
    }

    // ── Test 13: Anisotropy — anisotropic → positive loss ────────────────────

    #[test]
    fn test_scale_anisotropy_anisotropic_positive() {
        let reg = ScaleRegularization::new(1.0, ScaleRegMode::Anisotropy).unwrap();
        // One very anisotropic Gaussian: log_scales differ by 4
        let log_scales = [4.0_f32, 0.0, 0.0];
        let loss = reg.loss(&log_scales).unwrap();
        assert!(
            loss > 0.0,
            "anisotropic Gaussian should have positive loss, got {loss}"
        );
        assert!(
            approx_eq(loss, 4.0, EPS),
            "anisotropy = max-min = 4-0 = 4, got {loss}"
        );
    }

    // ── Test 14: ScaleReg empty input → error ────────────────────────────────

    #[test]
    fn test_scale_empty_input() {
        let reg = ScaleRegularization::new(1.0, ScaleRegMode::Volume).unwrap();
        let result = reg.loss(&[]);
        assert!(matches!(result, Err(RegularizationError::EmptyInput)));
    }

    // ── Test 15: ScaleReg non-multiple-of-3 → LengthMismatch ─────────────────

    #[test]
    fn test_scale_length_mismatch() {
        let reg = ScaleRegularization::new(1.0, ScaleRegMode::Volume).unwrap();
        let result = reg.loss(&[1.0, 2.0]); // length 2, not multiple of 3
        assert!(matches!(
            result,
            Err(RegularizationError::LengthMismatch { .. })
        ));
    }

    // ── Test 16: PositionalReg L2 loss ───────────────────────────────────────

    #[test]
    fn test_positional_l2_loss() {
        let weight = 0.5_f32;
        let reg = PositionalRegularization::new(weight, PosRegMode::L2FromOrigin).unwrap();
        // One Gaussian at (1, 2, 3): x²+y²+z² = 1+4+9 = 14
        let positions = [1.0_f32, 2.0, 3.0];
        let loss = reg.loss(&positions).unwrap();
        let expected = weight * 14.0;
        assert!(
            approx_eq(loss, expected, EPS),
            "expected {expected}, got {loss}"
        );
    }

    // ── Test 17: PositionalReg drift — zero drift → zero loss ────────────────

    #[test]
    fn test_positional_drift_zero() {
        let reg = PositionalRegularization::new(1.0, PosRegMode::L2FromOrigin).unwrap();
        let positions = [1.0_f32, 2.0, 3.0, -1.0, 0.5, 0.0];
        let initial = positions.to_vec();
        let loss = reg.drift_loss(&positions, &initial).unwrap();
        assert!(
            approx_eq(loss, 0.0, EPS),
            "zero drift should give zero loss, got {loss}"
        );
    }

    // ── Test 18: PositionalReg drift length mismatch → Err ───────────────────

    #[test]
    fn test_positional_drift_length_mismatch() {
        let reg = PositionalRegularization::new(1.0, PosRegMode::L2FromOrigin).unwrap();
        let positions = [1.0_f32, 2.0, 3.0];
        let initial = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // different length
        let result = reg.drift_loss(&positions, &initial);
        assert!(matches!(
            result,
            Err(RegularizationError::LengthMismatch { .. })
        ));
    }

    // ── Test 19: CompositeRegularizer total_loss combines components ───────────

    #[test]
    fn test_composite_total_loss() {
        let config = RegularizationConfig {
            l2_weight: 0.1,
            opacity_weight: 0.01,
            opacity_mode: OpacityRegMode::Sparsity,
            scale_weight: 0.005,
            scale_mode: ScaleRegMode::Volume,
            positional_weight: 0.0,
            positional_mode: PosRegMode::L2FromOrigin,
        };
        let reg = CompositeRegularizer::new(config);
        let color_params = [0.5_f32, -0.3, 0.1];
        let opacity_logits = [0.0_f32, 2.0];
        let log_scales = [0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let positions = [0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let total = reg
            .total_loss(&color_params, &opacity_logits, &log_scales, &positions)
            .unwrap();
        assert!(total >= 0.0, "total loss must be non-negative, got {total}");
    }

    // ── Test 20: loss_breakdown sums to total ─────────────────────────────────

    #[test]
    fn test_composite_breakdown_sums_to_total() {
        let config = RegularizationConfig {
            l2_weight: 0.1,
            opacity_weight: 0.05,
            opacity_mode: OpacityRegMode::Binary,
            scale_weight: 0.01,
            scale_mode: ScaleRegMode::Anisotropy,
            positional_weight: 0.001,
            positional_mode: PosRegMode::L1FromOrigin,
        };
        let reg = CompositeRegularizer::new(config);
        let color_params = [1.0_f32, 2.0];
        let opacity_logits = [0.5_f32, -0.5, 1.0];
        let log_scales = [0.0_f32, 0.5, 1.0];
        let positions = [1.0_f32, 0.0, -1.0];
        let bd = reg
            .loss_breakdown(&color_params, &opacity_logits, &log_scales, &positions)
            .unwrap();
        let manual_sum = bd.l2_loss + bd.opacity_loss + bd.scale_loss + bd.positional_loss;
        assert!(
            approx_eq(bd.total, manual_sum, EPS),
            "breakdown total {} ≠ component sum {}",
            bd.total,
            manual_sum
        );
    }

    // ── Test 21: sigmoid values ───────────────────────────────────────────────

    #[test]
    fn test_sigmoid_values() {
        assert!(approx_eq(sigmoid(0.0), 0.5, EPS));
        // Large positive → approaches 1
        let s_large = sigmoid(20.0);
        assert!(
            s_large > 0.99,
            "sigmoid(20) should be near 1, got {s_large}"
        );
        // Large negative → approaches 0
        let s_small = sigmoid(-20.0);
        assert!(
            s_small < 0.01,
            "sigmoid(-20) should be near 0, got {s_small}"
        );
    }

    // ── Test 22: rms correct value ────────────────────────────────────────────

    #[test]
    fn test_rms_correct() {
        // rms([3, 4]) = sqrt((9+16)/2) = sqrt(12.5)
        let vals = [3.0_f32, 4.0];
        let expected = (12.5_f32).sqrt();
        assert!(approx_eq(rms(&vals), expected, EPS));
        // Empty slice → 0
        assert!(approx_eq(rms(&[]), 0.0, EPS));
        // Single value: rms([x]) = |x|
        assert!(approx_eq(rms(&[5.0]), 5.0, EPS));
    }

    // ── Bonus: soft_l1 helper ─────────────────────────────────────────────────

    #[test]
    fn test_soft_l1_helpers() {
        let eps = 0.1_f32;
        // At x=0: soft_l1(0, eps) = sqrt(eps²) - eps = 0
        assert!(approx_eq(soft_l1(0.0, eps), 0.0, EPS));
        // For large x, soft_l1(x, eps) ≈ |x| - eps (approaches |x| from below)
        let expected_large = 100.0_f32 - eps;
        assert!(
            approx_eq(soft_l1(100.0, eps), expected_large, 0.01),
            "soft_l1(100, {eps}) expected ≈{expected_large}, got {}",
            soft_l1(100.0, eps)
        );
        // Gradient at x=0 is 0
        assert!(approx_eq(soft_l1_grad(0.0, eps), 0.0, EPS));
        // Gradient for large positive x ≈ 1
        let g = soft_l1_grad(100.0, eps);
        assert!(
            (g - 1.0).abs() < 0.001,
            "soft_l1_grad(100, eps) should ≈ 1, got {g}"
        );
    }

    // ── Bonus: invalid config rejection ──────────────────────────────────────

    #[test]
    fn test_invalid_config_negative_weight() {
        assert!(L1Regularization::new(-0.1).is_err());
        assert!(L2Regularization::new(-1.0).is_err());
        assert!(OpacityRegularization::new(-0.5, OpacityRegMode::Sparsity).is_err());
        assert!(ScaleRegularization::new(-0.1, ScaleRegMode::Volume).is_err());
        assert!(PositionalRegularization::new(-1.0, PosRegMode::L2FromOrigin).is_err());
    }

    // ── Bonus: ScaleReg Combined sub-weight validation ────────────────────────

    #[test]
    fn test_scale_combined_negative_subweight() {
        let result = ScaleRegularization::new(
            1.0,
            ScaleRegMode::Combined {
                volume_weight: -0.1,
                anisotropy_weight: 1.0,
            },
        );
        assert!(result.is_err());
        let result2 = ScaleRegularization::new(
            1.0,
            ScaleRegMode::Combined {
                volume_weight: 1.0,
                anisotropy_weight: -0.5,
            },
        );
        assert!(result2.is_err());
    }

    // ── Bonus: RegBreakdown format contains key fields ────────────────────────

    #[test]
    fn test_breakdown_format() {
        let bd = RegBreakdown {
            l2_loss: 0.1,
            opacity_loss: 0.2,
            scale_loss: 0.05,
            positional_loss: 0.0,
            total: 0.35,
        };
        let s = bd.format();
        assert!(s.contains("l2"), "format should mention l2: {s}");
        assert!(s.contains("opacity"), "format should mention opacity: {s}");
        assert!(s.contains("scale"), "format should mention scale: {s}");
        assert!(s.contains("total"), "format should mention total: {s}");
    }
}
