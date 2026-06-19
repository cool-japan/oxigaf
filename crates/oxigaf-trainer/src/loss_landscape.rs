//! Loss Landscape Analysis
//!
//! Characterizes the geometry of the optimization landscape, measuring
//! sharpness, flatness, and connectivity of loss minima.
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::loss_landscape::{QuadraticLoss, LossEvaluator, SharpnessConfig, compute_sharpness};
//!
//! let loss = QuadraticLoss::identity(4);
//! let params = loss.minimum.clone();
//! let grad = loss.gradient(&params).unwrap();
//! let config = SharpnessConfig::default();
//! let metrics = compute_sharpness(&params, &grad, &loss, &config).unwrap();
//! assert!(metrics.gradient_norm < 1e-6);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG (xorshift64 — no rand crate)
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn xorshift64(s: &mut u64) -> u64 {
    let mut x = *s;
    if x == 0 {
        x = 1;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

/// Box-Muller transform — returns a standard-normal sample.
#[inline(always)]
fn xorshift_randn(s: &mut u64) -> f32 {
    let u1 = (xorshift64(s) >> 11) as f32 / (1u64 << 53) as f32;
    let u2 = (xorshift64(s) >> 11) as f32 / (1u64 << 53) as f32;
    let u1 = u1.max(1e-10_f32);
    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos()
}

/// Sample a uniformly-distributed unit-sphere direction of length `dim`.
fn random_unit_sphere(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut d: Vec<f32> = (0..dim).map(|_| xorshift_randn(&mut s)).collect();
    let norm: f32 = d.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in d.iter_mut() {
            *x /= norm;
        }
    }
    d
}

// ─────────────────────────────────────────────────────────────────────────────
// LandscapeError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the loss landscape subsystem.
#[derive(Debug, Error)]
pub enum LandscapeError {
    #[error("Parameter and gradient lengths differ: {len_p} vs {len_g}")]
    LengthMismatch { len_p: usize, len_g: usize },

    #[error("No parameters provided")]
    EmptyParameters,

    #[error("Invalid perturbation scale {0}: must be positive")]
    InvalidPerturbationScale(f32),

    #[error("Invalid interpolation steps {0}: must be ≥ 2")]
    InvalidInterpolationSteps(usize),

    #[error("Loss evaluator returned NaN/Inf")]
    NonFiniteLoss,

    #[error("Alpha out of range [0, 1]: {0}")]
    AlphaOutOfRange(f32),
}

// ─────────────────────────────────────────────────────────────────────────────
// LossEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates loss at given parameter values (used for landscape sampling).
pub trait LossEvaluator: Send + Sync {
    fn evaluate(&self, params: &[f32]) -> Result<f32, LandscapeError>;
    fn gradient(&self, params: &[f32]) -> Result<Vec<f32>, LandscapeError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// QuadraticLoss
// ─────────────────────────────────────────────────────────────────────────────

/// Simple quadratic loss: L(θ) = Σ a_i · (θ_i − b_i)²
///
/// Useful for testing landscape analysis without GPU.
pub struct QuadraticLoss {
    /// a_i values (curvatures — must be positive)
    pub curvature: Vec<f32>,
    /// b_i values (location of minimum)
    pub minimum: Vec<f32>,
}

impl QuadraticLoss {
    /// Create a new quadratic loss with explicit curvatures and minimum location.
    pub fn new(curvature: Vec<f32>, minimum: Vec<f32>) -> Self {
        Self { curvature, minimum }
    }

    /// Identity curvature (a_i = 1), minimum at origin.
    pub fn identity(dim: usize) -> Self {
        Self {
            curvature: vec![1.0; dim],
            minimum: vec![0.0; dim],
        }
    }

    /// Ill-conditioned: a_i = i + 1 (very different curvatures).
    pub fn ill_conditioned(dim: usize) -> Self {
        Self {
            curvature: (1..=dim).map(|i| i as f32).collect(),
            minimum: vec![0.0; dim],
        }
    }
}

impl LossEvaluator for QuadraticLoss {
    fn evaluate(&self, params: &[f32]) -> Result<f32, LandscapeError> {
        if params.len() != self.curvature.len() {
            return Err(LandscapeError::LengthMismatch {
                len_p: params.len(),
                len_g: self.curvature.len(),
            });
        }
        let loss: f32 = params
            .iter()
            .zip(self.curvature.iter())
            .zip(self.minimum.iter())
            .map(|((p, a), b)| a * (p - b) * (p - b))
            .sum();
        if !loss.is_finite() {
            return Err(LandscapeError::NonFiniteLoss);
        }
        Ok(loss)
    }

    fn gradient(&self, params: &[f32]) -> Result<Vec<f32>, LandscapeError> {
        if params.len() != self.curvature.len() {
            return Err(LandscapeError::LengthMismatch {
                len_p: params.len(),
                len_g: self.curvature.len(),
            });
        }
        let grad: Vec<f32> = params
            .iter()
            .zip(self.curvature.iter())
            .zip(self.minimum.iter())
            .map(|((p, a), b)| 2.0 * a * (p - b))
            .collect();
        Ok(grad)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SharpnessConfig / SharpnessMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for sharpness computation.
#[derive(Debug, Clone)]
pub struct SharpnessConfig {
    /// Perturbation radius ε. Default: 0.05
    pub epsilon: f32,
    /// Number of random perturbation samples for SAM approximation. Default: 10
    pub num_samples: usize,
    /// Seed for reproducible results.
    pub seed: u64,
}

impl Default for SharpnessConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            num_samples: 10,
            seed: 42,
        }
    }
}

/// Sharpness metrics for the current loss landscape.
#[derive(Debug, Clone)]
pub struct SharpnessMetrics {
    /// SAM sharpness: max_k L(θ + ε·d_k) − L(θ)
    pub sam_sharpness: f32,
    /// Adaptive SAM: relative perturbation per parameter group.
    pub adaptive_sam: f32,
    /// Gradient-based curvature estimate: ||g(θ + ε∇̂) − g(θ)|| / ε
    pub curvature_estimate: f32,
    /// Gradient norm (||∇L||₂)
    pub gradient_norm: f32,
    /// Parameter norm (||θ||₂)
    pub parameter_norm: f32,
    /// Epsilon used for perturbation
    pub epsilon: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_sharpness
// ─────────────────────────────────────────────────────────────────────────────

/// Compute sharpness metrics at the current parameter location.
///
/// SAM approximation using random unit-ball samples:
/// 1. Sample n unit-sphere directions d_k
/// 2. Compute perturbed loss at θ + ε · d_k
/// 3. SAM = max_k L(θ + ε·d_k) − L(θ)
///
/// Curvature estimate (Hessian-vector product approximation):
/// g1 = gradient(θ + ε · ∇L/||∇L||)
/// g0 = gradient(θ)
/// curvature ≈ ||g1 − g0|| / ε
pub fn compute_sharpness(
    params: &[f32],
    gradient: &[f32],
    evaluator: &dyn LossEvaluator,
    config: &SharpnessConfig,
) -> Result<SharpnessMetrics, LandscapeError> {
    let dim = params.len();
    if dim == 0 {
        return Err(LandscapeError::EmptyParameters);
    }
    if gradient.len() != dim {
        return Err(LandscapeError::LengthMismatch {
            len_p: dim,
            len_g: gradient.len(),
        });
    }
    if !config.epsilon.is_finite() || config.epsilon <= 0.0 {
        return Err(LandscapeError::InvalidPerturbationScale(config.epsilon));
    }

    let eps = config.epsilon;

    // ── Base loss ────────────────────────────────────────────────────────────
    let base_loss = evaluator.evaluate(params)?;
    if !base_loss.is_finite() {
        return Err(LandscapeError::NonFiniteLoss);
    }

    // ── Gradient norm & parameter norm ───────────────────────────────────────
    let gradient_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();
    let parameter_norm: f32 = params.iter().map(|p| p * p).sum::<f32>().sqrt();

    // ── SAM sharpness — random unit-sphere perturbations ────────────────────
    let mut sam_sharpness: f32 = 0.0;
    for k in 0..config.num_samples {
        let seed = config.seed.wrapping_add(k as u64);
        let dir = random_unit_sphere(dim, seed);
        let perturbed: Vec<f32> = params
            .iter()
            .zip(dir.iter())
            .map(|(p, d)| p + eps * d)
            .collect();
        let p_loss = evaluator.evaluate(&perturbed)?;
        if !p_loss.is_finite() {
            return Err(LandscapeError::NonFiniteLoss);
        }
        let delta = p_loss - base_loss;
        if delta > sam_sharpness {
            sam_sharpness = delta;
        }
    }

    // ── Adaptive SAM — per-parameter scale by |g_i| / ||g|| ─────────────────
    // δ_i = ε · |g_i| / ||g||  (when ||g|| == 0, adaptive perturbation is 0)
    let adaptive_sam = if gradient_norm < 1e-12 {
        0.0
    } else {
        let perturbed: Vec<f32> = params
            .iter()
            .zip(gradient.iter())
            .map(|(p, g)| p + eps * g.abs() / gradient_norm)
            .collect();
        let p_loss = evaluator.evaluate(&perturbed)?;
        if !p_loss.is_finite() {
            return Err(LandscapeError::NonFiniteLoss);
        }
        (p_loss - base_loss).max(0.0)
    };

    // ── Curvature estimate via finite-difference along gradient direction ─────
    // When ||g|| ≈ 0, skip (curvature undefined at flat point here).
    let curvature_estimate = if gradient_norm < 1e-12 {
        0.0
    } else {
        let grad_dir: Vec<f32> = gradient.iter().map(|g| g / gradient_norm).collect();
        let perturbed: Vec<f32> = params
            .iter()
            .zip(grad_dir.iter())
            .map(|(p, d)| p + eps * d)
            .collect();
        let g1 = evaluator.gradient(&perturbed)?;
        let g0 = evaluator.gradient(params)?;
        let diff_norm: f32 = g1
            .iter()
            .zip(g0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        diff_norm / eps
    };

    Ok(SharpnessMetrics {
        sam_sharpness,
        adaptive_sam,
        curvature_estimate,
        gradient_norm,
        parameter_norm,
        epsilon: eps,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// LandscapeScan1D
// ─────────────────────────────────────────────────────────────────────────────

/// Result of scanning the loss landscape along a 1-D direction.
#[derive(Debug, Clone)]
pub struct LandscapeScan1D {
    /// Parameter offsets along the scan direction (relative to center).
    pub offsets: Vec<f32>,
    /// Loss values at each offset.
    pub losses: Vec<f32>,
    /// Gradient norms at each point (populated when `compute_gradients` is true).
    pub gradient_norms: Option<Vec<f32>>,
    /// Direction of the scan (unit vector).
    pub direction: Vec<f32>,
}

impl LandscapeScan1D {
    /// Find the minimum loss value and offset.
    pub fn min_loss(&self) -> (f32, f32) {
        self.losses.iter().zip(self.offsets.iter()).fold(
            (f32::INFINITY, 0.0),
            |(best_l, best_o), (&l, &o)| {
                if l < best_l {
                    (l, o)
                } else {
                    (best_l, best_o)
                }
            },
        )
    }

    /// Find the maximum loss value and offset.
    pub fn max_loss(&self) -> (f32, f32) {
        self.losses.iter().zip(self.offsets.iter()).fold(
            (f32::NEG_INFINITY, 0.0),
            |(best_l, best_o), (&l, &o)| {
                if l > best_l {
                    (l, o)
                } else {
                    (best_l, best_o)
                }
            },
        )
    }

    /// Loss variation: max − min.
    pub fn loss_variation(&self) -> f32 {
        let (min_l, _) = self.min_loss();
        let (max_l, _) = self.max_loss();
        max_l - min_l
    }

    /// Estimate local convexity by fitting a quadratic ax² + bx + c to the
    /// (offset, loss) pairs via least squares. Returns `true` if a > 0.
    pub fn is_locally_convex(&self) -> bool {
        let n = self.offsets.len();
        if n < 3 {
            return false;
        }
        // Build normal equations for [a, b, c] in ax² + bx + c
        // using Σ x^k and Σ x^k * y.
        let mut sx4 = 0.0_f64;
        let mut sx3 = 0.0_f64;
        let mut sx2 = 0.0_f64;
        let mut sx1 = 0.0_f64;
        let sx0 = n as f64;
        let mut sy_x2 = 0.0_f64;
        let mut sy_x1 = 0.0_f64;
        let mut sy_x0 = 0.0_f64;

        for (&x, &y) in self.offsets.iter().zip(self.losses.iter()) {
            let xd = x as f64;
            let yd = y as f64;
            let x2 = xd * xd;
            let x3 = x2 * xd;
            let x4 = x2 * x2;
            sx4 += x4;
            sx3 += x3;
            sx2 += x2;
            sx1 += xd;
            sy_x2 += yd * x2;
            sy_x1 += yd * xd;
            sy_x0 += yd;
        }
        // 3x3 system: M · [a, b, c]^T = r
        // [sx4 sx3 sx2] [a]   [sy_x2]
        // [sx3 sx2 sx1] [b] = [sy_x1]
        // [sx2 sx1 sx0] [c]   [sy_x0]
        let _ = sx0; // only used implicitly via sx0 alias
        let m = [[sx4, sx3, sx2], [sx3, sx2, sx1], [sx2, sx1, n as f64]];
        let r = [sy_x2, sy_x1, sy_x0];

        // Gaussian elimination (3x3)
        match solve_3x3(m, r) {
            Some(coeffs) => coeffs[0] > 0.0,
            None => false,
        }
    }

    /// Format the scan as a simple ASCII plot.
    /// Uses `'*'` for the loss curve and `'.'` for empty cells.
    pub fn format_ascii(&self, width: usize, height: usize) -> String {
        if self.losses.is_empty() || width == 0 || height == 0 {
            return String::new();
        }

        let (min_l, _) = self.min_loss();
        let (max_l, _) = self.max_loss();
        let loss_range = max_l - min_l;

        // Build a width × height grid (row 0 = top = high loss)
        let mut grid = vec![vec!['.'; width]; height];

        for (col, &l) in self.losses.iter().enumerate().take(width) {
            let row = if loss_range < 1e-12 {
                // All losses equal — place in the middle row
                height / 2
            } else {
                // Row 0 is the top (high loss), row height-1 is bottom (low loss)
                let frac = (l - min_l) / loss_range; // 0 = low, 1 = high
                let row_f = (1.0 - frac) * (height - 1) as f32;
                row_f.round().clamp(0.0, (height - 1) as f32) as usize
            };
            grid[row][col] = '*';
        }

        // Render into a string (one line per row, newline-separated)
        let lines: Vec<String> = grid
            .into_iter()
            .map(|row| row.into_iter().collect::<String>())
            .collect();
        lines.join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: 3×3 Gaussian elimination
// ─────────────────────────────────────────────────────────────────────────────

fn solve_3x3(m: [[f64; 3]; 3], r: [f64; 3]) -> Option<[f64; 3]> {
    let mut a = [
        [m[0][0], m[0][1], m[0][2], r[0]],
        [m[1][0], m[1][1], m[1][2], r[1]],
        [m[2][0], m[2][1], m[2][2], r[2]],
    ];

    for col in 0..3 {
        // Partial pivot
        let mut max_row = col;
        for row in (col + 1)..3 {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }
        a.swap(col, max_row);
        let pivot = a[col][col];
        if pivot.abs() < 1e-14 {
            return None;
        }
        for row in (col + 1)..3 {
            let factor = a[row][col] / pivot;
            let col_row = a[col];
            for (&col_k, row_k) in col_row[col..].iter().zip(a[row][col..].iter_mut()) {
                *row_k -= factor * col_k;
            }
        }
    }

    // Back substitution
    let mut x = [0.0_f64; 3];
    for i in (0..3).rev() {
        x[i] = a[i][3];
        for j in (i + 1)..3 {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
    Some(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// scan_1d
// ─────────────────────────────────────────────────────────────────────────────

/// Scan the loss landscape along a given direction.
///
/// Evaluates the loss (and optionally gradients) at `num_steps` evenly-spaced
/// offsets in `[−range, +range]` along the unit-normalised `direction`.
pub fn scan_1d(
    center_params: &[f32],
    direction: &[f32],
    evaluator: &dyn LossEvaluator,
    range: f32,
    num_steps: usize,
    compute_gradients: bool,
) -> Result<LandscapeScan1D, LandscapeError> {
    let dim = center_params.len();
    if dim == 0 {
        return Err(LandscapeError::EmptyParameters);
    }
    if direction.len() != dim {
        return Err(LandscapeError::LengthMismatch {
            len_p: dim,
            len_g: direction.len(),
        });
    }
    if num_steps < 2 {
        return Err(LandscapeError::InvalidInterpolationSteps(num_steps));
    }

    // Normalise direction
    let dir_norm: f32 = direction.iter().map(|d| d * d).sum::<f32>().sqrt();
    let unit_dir: Vec<f32> = if dir_norm < 1e-12 {
        // Degenerate direction — use canonical first axis
        let mut v = vec![0.0_f32; dim];
        v[0] = 1.0;
        v
    } else {
        direction.iter().map(|d| d / dir_norm).collect()
    };

    // Linspace from -range to +range
    let offsets: Vec<f32> = (0..num_steps)
        .map(|i| -range + 2.0 * range * (i as f32 / (num_steps - 1) as f32))
        .collect();

    let mut losses = Vec::with_capacity(num_steps);
    let mut grad_norms = if compute_gradients {
        Some(Vec::with_capacity(num_steps))
    } else {
        None
    };

    for &offset in &offsets {
        let p: Vec<f32> = center_params
            .iter()
            .zip(unit_dir.iter())
            .map(|(c, d)| c + offset * d)
            .collect();
        let l = evaluator.evaluate(&p)?;
        if !l.is_finite() {
            return Err(LandscapeError::NonFiniteLoss);
        }
        losses.push(l);

        if let Some(ref mut gn) = grad_norms {
            let g = evaluator.gradient(&p)?;
            let norm: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
            gn.push(norm);
        }
    }

    Ok(LandscapeScan1D {
        offsets,
        losses,
        gradient_norms: grad_norms,
        direction: unit_dir,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// InterpolationResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of interpolating between two parameter sets.
#[derive(Debug, Clone)]
pub struct InterpolationResult {
    /// Interpolation coefficients in [0, 1].
    pub alphas: Vec<f32>,
    /// Loss at each alpha.
    pub losses: Vec<f32>,
    /// Maximum loss above the linear interpolation of endpoints.
    /// Positive ⇒ barrier present; ≤ 0 ⇒ path is (weakly) convex.
    pub barrier_height: f32,
    /// Whether the path is convex (loss stays at or below linear interpolation).
    pub is_convex: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// interpolate_params
// ─────────────────────────────────────────────────────────────────────────────

/// Interpolate between `params_a` and `params_b`:
///   θ(α) = (1−α)·params_a + α·params_b
///
/// Evaluates the loss at `num_steps` evenly-spaced α values in [0, 1].
pub fn interpolate_params(
    params_a: &[f32],
    params_b: &[f32],
    evaluator: &dyn LossEvaluator,
    num_steps: usize,
) -> Result<InterpolationResult, LandscapeError> {
    if params_a.is_empty() {
        return Err(LandscapeError::EmptyParameters);
    }
    if params_a.len() != params_b.len() {
        return Err(LandscapeError::LengthMismatch {
            len_p: params_a.len(),
            len_g: params_b.len(),
        });
    }
    if num_steps < 2 {
        return Err(LandscapeError::InvalidInterpolationSteps(num_steps));
    }

    let alphas: Vec<f32> = (0..num_steps)
        .map(|i| i as f32 / (num_steps - 1) as f32)
        .collect();

    let mut losses = Vec::with_capacity(num_steps);
    for &alpha in &alphas {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(LandscapeError::AlphaOutOfRange(alpha));
        }
        let p: Vec<f32> = params_a
            .iter()
            .zip(params_b.iter())
            .map(|(a, b)| (1.0 - alpha) * a + alpha * b)
            .collect();
        let l = evaluator.evaluate(&p)?;
        if !l.is_finite() {
            return Err(LandscapeError::NonFiniteLoss);
        }
        losses.push(l);
    }

    let loss_a = losses[0];
    let loss_b = *losses.last().unwrap_or(&loss_a);

    // Barrier: max over all steps of (actual_loss − linear_interpolation)
    let mut barrier_height = f32::NEG_INFINITY;
    for (i, (&alpha, &l)) in alphas.iter().zip(losses.iter()).enumerate() {
        let _ = i;
        let linear_interp = (1.0 - alpha) * loss_a + alpha * loss_b;
        let diff = l - linear_interp;
        if diff > barrier_height {
            barrier_height = diff;
        }
    }
    if barrier_height == f32::NEG_INFINITY {
        barrier_height = 0.0;
    }

    let is_convex = barrier_height <= 1e-6;

    Ok(InterpolationResult {
        alphas,
        losses,
        barrier_height,
        is_convex,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// LandscapeStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics of the loss landscape at a given parameter location.
#[derive(Debug, Clone)]
pub struct LandscapeStats {
    pub base_loss: f32,
    pub sharpness: SharpnessMetrics,
    pub gradient_norm: f32,
    /// Effective dimensionality estimate: ||g||² / (gᵀ H g)
    /// where H is approximated via a finite-difference Jacobian.
    /// Falls back to `params.len()` when gradient is zero.
    pub effective_dim: f32,
    /// Losses at random perturbations within the ε-ball.
    pub loss_at_grid: Vec<f32>,
    pub mean_perturbed_loss: f32,
    pub std_perturbed_loss: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_landscape_stats
// ─────────────────────────────────────────────────────────────────────────────

/// Compute comprehensive landscape statistics at the given location.
pub fn compute_landscape_stats(
    params: &[f32],
    gradient: &[f32],
    evaluator: &dyn LossEvaluator,
    config: &SharpnessConfig,
) -> Result<LandscapeStats, LandscapeError> {
    let dim = params.len();
    if dim == 0 {
        return Err(LandscapeError::EmptyParameters);
    }
    if gradient.len() != dim {
        return Err(LandscapeError::LengthMismatch {
            len_p: dim,
            len_g: gradient.len(),
        });
    }

    let base_loss = evaluator.evaluate(params)?;
    if !base_loss.is_finite() {
        return Err(LandscapeError::NonFiniteLoss);
    }

    let sharpness = compute_sharpness(params, gradient, evaluator, config)?;
    let gradient_norm = sharpness.gradient_norm;

    // ── Effective dimensionality: ||g||² / (gᵀ H g) ─────────────────────────
    // H·g ≈ (∇L(θ + ε·g/||g||) − ∇L(θ)) / ε  (finite diff along g direction)
    let effective_dim = if gradient_norm < 1e-12 {
        // Gradient is zero — landscape is flat; return parameter count as proxy.
        dim as f32
    } else {
        let eps = config.epsilon;
        let grad_dir: Vec<f32> = gradient.iter().map(|g| g / gradient_norm).collect();
        let p_shifted: Vec<f32> = params
            .iter()
            .zip(grad_dir.iter())
            .map(|(p, d)| p + eps * d)
            .collect();
        let g1 = evaluator.gradient(&p_shifted)?;
        let g0 = evaluator.gradient(params)?;
        // Hg ≈ (g1 - g0) / eps
        let hg: Vec<f32> = g1
            .iter()
            .zip(g0.iter())
            .map(|(a, b)| (a - b) / eps)
            .collect();
        // gᵀHg
        let g_t_hg: f32 = gradient.iter().zip(hg.iter()).map(|(g, h)| g * h).sum();
        let g_sq: f32 = gradient_norm * gradient_norm;
        if g_t_hg.abs() < 1e-12 {
            dim as f32
        } else {
            (g_sq / g_t_hg).max(1.0)
        }
    };

    // ── Random perturbation sample losses ────────────────────────────────────
    let num_samples = config.num_samples;
    let eps = config.epsilon;
    let mut loss_at_grid = Vec::with_capacity(num_samples);
    for k in 0..num_samples {
        let seed = config.seed.wrapping_add(k as u64).wrapping_add(0xDEAD_BEEF);
        let dir = random_unit_sphere(dim, seed);
        let perturbed: Vec<f32> = params
            .iter()
            .zip(dir.iter())
            .map(|(p, d)| p + eps * d)
            .collect();
        let l = evaluator.evaluate(&perturbed)?;
        if !l.is_finite() {
            return Err(LandscapeError::NonFiniteLoss);
        }
        loss_at_grid.push(l);
    }

    let mean_perturbed_loss = if loss_at_grid.is_empty() {
        base_loss
    } else {
        loss_at_grid.iter().sum::<f32>() / loss_at_grid.len() as f32
    };

    let std_perturbed_loss = if loss_at_grid.len() < 2 {
        0.0
    } else {
        let var = loss_at_grid
            .iter()
            .map(|l| {
                let d = l - mean_perturbed_loss;
                d * d
            })
            .sum::<f32>()
            / loss_at_grid.len() as f32;
        var.sqrt()
    };

    Ok(LandscapeStats {
        base_loss,
        sharpness,
        gradient_norm,
        effective_dim,
        loss_at_grid,
        mean_perturbed_loss,
        std_perturbed_loss,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_at_minimum() -> (QuadraticLoss, Vec<f32>, Vec<f32>) {
        let dim = 4;
        let loss = QuadraticLoss::identity(dim);
        let params = loss.minimum.clone();
        let grad = loss.gradient(&params).unwrap();
        (loss, params, grad)
    }

    // ── Test 1: evaluate at minimum → 0 ─────────────────────────────────────
    #[test]
    fn test_quadratic_evaluate_at_minimum() {
        let loss = QuadraticLoss::identity(4);
        let result = loss.evaluate(&loss.minimum.clone()).unwrap();
        assert!(
            (result - 0.0).abs() < 1e-6,
            "Expected 0 at minimum, got {}",
            result
        );
    }

    // ── Test 2: evaluate away from minimum → positive ────────────────────────
    #[test]
    fn test_quadratic_evaluate_away_from_minimum() {
        let loss = QuadraticLoss::identity(4);
        let params = vec![1.0, 0.0, 0.0, 0.0];
        let result = loss.evaluate(&params).unwrap();
        assert!(
            result > 0.0,
            "Expected positive loss away from minimum, got {}",
            result
        );
    }

    // ── Test 3: gradient at minimum → zero vector ────────────────────────────
    #[test]
    fn test_quadratic_gradient_at_minimum() {
        let loss = QuadraticLoss::identity(4);
        let grad = loss.gradient(&loss.minimum.clone()).unwrap();
        for g in &grad {
            assert!(g.abs() < 1e-6, "Expected ~0 gradient at minimum, got {}", g);
        }
    }

    // ── Test 4: gradient linear in displacement ──────────────────────────────
    #[test]
    fn test_quadratic_gradient_linear_in_displacement() {
        let loss = QuadraticLoss::identity(4);
        let p1 = vec![1.0, 0.0, 0.0, 0.0];
        let p2 = vec![2.0, 0.0, 0.0, 0.0];
        let g1 = loss.gradient(&p1).unwrap();
        let g2 = loss.gradient(&p2).unwrap();
        // g_0 at p2 should be twice g_0 at p1
        assert!((g2[0] - 2.0 * g1[0]).abs() < 1e-5);
    }

    // ── Test 5: SAM ≈ ε² at identity quadratic minimum ──────────────────────
    #[test]
    fn test_compute_sharpness_sam_at_identity_minimum() {
        let (loss, params, grad) = make_identity_at_minimum();
        let config = SharpnessConfig {
            epsilon: 0.05,
            num_samples: 50,
            seed: 42,
        };
        let metrics = compute_sharpness(&params, &grad, &loss, &config).unwrap();
        let eps_sq = config.epsilon * config.epsilon;
        // For unit-sphere direction d on identity quadratic: L(ε·d) = ε²
        // With 50 samples we should be very close to eps_sq.
        assert!(
            (metrics.sam_sharpness - eps_sq).abs() < 0.5 * eps_sq,
            "SAM={} expected ≈ ε²={}",
            metrics.sam_sharpness,
            eps_sq
        );
    }

    // ── Test 6: length mismatch → LengthMismatch error ──────────────────────
    #[test]
    fn test_compute_sharpness_length_mismatch() {
        let loss = QuadraticLoss::identity(4);
        let params = vec![0.0; 4];
        let gradient = vec![0.0; 3]; // wrong length
        let config = SharpnessConfig::default();
        let result = compute_sharpness(&params, &gradient, &loss, &config);
        assert!(matches!(result, Err(LandscapeError::LengthMismatch { .. })));
    }

    // ── Test 7: scan_1d output lengths match num_steps ───────────────────────
    #[test]
    fn test_scan_1d_output_lengths() {
        let loss = QuadraticLoss::identity(4);
        let center = vec![0.0_f32; 4];
        let direction = vec![1.0, 0.0, 0.0, 0.0];
        let scan = scan_1d(&center, &direction, &loss, 1.0, 20, false).unwrap();
        assert_eq!(scan.offsets.len(), 20);
        assert_eq!(scan.losses.len(), 20);
        assert!(scan.gradient_norms.is_none());
    }

    // ── Test 8: center is minimum for identity quadratic ─────────────────────
    #[test]
    fn test_scan_1d_center_is_minimum() {
        let loss = QuadraticLoss::identity(4);
        let center = loss.minimum.clone();
        let direction = vec![1.0, 0.0, 0.0, 0.0];
        let scan = scan_1d(&center, &direction, &loss, 1.0, 21, false).unwrap();
        // Middle element (offset 0) should have loss 0
        let mid = 10; // index 10 of 21 steps
        assert!(
            scan.losses[mid] < 1e-6,
            "Center loss should be 0, got {}",
            scan.losses[mid]
        );
    }

    // ── Test 9: is_locally_convex for convex quadratic ───────────────────────
    #[test]
    fn test_scan_1d_is_locally_convex() {
        let loss = QuadraticLoss::identity(4);
        let center = loss.minimum.clone();
        let direction = vec![1.0, 0.0, 0.0, 0.0];
        let scan = scan_1d(&center, &direction, &loss, 1.0, 21, false).unwrap();
        assert!(
            scan.is_locally_convex(),
            "Expected convex scan for quadratic"
        );
    }

    // ── Test 10: min_loss finds 0 at center ──────────────────────────────────
    #[test]
    fn test_scan_1d_min_loss_at_center() {
        let loss = QuadraticLoss::identity(4);
        let center = loss.minimum.clone();
        let direction = vec![1.0, 0.0, 0.0, 0.0];
        let scan = scan_1d(&center, &direction, &loss, 1.0, 21, false).unwrap();
        let (min_l, min_o) = scan.min_loss();
        assert!(min_l < 1e-6, "Min loss should be ~0, got {}", min_l);
        assert!(min_o.abs() < 1e-5, "Min offset should be ~0, got {}", min_o);
    }

    // ── Test 11: loss_variation > 0 for non-trivial scan ─────────────────────
    #[test]
    fn test_scan_1d_loss_variation_positive() {
        let loss = QuadraticLoss::identity(4);
        let center = loss.minimum.clone();
        let direction = vec![1.0, 0.0, 0.0, 0.0];
        let scan = scan_1d(&center, &direction, &loss, 1.0, 21, false).unwrap();
        assert!(scan.loss_variation() > 0.0, "Expected non-zero variation");
    }

    // ── Test 12: interpolate_params returns num_steps alpha values ───────────
    #[test]
    fn test_interpolate_params_length() {
        let loss = QuadraticLoss::identity(4);
        let a = vec![0.0_f32; 4];
        let b = vec![1.0_f32; 4];
        let result = interpolate_params(&a, &b, &loss, 10).unwrap();
        assert_eq!(result.alphas.len(), 10);
        assert_eq!(result.losses.len(), 10);
    }

    // ── Test 13: alpha[0] = 0, alpha[-1] = 1 ─────────────────────────────────
    #[test]
    fn test_interpolate_params_alpha_endpoints() {
        let loss = QuadraticLoss::identity(4);
        let a = vec![0.0_f32; 4];
        let b = vec![1.0_f32; 4];
        let result = interpolate_params(&a, &b, &loss, 10).unwrap();
        assert!((result.alphas[0] - 0.0).abs() < 1e-6);
        assert!((result.alphas[9] - 1.0).abs() < 1e-6);
    }

    // ── Test 14: same params → flat loss profile ──────────────────────────────
    #[test]
    fn test_interpolate_params_same_params_flat() {
        let loss = QuadraticLoss::identity(4);
        let a = vec![0.5_f32; 4];
        let result = interpolate_params(&a, &a, &loss, 10).unwrap();
        let first = result.losses[0];
        for &l in &result.losses {
            assert!((l - first).abs() < 1e-5, "Expected flat profile, got {}", l);
        }
    }

    // ── Test 15: is_convex for quadratic between two points ──────────────────
    #[test]
    fn test_interpolate_params_is_convex_quadratic() {
        let loss = QuadraticLoss::identity(4);
        let a = vec![-1.0_f32; 4];
        let b = vec![1.0_f32; 4];
        let result = interpolate_params(&a, &b, &loss, 20).unwrap();
        assert!(
            result.is_convex,
            "Quadratic interpolation path should be convex"
        );
    }

    // ── Test 16: compute_landscape_stats returns valid stats ─────────────────
    #[test]
    fn test_compute_landscape_stats_valid() {
        let (loss, params, grad) = make_identity_at_minimum();
        let config = SharpnessConfig::default();
        let stats = compute_landscape_stats(&params, &grad, &loss, &config).unwrap();
        assert!(stats.base_loss >= 0.0);
        assert!(stats.mean_perturbed_loss.is_finite());
        assert!(stats.std_perturbed_loss >= 0.0);
        assert!(stats.effective_dim >= 1.0);
    }

    // ── Test 17: base_loss matches evaluator ─────────────────────────────────
    #[test]
    fn test_compute_landscape_stats_base_loss_matches() {
        let (loss, params, grad) = make_identity_at_minimum();
        let config = SharpnessConfig::default();
        let stats = compute_landscape_stats(&params, &grad, &loss, &config).unwrap();
        let direct = loss.evaluate(&params).unwrap();
        assert!((stats.base_loss - direct).abs() < 1e-6);
    }

    // ── Test 18: format_ascii returns correct number of lines ────────────────
    #[test]
    fn test_scan_1d_format_ascii_line_count() {
        let loss = QuadraticLoss::identity(4);
        let center = vec![0.0_f32; 4];
        let direction = vec![1.0, 0.0, 0.0, 0.0];
        let scan = scan_1d(&center, &direction, &loss, 1.0, 40, false).unwrap();
        let ascii = scan.format_ascii(40, 10);
        let lines: Vec<&str> = ascii.lines().collect();
        assert_eq!(lines.len(), 10, "Expected 10 lines, got {}", lines.len());
    }

    // ── Test 19: SharpnessConfig::default has valid epsilon/num_samples ───────
    #[test]
    fn test_sharpness_config_default_valid() {
        let config = SharpnessConfig::default();
        assert!(config.epsilon > 0.0);
        assert!(config.num_samples > 0);
    }

    // ── Test 20: gradient_norm ≈ 0 at minimum ────────────────────────────────
    #[test]
    fn test_compute_sharpness_gradient_norm_at_minimum() {
        let (loss, params, grad) = make_identity_at_minimum();
        let config = SharpnessConfig::default();
        let metrics = compute_sharpness(&params, &grad, &loss, &config).unwrap();
        assert!(
            metrics.gradient_norm < 1e-6,
            "Expected gradient_norm ≈ 0 at minimum, got {}",
            metrics.gradient_norm
        );
    }
}
