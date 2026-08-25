//! Finite-difference gradient computation for gradient verification.
//!
//! This module implements numerical gradient approximation using central-difference:
//! `∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)`
//!
//! The numerical gradients serve as ground truth for verifying analytical gradients
//! computed by the backward pass.

use rayon::prelude::*;

use oxigaf_render::config::RasterConfig;
use oxigaf_render::gaussian::GaussianModel;
use oxigaf_render::{CpuCamera, CpuRasterizer, RenderError};

/// Context for computing gradients for a single Gaussian.
struct GradientContext<'a> {
    model: &'a GaussianModel,
    rasterizer: &'a CpuRasterizer,
    camera: &'a CpuCamera,
    target: &'a [f32],
    loss_fn: &'a dyn LossFunction,
    gaussian_idx: usize,
    epsilon: f32,
}

/// Default epsilon for finite-difference approximation.
/// Using 5e-3 for better f32 precision balance between truncation and rounding error.
const DEFAULT_EPSILON: f32 = 5e-3;

/// Loss function for gradient computation.
pub trait LossFunction: Send + Sync {
    /// Compute loss value given rendered output.
    fn compute_loss(&self, color_data: &[f32], target: &[f32]) -> Result<f32, RenderError>;
}

/// Mean Squared Error (MSE) loss.
pub struct MseLoss;

impl LossFunction for MseLoss {
    fn compute_loss(&self, color_data: &[f32], target: &[f32]) -> Result<f32, RenderError> {
        if color_data.len() != target.len() {
            return Err(RenderError::MismatchedBufferSizes {
                expected: target.len(),
                actual: color_data.len(),
            });
        }
        // `chunks(4)` (below `chunks_exact`) yields a short final chunk
        // when the length isn't a multiple of 4, and indexing `c[1]`/`c[2]`
        // on it would panic; an empty buffer would additionally divide by
        // zero (`rgb_count == 0`) and silently produce NaN. Reject both
        // up front instead.
        if color_data.is_empty() || !color_data.len().is_multiple_of(4) {
            return Err(RenderError::ValidationError(format!(
                "MseLoss expects a non-empty RGBA buffer (length a multiple of 4), got length {}",
                color_data.len()
            )));
        }

        // RGB-only MSE: skip alpha channel (every 4th element)
        // This matches the GPU backward pass which only processes RGB
        let num_pixels = color_data.len() / 4;
        let rgb_count = num_pixels * 3;
        let mse: f32 = color_data
            .chunks_exact(4)
            .zip(target.chunks_exact(4))
            .map(|(c, t)| (c[0] - t[0]).powi(2) + (c[1] - t[1]).powi(2) + (c[2] - t[2]).powi(2))
            .sum::<f32>()
            / rgb_count as f32;

        Ok(mse)
    }
}

/// Parameters for finite-difference gradient computation.
#[derive(Debug, Clone)]
pub struct FiniteDiffConfig {
    /// Epsilon for finite-difference approximation.
    pub epsilon: f32,
    /// Whether to use parallel computation (rayon).
    pub parallel: bool,
}

impl Default for FiniteDiffConfig {
    fn default() -> Self {
        Self {
            epsilon: DEFAULT_EPSILON,
            parallel: true,
        }
    }
}

/// Compute numerical gradient w.r.t. Gaussian positions using central-difference.
///
/// For each Gaussian position (x, y, z), computes:
/// - ∂L/∂x ≈ (L(x + ε) - L(x - ε)) / (2ε)
/// - ∂L/∂y ≈ (L(y + ε) - L(y - ε)) / (2ε)
/// - ∂L/∂z ≈ (L(z + ε) - L(z - ε)) / (2ε)
pub fn compute_position_gradients(
    model: &GaussianModel,
    config: &RasterConfig,
    camera: &CpuCamera,
    target: &[f32],
    loss_fn: &dyn LossFunction,
    fd_config: &FiniteDiffConfig,
) -> Result<Vec<[f32; 3]>, RenderError> {
    let num_gaussians = model.len();
    let rasterizer = CpuRasterizer::new(config.clone());

    // Compute gradients for each Gaussian
    let gradients = if fd_config.parallel {
        (0..num_gaussians)
            .into_par_iter()
            .map(|i| {
                let ctx = GradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    gaussian_idx: i,
                    epsilon: fd_config.epsilon,
                };
                compute_position_gradient_single(&ctx)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..num_gaussians)
            .map(|i| {
                let ctx = GradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    gaussian_idx: i,
                    epsilon: fd_config.epsilon,
                };
                compute_position_gradient_single(&ctx)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(gradients)
}

/// Compute position gradient for a single Gaussian using central-difference.
fn compute_position_gradient_single(ctx: &GradientContext<'_>) -> Result<[f32; 3], RenderError> {
    let mut grad = [0.0f32; 3];
    // Clone the model once per Gaussian (not once per axis per sign) and
    // reuse it for every perturbation below, restoring the original value
    // after each axis so later axes still start from the unperturbed
    // model. This cuts 2 clones per axis (6 for position) down to 1.
    let mut perturbed = ctx.model.clone();

    for (axis, grad_elem) in grad.iter_mut().enumerate() {
        let original = perturbed.gaussians[ctx.gaussian_idx].position[axis];

        // Perturb position forward: f(x + ε)
        perturbed.gaussians[ctx.gaussian_idx].position[axis] = original + ctx.epsilon;
        let output_plus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
        let loss_plus = ctx
            .loss_fn
            .compute_loss(&output_plus.color_data, ctx.target)?;

        // Perturb position backward: f(x - ε)
        perturbed.gaussians[ctx.gaussian_idx].position[axis] = original - ctx.epsilon;
        let output_minus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
        let loss_minus = ctx
            .loss_fn
            .compute_loss(&output_minus.color_data, ctx.target)?;

        // Restore before perturbing the next axis.
        perturbed.gaussians[ctx.gaussian_idx].position[axis] = original;

        // Central-difference gradient: ∂L/∂x ≈ (L(x + ε) - L(x - ε)) / (2ε)
        *grad_elem = (loss_plus - loss_minus) / (2.0 * ctx.epsilon);
    }

    Ok(grad)
}

/// Compute numerical gradient w.r.t. Gaussian rotations (quaternions).
pub fn compute_rotation_gradients(
    model: &GaussianModel,
    config: &RasterConfig,
    camera: &CpuCamera,
    target: &[f32],
    loss_fn: &dyn LossFunction,
    fd_config: &FiniteDiffConfig,
) -> Result<Vec<[f32; 4]>, RenderError> {
    let num_gaussians = model.len();
    let rasterizer = CpuRasterizer::new(config.clone());

    let gradients = if fd_config.parallel {
        (0..num_gaussians)
            .into_par_iter()
            .map(|i| {
                let ctx = GradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    gaussian_idx: i,
                    epsilon: fd_config.epsilon,
                };
                compute_rotation_gradient_single(&ctx)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..num_gaussians)
            .map(|i| {
                let ctx = GradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    gaussian_idx: i,
                    epsilon: fd_config.epsilon,
                };
                compute_rotation_gradient_single(&ctx)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(gradients)
}

/// Compute rotation gradient for a single Gaussian (quaternion) using central-difference.
fn compute_rotation_gradient_single(ctx: &GradientContext<'_>) -> Result<[f32; 4], RenderError> {
    let mut grad = [0.0f32; 4];
    // See the matching comment in `compute_position_gradient_single`: clone
    // once per Gaussian and reuse across axes instead of twice per axis.
    let mut perturbed = ctx.model.clone();

    for (axis, grad_elem) in grad.iter_mut().enumerate() {
        // Do NOT normalize quaternion after perturbation.
        // The GPU's quat_to_mat() uses the raw quaternion without normalization,
        // so the numerical gradient must match by also using raw quaternions.
        let original = perturbed.gaussians[ctx.gaussian_idx].rotation[axis];

        // Perturb rotation forward: f(q + ε)
        perturbed.gaussians[ctx.gaussian_idx].rotation[axis] = original + ctx.epsilon;
        let output_plus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
        let loss_plus = ctx
            .loss_fn
            .compute_loss(&output_plus.color_data, ctx.target)?;

        // Perturb rotation backward: f(q - ε)
        perturbed.gaussians[ctx.gaussian_idx].rotation[axis] = original - ctx.epsilon;
        let output_minus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
        let loss_minus = ctx
            .loss_fn
            .compute_loss(&output_minus.color_data, ctx.target)?;

        // Restore before perturbing the next axis.
        perturbed.gaussians[ctx.gaussian_idx].rotation[axis] = original;

        // Central-difference gradient: ∂L/∂q ≈ (L(q + ε) - L(q - ε)) / (2ε)
        *grad_elem = (loss_plus - loss_minus) / (2.0 * ctx.epsilon);
    }

    Ok(grad)
}

/// Compute numerical gradient w.r.t. Gaussian scales (log-scale).
pub fn compute_scale_gradients(
    model: &GaussianModel,
    config: &RasterConfig,
    camera: &CpuCamera,
    target: &[f32],
    loss_fn: &dyn LossFunction,
    fd_config: &FiniteDiffConfig,
) -> Result<Vec<[f32; 3]>, RenderError> {
    let num_gaussians = model.len();
    let rasterizer = CpuRasterizer::new(config.clone());

    let gradients = if fd_config.parallel {
        (0..num_gaussians)
            .into_par_iter()
            .map(|i| {
                let ctx = GradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    gaussian_idx: i,
                    epsilon: fd_config.epsilon,
                };
                compute_scale_gradient_single(&ctx)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..num_gaussians)
            .map(|i| {
                let ctx = GradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    gaussian_idx: i,
                    epsilon: fd_config.epsilon,
                };
                compute_scale_gradient_single(&ctx)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(gradients)
}

/// Compute scale gradient for a single Gaussian using central-difference.
fn compute_scale_gradient_single(ctx: &GradientContext<'_>) -> Result<[f32; 3], RenderError> {
    let mut grad = [0.0f32; 3];
    // See the matching comment in `compute_position_gradient_single`: clone
    // once per Gaussian and reuse across axes instead of twice per axis.
    let mut perturbed = ctx.model.clone();

    for (axis, grad_elem) in grad.iter_mut().enumerate() {
        let original = perturbed.gaussians[ctx.gaussian_idx].scale[axis];

        // Perturb scale forward: f(s + ε)
        perturbed.gaussians[ctx.gaussian_idx].scale[axis] = original + ctx.epsilon;
        let output_plus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
        let loss_plus = ctx
            .loss_fn
            .compute_loss(&output_plus.color_data, ctx.target)?;

        // Perturb scale backward: f(s - ε)
        perturbed.gaussians[ctx.gaussian_idx].scale[axis] = original - ctx.epsilon;
        let output_minus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
        let loss_minus = ctx
            .loss_fn
            .compute_loss(&output_minus.color_data, ctx.target)?;

        // Restore before perturbing the next axis.
        perturbed.gaussians[ctx.gaussian_idx].scale[axis] = original;

        // Central-difference gradient: ∂L/∂s ≈ (L(s + ε) - L(s - ε)) / (2ε)
        *grad_elem = (loss_plus - loss_minus) / (2.0 * ctx.epsilon);
    }

    Ok(grad)
}

/// Compute numerical gradient w.r.t. Gaussian opacities (sigmoid-inverse).
pub fn compute_opacity_gradients(
    model: &GaussianModel,
    config: &RasterConfig,
    camera: &CpuCamera,
    target: &[f32],
    loss_fn: &dyn LossFunction,
    fd_config: &FiniteDiffConfig,
) -> Result<Vec<f32>, RenderError> {
    let num_gaussians = model.len();
    let rasterizer = CpuRasterizer::new(config.clone());

    let gradients = if fd_config.parallel {
        (0..num_gaussians)
            .into_par_iter()
            .map(|i| {
                let ctx = GradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    gaussian_idx: i,
                    epsilon: fd_config.epsilon,
                };
                compute_opacity_gradient_single(&ctx)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..num_gaussians)
            .map(|i| {
                let ctx = GradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    gaussian_idx: i,
                    epsilon: fd_config.epsilon,
                };
                compute_opacity_gradient_single(&ctx)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(gradients)
}

/// Compute opacity gradient for a single Gaussian using central-difference.
fn compute_opacity_gradient_single(ctx: &GradientContext<'_>) -> Result<f32, RenderError> {
    // Clone once and reuse for both perturbations (see the matching comment
    // in `compute_position_gradient_single`); no restore needed afterward
    // since this is the only perturbation this context computes.
    let mut perturbed = ctx.model.clone();
    let original = perturbed.gaussians[ctx.gaussian_idx].opacity;

    // Perturb opacity forward: f(o + ε)
    perturbed.gaussians[ctx.gaussian_idx].opacity = original + ctx.epsilon;
    let output_plus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
    let loss_plus = ctx
        .loss_fn
        .compute_loss(&output_plus.color_data, ctx.target)?;

    // Perturb opacity backward: f(o - ε)
    perturbed.gaussians[ctx.gaussian_idx].opacity = original - ctx.epsilon;
    let output_minus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
    let loss_minus = ctx
        .loss_fn
        .compute_loss(&output_minus.color_data, ctx.target)?;

    // Central-difference gradient: ∂L/∂o ≈ (L(o + ε) - L(o - ε)) / (2ε)
    let grad = (loss_plus - loss_minus) / (2.0 * ctx.epsilon);
    Ok(grad)
}

/// Compute numerical gradient w.r.t. SH coefficients.
///
/// For each Gaussian, perturb each SH coefficient and compute the gradient.
/// Returns a flat vector of gradients matching the layout of model.sh_coeffs.
pub fn compute_sh_gradients(
    model: &GaussianModel,
    config: &RasterConfig,
    camera: &CpuCamera,
    target: &[f32],
    loss_fn: &dyn LossFunction,
    fd_config: &FiniteDiffConfig,
) -> Result<Vec<f32>, RenderError> {
    let num_gaussians = model.len();
    let sh_degree = model.sh_degree;
    let coeffs_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
    let total_coeffs = coeffs_per_gaussian * num_gaussians;

    let rasterizer = CpuRasterizer::new(config.clone());

    // Compute gradients for each SH coefficient
    let gradients: Vec<f32> = if fd_config.parallel {
        (0..total_coeffs)
            .into_par_iter()
            .map(|coeff_idx| {
                compute_sh_gradient_single(&ShGradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    coeff_idx,
                    epsilon: fd_config.epsilon,
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..total_coeffs)
            .map(|coeff_idx| {
                compute_sh_gradient_single(&ShGradientContext {
                    model,
                    rasterizer: &rasterizer,
                    camera,
                    target,
                    loss_fn,
                    coeff_idx,
                    epsilon: fd_config.epsilon,
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(gradients)
}

/// Context for computing SH coefficient gradients for a single coefficient.
struct ShGradientContext<'a> {
    model: &'a GaussianModel,
    rasterizer: &'a CpuRasterizer,
    camera: &'a CpuCamera,
    target: &'a [f32],
    loss_fn: &'a dyn LossFunction,
    coeff_idx: usize,
    epsilon: f32,
}

/// Compute SH gradient for a single coefficient using central-difference.
fn compute_sh_gradient_single(ctx: &ShGradientContext<'_>) -> Result<f32, RenderError> {
    // Clone once and reuse for both perturbations (see the matching comment
    // in `compute_position_gradient_single`); no restore needed afterward
    // since this is the only perturbation this context computes.
    let mut perturbed = ctx.model.clone();
    let original = perturbed.sh_coeffs[ctx.coeff_idx];

    // Perturb SH coefficient forward: f(c + ε)
    perturbed.sh_coeffs[ctx.coeff_idx] = original + ctx.epsilon;
    let output_plus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
    let loss_plus = ctx
        .loss_fn
        .compute_loss(&output_plus.color_data, ctx.target)?;

    // Perturb SH coefficient backward: f(c - ε)
    perturbed.sh_coeffs[ctx.coeff_idx] = original - ctx.epsilon;
    let output_minus = ctx.rasterizer.render(&perturbed, ctx.camera)?;
    let loss_minus = ctx
        .loss_fn
        .compute_loss(&output_minus.color_data, ctx.target)?;

    // Central-difference gradient: ∂L/∂c ≈ (L(c + ε) - L(c - ε)) / (2ε)
    let grad = (loss_plus - loss_minus) / (2.0 * ctx.epsilon);
    Ok(grad)
}

/// Compute relative error between analytical and numerical gradients.
///
/// Uses max-based denominator for symmetric relative error:
/// - If both values are near zero (< 1e-6), returns absolute difference.
/// - Otherwise, returns `|a - n| / (max(|a|, |n|) + 1e-8)`.
///
/// This matches the superior `gradient_error` metric from GPU self-consistent tests.
pub fn compute_relative_error(analytical: f32, numerical: f32) -> f32 {
    let diff = (analytical - numerical).abs();
    let max_abs = analytical.abs().max(numerical.abs());
    if max_abs < 1e-6 {
        // Both near zero - use absolute difference
        diff
    } else {
        // Use relative error with max-based denominator
        diff / (max_abs + 1e-8)
    }
}

/// Compute maximum relative error for a batch of gradients.
///
/// Returns `f32::NAN` - which compares `false` against every threshold,
/// so a downstream `max_err < THRESHOLD` assertion fails loudly - when:
///
/// * either slice is empty, or the two have different lengths (nothing, or
///   only a prefix, would actually be compared, and a naive fold over an
///   empty iterator reports a perfect `0.0`); or
/// * any pairwise error is non-finite, e.g. because a backward shader
///   emitted a NaN/Inf gradient. `f32::max` silently *discards* NaN (it
///   returns the non-NaN operand), so the previous `fold(0.0, f32::max)`
///   reported the largest finite error and let a NaN gradient pass.
pub fn max_relative_error(analytical: &[f32], numerical: &[f32]) -> f32 {
    if analytical.is_empty() || analytical.len() != numerical.len() {
        return f32::NAN;
    }

    let mut max = 0.0f32;
    for (a, n) in analytical.iter().zip(numerical.iter()) {
        let err = compute_relative_error(*a, *n);
        if !err.is_finite() {
            return f32::NAN;
        }
        if err > max {
            max = err;
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_error() {
        // Perfect match
        assert_eq!(compute_relative_error(1.0, 1.0), 0.0);

        // ~9.09% error with max-based denominator: |1.1 - 1.0| / (1.1 + 1e-8)
        let err = compute_relative_error(1.1, 1.0);
        let expected = 0.1 / (1.1 + 1e-8);
        assert!((err - expected).abs() < 1e-6);

        // Handle zero numerical gradient: both near zero falls into abs-diff branch
        let err = compute_relative_error(1e-7, 0.0);
        assert!(err.is_finite());
        assert!(err < 1e-6); // Both near zero, returns abs diff

        // One value large, one zero: |0.1 - 0.0| / (0.1 + 1e-8) ~ 1.0
        let err = compute_relative_error(0.1, 0.0);
        assert!(err.is_finite());
        assert!((err - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_relative_error_reports_largest_finite_error() {
        let analytical = [1.0_f32, 2.0, 3.0];
        let numerical = [1.0_f32, 2.0, 3.3];

        let err = max_relative_error(&analytical, &numerical);
        let expected = compute_relative_error(3.0, 3.3);
        assert!(
            (err - expected).abs() < 1e-6,
            "expected {expected}, got {err}"
        );
    }

    /// Regression test: a NaN gradient must not be silently discarded.
    /// `fold(0.0, f32::max)` returns the non-NaN operand for a NaN
    /// comparison, so the pre-fix implementation reported the largest
    /// *finite* error and a `max_err < THRESHOLD` assertion passed even
    /// though a backward shader had emitted a non-finite gradient.
    #[test]
    fn test_max_relative_error_nan_propagates() {
        let analytical = [0.0_f32, f32::NAN, 0.0];
        let numerical = [0.0_f32, 0.0, 0.0];

        let err = max_relative_error(&analytical, &numerical);
        assert!(err.is_nan(), "a non-finite gradient must yield NaN");
        // `partial_cmp` rather than `!(err < 1e-3)`: the point of this test is
        // that NaN is *incomparable*, and spelling it this way keeps clippy's
        // `neg_cmp_op_on_partial_ord` lint satisfied.
        assert!(
            !matches!(err.partial_cmp(&1e-3), Some(std::cmp::Ordering::Less)),
            "NaN must make a `< THRESHOLD` assertion fail, not pass"
        );
    }

    /// Regression test: comparing nothing must not report a perfect `0.0`.
    #[test]
    fn test_max_relative_error_empty_is_not_a_silent_pass() {
        assert!(max_relative_error(&[], &[]).is_nan());
    }

    /// Regression test: `zip` silently truncates to the shorter slice, so a
    /// length mismatch used to compare only a prefix and report a passing
    /// error for gradients that were never checked.
    #[test]
    fn test_max_relative_error_length_mismatch_is_not_a_silent_pass() {
        let analytical = [1.0_f32, 2.0, 3.0];
        let numerical = [1.0_f32];

        assert!(max_relative_error(&analytical, &numerical).is_nan());
    }

    #[test]
    fn test_mse_loss() {
        let loss = MseLoss;
        let color_data = vec![1.0, 2.0, 3.0, 4.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];

        let result = loss.compute_loss(&color_data, &target).ok();
        assert_eq!(result, Some(0.0));

        let target2 = vec![2.0, 3.0, 4.0, 5.0];
        let result2 = loss.compute_loss(&color_data, &target2).ok();
        assert!(result2.is_some());
        assert_eq!(result2, Some(1.0)); // RGB-only MSE = ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = 3 / 3 = 1.0
    }

    #[test]
    fn test_mse_loss_rejects_length_not_multiple_of_four() {
        // Regression test: `chunks(4)` used to yield a short final chunk
        // for a length not a multiple of 4, and indexing `c[1]`/`c[2]` on
        // it would panic instead of returning an error.
        let loss = MseLoss;
        let color_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]; // length 5
        let target = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];

        let result = loss.compute_loss(&color_data, &target);
        assert!(
            matches!(result, Err(RenderError::ValidationError(_))),
            "expected ValidationError, got {result:?}"
        );
    }

    #[test]
    fn test_mse_loss_rejects_empty_buffers() {
        // Regression test: an empty (but equal-length) pair of buffers used
        // to divide `0.0 / 0` (rgb_count == 0), silently producing NaN
        // instead of an error.
        let loss = MseLoss;
        let color_data: Vec<f32> = vec![];
        let target: Vec<f32> = vec![];

        let result = loss.compute_loss(&color_data, &target);
        assert!(
            matches!(result, Err(RenderError::ValidationError(_))),
            "expected ValidationError, got {result:?}"
        );
    }
}
