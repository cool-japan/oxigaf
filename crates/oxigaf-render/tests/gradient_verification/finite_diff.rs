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
#[allow(dead_code)]
struct GradientContext<'a> {
    model: &'a GaussianModel,
    rasterizer: &'a CpuRasterizer,
    camera: &'a CpuCamera,
    target: &'a [f32],
    loss_fn: &'a dyn LossFunction,
    base_loss: f32,
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

        // RGB-only MSE: skip alpha channel (every 4th element)
        // This matches the GPU backward pass which only processes RGB
        let num_pixels = color_data.len() / 4;
        let rgb_count = num_pixels * 3;
        let mse: f32 = color_data
            .chunks(4)
            .zip(target.chunks(4))
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

    // Compute base loss
    let output = rasterizer.render(model, camera)?;
    let base_loss = loss_fn.compute_loss(&output.color_data, target)?;

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
                    base_loss,
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
                    base_loss,
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

    for (axis, grad_elem) in grad.iter_mut().enumerate() {
        // Perturb position forward: f(x + ε)
        let mut perturbed_plus = ctx.model.clone();
        perturbed_plus.gaussians[ctx.gaussian_idx].position[axis] += ctx.epsilon;
        let output_plus = ctx.rasterizer.render(&perturbed_plus, ctx.camera)?;
        let loss_plus = ctx
            .loss_fn
            .compute_loss(&output_plus.color_data, ctx.target)?;

        // Perturb position backward: f(x - ε)
        let mut perturbed_minus = ctx.model.clone();
        perturbed_minus.gaussians[ctx.gaussian_idx].position[axis] -= ctx.epsilon;
        let output_minus = ctx.rasterizer.render(&perturbed_minus, ctx.camera)?;
        let loss_minus = ctx
            .loss_fn
            .compute_loss(&output_minus.color_data, ctx.target)?;

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

    let output = rasterizer.render(model, camera)?;
    let base_loss = loss_fn.compute_loss(&output.color_data, target)?;

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
                    base_loss,
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
                    base_loss,
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

    for (axis, grad_elem) in grad.iter_mut().enumerate() {
        // Do NOT normalize quaternion after perturbation.
        // The GPU's quat_to_mat() uses the raw quaternion without normalization,
        // so the numerical gradient must match by also using raw quaternions.

        // Perturb rotation forward: f(q + ε)
        let mut perturbed_plus = ctx.model.clone();
        perturbed_plus.gaussians[ctx.gaussian_idx].rotation[axis] += ctx.epsilon;
        let output_plus = ctx.rasterizer.render(&perturbed_plus, ctx.camera)?;
        let loss_plus = ctx
            .loss_fn
            .compute_loss(&output_plus.color_data, ctx.target)?;

        // Perturb rotation backward: f(q - ε)
        let mut perturbed_minus = ctx.model.clone();
        perturbed_minus.gaussians[ctx.gaussian_idx].rotation[axis] -= ctx.epsilon;
        let output_minus = ctx.rasterizer.render(&perturbed_minus, ctx.camera)?;
        let loss_minus = ctx
            .loss_fn
            .compute_loss(&output_minus.color_data, ctx.target)?;

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

    let output = rasterizer.render(model, camera)?;
    let base_loss = loss_fn.compute_loss(&output.color_data, target)?;

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
                    base_loss,
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
                    base_loss,
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

    for (axis, grad_elem) in grad.iter_mut().enumerate() {
        // Perturb scale forward: f(s + ε)
        let mut perturbed_plus = ctx.model.clone();
        perturbed_plus.gaussians[ctx.gaussian_idx].scale[axis] += ctx.epsilon;
        let output_plus = ctx.rasterizer.render(&perturbed_plus, ctx.camera)?;
        let loss_plus = ctx
            .loss_fn
            .compute_loss(&output_plus.color_data, ctx.target)?;

        // Perturb scale backward: f(s - ε)
        let mut perturbed_minus = ctx.model.clone();
        perturbed_minus.gaussians[ctx.gaussian_idx].scale[axis] -= ctx.epsilon;
        let output_minus = ctx.rasterizer.render(&perturbed_minus, ctx.camera)?;
        let loss_minus = ctx
            .loss_fn
            .compute_loss(&output_minus.color_data, ctx.target)?;

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

    let output = rasterizer.render(model, camera)?;
    let base_loss = loss_fn.compute_loss(&output.color_data, target)?;

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
                    base_loss,
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
                    base_loss,
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
    // Perturb opacity forward: f(o + ε)
    let mut perturbed_plus = ctx.model.clone();
    perturbed_plus.gaussians[ctx.gaussian_idx].opacity += ctx.epsilon;
    let output_plus = ctx.rasterizer.render(&perturbed_plus, ctx.camera)?;
    let loss_plus = ctx
        .loss_fn
        .compute_loss(&output_plus.color_data, ctx.target)?;

    // Perturb opacity backward: f(o - ε)
    let mut perturbed_minus = ctx.model.clone();
    perturbed_minus.gaussians[ctx.gaussian_idx].opacity -= ctx.epsilon;
    let output_minus = ctx.rasterizer.render(&perturbed_minus, ctx.camera)?;
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
    // Perturb SH coefficient forward: f(c + ε)
    let mut perturbed_plus = ctx.model.clone();
    perturbed_plus.sh_coeffs[ctx.coeff_idx] += ctx.epsilon;
    let output_plus = ctx.rasterizer.render(&perturbed_plus, ctx.camera)?;
    let loss_plus = ctx
        .loss_fn
        .compute_loss(&output_plus.color_data, ctx.target)?;

    // Perturb SH coefficient backward: f(c - ε)
    let mut perturbed_minus = ctx.model.clone();
    perturbed_minus.sh_coeffs[ctx.coeff_idx] -= ctx.epsilon;
    let output_minus = ctx.rasterizer.render(&perturbed_minus, ctx.camera)?;
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
#[allow(dead_code)]
pub fn max_relative_error(analytical: &[f32], numerical: &[f32]) -> f32 {
    analytical
        .iter()
        .zip(numerical.iter())
        .map(|(a, n)| compute_relative_error(*a, *n))
        .fold(0.0f32, f32::max)
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
}
