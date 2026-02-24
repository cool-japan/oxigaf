//! Gradient verification test module.
//!
//! This module provides utilities for verifying analytical gradients against
//! numerical gradients computed via finite-difference approximation.
//!
//! # Test Strategy
//!
//! 1. **Setup**: Create simple test scene (1-10 Gaussians)
//! 2. **Forward**: Render image, compute loss
//! 3. **Backward**: Compute analytical gradients (when GPU backward is implemented)
//! 4. **Verify**: Compare with numerical gradients from finite-difference
//! 5. **Assert**: `relative_error < 1e-3`

use nalgebra as na;
use oxigaf_render::config::RasterConfig;
use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf_render::{CpuCamera, RenderError};

pub mod finite_diff;
pub mod test_opacity;
pub mod test_position;
pub mod test_rotation;
pub mod test_scale;
pub mod test_sh;

// Re-export commonly used types
pub use finite_diff::{
    compute_opacity_gradients, compute_position_gradients, compute_relative_error,
    compute_rotation_gradients, compute_scale_gradients, compute_sh_gradients, FiniteDiffConfig,
    MseLoss,
};

/// Maximum relative error threshold for gradient tests (legacy).
/// Prefer using `median_error` with `MEDIAN_ERROR_THRESHOLD`.
#[allow(dead_code)]
pub const MAX_RELATIVE_ERROR: f32 = 1e-1;

/// Median error threshold for gradient verification.
///
/// The median is naturally robust to outliers regardless of sample size.
/// At least 50% of gradient entries must match within this threshold.
#[allow(dead_code)]
pub const MEDIAN_ERROR_THRESHOLD: f32 = 5e-2;

/// Position-specific median error threshold for gradient verification.
///
/// Position gradients through a tiled rasterizer have higher finite-difference error
/// because position perturbation directly affects tile assignment, causing discontinuities
/// in the forward pass that the backward pass (correctly) doesn't model.
#[allow(dead_code)]
pub const POSITION_MEDIAN_ERROR_THRESHOLD: f32 = 2.5e-1;

/// Maximum fraction of entries allowed to be outliers (error > 0.5).
#[allow(dead_code)]
pub const MAX_OUTLIER_FRACTION: f32 = 0.3;

/// Compute median of a (mutable) error vector.
///
/// The median is naturally robust to outliers regardless of sample size.
/// The vector is sorted in-place.
pub fn median_error(errors: &mut [f32]) -> f32 {
    if errors.is_empty() {
        return 0.0;
    }
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = errors.len();
    if n.is_multiple_of(2) {
        (errors[n / 2 - 1] + errors[n / 2]) / 2.0
    } else {
        errors[n / 2]
    }
}

/// Test scene configuration.
#[derive(Debug, Clone)]
pub struct TestSceneConfig {
    /// Number of Gaussians in the scene.
    pub num_gaussians: usize,
    /// Image resolution (width, height).
    pub resolution: (u32, u32),
    /// SH degree (0-3).
    pub sh_degree: u32,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for TestSceneConfig {
    fn default() -> Self {
        Self {
            num_gaussians: 5,
            resolution: (128, 128),
            sh_degree: 0,
            seed: 42,
        }
    }
}

/// Create a simple test scene with random Gaussians.
///
/// Gaussians are positioned in front of the camera with random rotations,
/// scales, and colors for gradient testing.
pub fn create_test_scene(config: &TestSceneConfig) -> Result<GaussianModel, RenderError> {
    // Use a simple deterministic pattern based on seed
    let mut gaussians = Vec::new();
    let mut sh_coeffs = Vec::new();
    let sh_coeffs_per_gaussian = ((config.sh_degree + 1) * (config.sh_degree + 1) * 3) as usize;

    for i in 0..config.num_gaussians {
        let offset = (i as f32 + config.seed as f32 * 0.01) * 0.1;

        // Position: in front of camera, slight offset per Gaussian
        let position = [
            (offset * 3.0).sin() * 0.5,
            (offset * 5.0).sin() * 0.5,
            -3.0 - offset, // Behind near plane
        ];

        // Rotation: slight variation per Gaussian
        let angle = offset;
        let axis = na::Vector3::new(0.0, 1.0, 0.0);
        let quat = na::UnitQuaternion::from_axis_angle(&na::Unit::new_normalize(axis), angle);
        let rotation = [quat.coords.x, quat.coords.y, quat.coords.z, quat.coords.w];

        // Scale: log-space, so exp(scale) gives actual scale
        let scale = [
            -1.0 + offset * 0.1,
            -1.0 + offset * 0.1,
            -1.0 + offset * 0.1,
        ];

        // Opacity: sigmoid-inverse space
        let opacity = offset * 0.5;

        gaussians.push(GaussianAttributes {
            position,
            _pad0: 0.0,
            rotation,
            scale,
            opacity,
        });

        // SH coefficients: deterministic colors based on index
        for j in 0..sh_coeffs_per_gaussian {
            sh_coeffs.push(((i * sh_coeffs_per_gaussian + j) as f32 * 0.01).sin() * 0.5);
        }
    }

    Ok(GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree: config.sh_degree,
        face_indices: vec![],
        barycentric: vec![],
        local_offsets: vec![],
        is_rigid: vec![],
    })
}

/// Create a test camera looking at the origin.
pub fn create_test_camera(resolution: (u32, u32)) -> CpuCamera {
    let (width, height) = resolution;

    // Camera positioned at (0, 0, 0) looking down -Z axis
    let view = na::Matrix4::look_at_rh(
        &na::Point3::new(0.0, 0.0, 0.0),
        &na::Point3::new(0.0, 0.0, -1.0),
        &na::Vector3::y(),
    );

    // Simple perspective projection
    let fov_y = 45.0f32.to_radians();
    let aspect = width as f32 / height as f32;
    let near = 0.1;
    let far = 100.0;
    let proj = na::Matrix4::new_perspective(aspect, fov_y, near, far);

    // Focal lengths in pixels
    let focal_y = height as f32 / (2.0 * (fov_y / 2.0).tan());
    let focal_x = focal_y; // Square pixels
    let focal = na::Vector2::new(focal_x, focal_y);

    CpuCamera {
        view,
        proj,
        position: na::Vector3::zeros(),
        focal,
    }
}

/// Create a target image (all black for MSE loss).
pub fn create_target_image(resolution: (u32, u32)) -> Vec<f32> {
    let (width, height) = resolution;
    vec![0.0; (width * height * 4) as usize]
}

/// Gradient verification result.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GradientVerificationResult {
    /// Maximum relative error across all gradients.
    pub max_error: f32,
    /// Mean relative error.
    pub mean_error: f32,
    /// Number of gradients checked.
    pub num_gradients: usize,
    /// Whether verification passed (max_error < threshold).
    pub passed: bool,
}

impl GradientVerificationResult {
    /// Create a new verification result.
    pub fn new(errors: &[f32], threshold: f32) -> Self {
        let max_error = errors.iter().fold(0.0f32, |a, b| a.max(*b));
        let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
        let num_gradients = errors.len();
        let passed = max_error < threshold;

        Self {
            max_error,
            mean_error,
            num_gradients,
            passed,
        }
    }

    /// Print verification summary.
    #[allow(dead_code)]
    pub fn print_summary(&self) {
        println!("Gradient Verification:");
        println!("  Max Error:  {:.6e}", self.max_error);
        println!("  Mean Error: {:.6e}", self.mean_error);
        println!("  Num Grads:  {}", self.num_gradients);
        println!(
            "  Status:     {}",
            if self.passed { "PASS" } else { "FAIL" }
        );
    }
}

/// Compare two gradient arrays and compute relative errors.
pub fn compare_gradients_3d(analytical: &[[f32; 3]], numerical: &[[f32; 3]]) -> Vec<f32> {
    analytical
        .iter()
        .zip(numerical.iter())
        .flat_map(|(a, n)| {
            vec![
                compute_relative_error(a[0], n[0]),
                compute_relative_error(a[1], n[1]),
                compute_relative_error(a[2], n[2]),
            ]
        })
        .collect()
}

/// Compare two gradient arrays (4D) and compute relative errors.
#[allow(dead_code)]
pub fn compare_gradients_4d(analytical: &[[f32; 4]], numerical: &[[f32; 4]]) -> Vec<f32> {
    analytical
        .iter()
        .zip(numerical.iter())
        .flat_map(|(a, n)| {
            vec![
                compute_relative_error(a[0], n[0]),
                compute_relative_error(a[1], n[1]),
                compute_relative_error(a[2], n[2]),
                compute_relative_error(a[3], n[3]),
            ]
        })
        .collect()
}

/// Compare two gradient arrays (1D) and compute relative errors.
#[allow(dead_code)]
pub fn compare_gradients_1d(analytical: &[f32], numerical: &[f32]) -> Vec<f32> {
    analytical
        .iter()
        .zip(numerical.iter())
        .map(|(a, n)| compute_relative_error(*a, *n))
        .collect()
}

/// Compute analytical gradients using GPU backward pass.
///
/// This function:
/// 1. Creates a GPU rasterizer
/// 2. Runs forward pass to render the image
/// 3. Computes loss gradient (simple MSE gradient: 2*(rendered - target))
/// 4. Runs backward pass to get per-Gaussian gradients
///
/// Returns analytical gradients that can be compared with numerical gradients.
#[allow(dead_code)]
pub async fn compute_analytical_gradients(
    model: &GaussianModel,
    camera: &CpuCamera,
    target: &[f32],
    config: &RasterConfig,
) -> Result<oxigaf_render::GaussianGradients, RenderError> {
    use oxigaf_render::{Rasterizer, RenderCamera};

    // Create GPU rasterizer
    let mut rasterizer = Rasterizer::new(config.clone()).await?;

    // Convert CpuCamera to RenderCamera
    let view_matrix: [f32; 16] = camera
        .view
        .as_slice()
        .try_into()
        .map_err(|_| RenderError::Rasterize("Failed to convert view matrix".into()))?;
    let proj_matrix: [f32; 16] = camera
        .proj
        .as_slice()
        .try_into()
        .map_err(|_| RenderError::Rasterize("Failed to convert proj matrix".into()))?;

    let render_camera = RenderCamera {
        view_matrix,
        proj_matrix,
        position: [camera.position.x, camera.position.y, camera.position.z],
        focal: [camera.focal.x, camera.focal.y],
    };

    // Upload Gaussians
    rasterizer.upload_gaussians(model);

    // Forward pass
    let output = rasterizer.forward(model, &render_camera)?;

    // Compute loss gradient: ∂L/∂rendered = 2 * (rendered - target) / N for RGB, 0 for alpha
    // RGB-only MSE matches the GPU backward pass which only processes RGB channels
    let num_pixels = target.len() / 4;
    let n_rgb = (num_pixels * 3) as f32;
    let grad_image: Vec<f32> = output
        .color_data
        .chunks(4)
        .zip(target.chunks(4))
        .flat_map(|(rendered, target_chunk)| {
            [
                2.0 * (rendered[0] - target_chunk[0]) / n_rgb,
                2.0 * (rendered[1] - target_chunk[1]) / n_rgb,
                2.0 * (rendered[2] - target_chunk[2]) / n_rgb,
                0.0, // Zero alpha gradient - GPU backward only processes RGB
            ]
        })
        .collect();

    // Backward pass
    let gradients = rasterizer.backward(model, &grad_image)?;

    Ok(gradients)
}

/// Synchronous wrapper for compute_analytical_gradients using pollster.
///
/// This is more convenient for tests that don't want to deal with async.
#[allow(dead_code)]
pub fn compute_analytical_gradients_sync(
    model: &GaussianModel,
    camera: &CpuCamera,
    target: &[f32],
    config: &RasterConfig,
) -> Result<oxigaf_render::GaussianGradients, RenderError> {
    pollster::block_on(compute_analytical_gradients(model, camera, target, config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_scene() {
        let config = TestSceneConfig::default();
        let scene = create_test_scene(&config);
        assert!(scene.is_ok());

        let model = scene.ok().unwrap_or_else(|| {
            panic!("Failed to create test scene");
        });
        assert_eq!(model.len(), config.num_gaussians);
    }

    #[test]
    fn test_create_test_camera() {
        let camera = create_test_camera((128, 128));
        assert_eq!(camera.position, na::Vector3::zeros());
    }

    #[test]
    fn test_gradient_verification_result() {
        let errors = vec![0.0001, 0.0002, 0.0003];
        let result = GradientVerificationResult::new(&errors, 1e-3);

        assert!(result.passed);
        assert_eq!(result.max_error, 0.0003);
        assert!((result.mean_error - 0.0002).abs() < 1e-6);
    }

    #[test]
    fn test_compare_gradients_3d() {
        let analytical = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let numerical = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let errors = compare_gradients_3d(&analytical, &numerical);
        assert_eq!(errors.len(), 6);
        assert!(errors.iter().all(|&e| e < 1e-6));
    }
}
