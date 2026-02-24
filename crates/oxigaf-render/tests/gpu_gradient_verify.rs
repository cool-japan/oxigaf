// Allow range loops in gradient verification — indices are used to access
// multiple parallel arrays (positions, names, gradients) simultaneously.
#![allow(clippy::needless_range_loop)]
//! GPU self-consistent gradient verification tests.
//!
//! These tests compute BOTH numerical and analytical gradients using the GPU
//! forward pass, eliminating CPU-GPU rendering differences as a confounding
//! factor when debugging the backward pass.
//!
//! **Approach**:
//! 1. Run GPU forward pass on the base model, compute base MSE loss.
//! 2. Run GPU backward pass to get analytical gradients.
//! 3. For each parameter perturbation, upload the perturbed model to GPU,
//!    run the forward pass, and compute the numerical gradient via
//!    central-difference: `(loss_plus - loss_minus) / (2 * epsilon)`.
//! 4. Compare analytical vs. numerical gradients.
//!
//! This isolates backward-pass correctness from any CPU-GPU forward-pass
//! discrepancy.

use nalgebra as na;
use oxigaf_render::config::RasterConfig;
use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf_render::{CpuCamera, Rasterizer, RenderCamera, RenderError};

// ---------------------------------------------------------------------------
// Scene & camera helpers (mirror gradient_verification/mod.rs exactly)
// ---------------------------------------------------------------------------

/// Create the standard test scene matching gradient_verification::create_test_scene.
fn create_test_scene(num_gaussians: usize, sh_degree: u32, seed: u64) -> GaussianModel {
    let mut gaussians = Vec::new();
    let mut sh_coeffs = Vec::new();
    let sh_coeffs_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;

    for i in 0..num_gaussians {
        let offset = (i as f32 + seed as f32 * 0.01) * 0.1;

        let position = [
            (offset * 3.0).sin() * 0.5,
            (offset * 5.0).sin() * 0.5,
            -3.0 - offset,
        ];

        let angle = offset;
        let axis = na::Vector3::new(0.0, 1.0, 0.0);
        let quat = na::UnitQuaternion::from_axis_angle(&na::Unit::new_normalize(axis), angle);
        let rotation = [quat.coords.x, quat.coords.y, quat.coords.z, quat.coords.w];

        let scale = [
            -1.0 + offset * 0.1,
            -1.0 + offset * 0.1,
            -1.0 + offset * 0.1,
        ];

        let opacity = offset * 0.5;

        gaussians.push(GaussianAttributes {
            position,
            _pad0: 0.0,
            rotation,
            scale,
            opacity,
        });

        for j in 0..sh_coeffs_per_gaussian {
            sh_coeffs.push(((i * sh_coeffs_per_gaussian + j) as f32 * 0.01).sin() * 0.5);
        }
    }

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: vec![],
        barycentric: vec![],
        local_offsets: vec![],
        is_rigid: vec![],
    }
}

/// Create the standard test camera matching gradient_verification::create_test_camera.
fn create_test_camera(resolution: (u32, u32)) -> CpuCamera {
    let (width, height) = resolution;

    let view = na::Matrix4::look_at_rh(
        &na::Point3::new(0.0, 0.0, 0.0),
        &na::Point3::new(0.0, 0.0, -1.0),
        &na::Vector3::y(),
    );

    let fov_y = 45.0f32.to_radians();
    let aspect = width as f32 / height as f32;
    let near = 0.1;
    let far = 100.0;
    let proj = na::Matrix4::new_perspective(aspect, fov_y, near, far);

    let focal_y = height as f32 / (2.0 * (fov_y / 2.0).tan());
    let focal_x = focal_y;
    let focal = na::Vector2::new(focal_x, focal_y);

    CpuCamera {
        view,
        proj,
        position: na::Vector3::zeros(),
        focal,
    }
}

/// Convert a CpuCamera to a GPU RenderCamera.
fn cpu_to_render_camera(camera: &CpuCamera) -> Result<RenderCamera, RenderError> {
    let view_matrix: [f32; 16] =
        camera.view.as_slice().try_into().map_err(|_| {
            RenderError::Rasterize("Failed to convert view matrix to [f32; 16]".into())
        })?;
    let proj_matrix: [f32; 16] =
        camera.proj.as_slice().try_into().map_err(|_| {
            RenderError::Rasterize("Failed to convert proj matrix to [f32; 16]".into())
        })?;

    Ok(RenderCamera {
        view_matrix,
        proj_matrix,
        position: [camera.position.x, camera.position.y, camera.position.z],
        focal: [camera.focal.x, camera.focal.y],
    })
}

// ---------------------------------------------------------------------------
// Loss & error utilities
// ---------------------------------------------------------------------------

/// Compute RGB-only MSE loss (skip alpha channel).
///
/// `loss = sum_pixels( (R-Rt)^2 + (G-Gt)^2 + (B-Bt)^2 ) / (num_pixels * 3)`
fn mse_loss(color_data: &[f32], target: &[f32]) -> f32 {
    let num_pixels = color_data.len() / 4;
    let rgb_count = (num_pixels * 3) as f32;
    color_data
        .chunks(4)
        .zip(target.chunks(4))
        .map(|(c, t)| (c[0] - t[0]).powi(2) + (c[1] - t[1]).powi(2) + (c[2] - t[2]).powi(2))
        .sum::<f32>()
        / rgb_count
}

/// Compute dL/d(rendered) for RGB-only MSE, producing an RGBA gradient image.
///
/// `dL/dR_i = 2 * (R_i - Rt_i) / (num_pixels * 3)`, similarly for G, B.
/// Alpha gradient is always 0.
fn mse_grad_image(color_data: &[f32], target: &[f32]) -> Vec<f32> {
    let num_pixels = color_data.len() / 4;
    let n_rgb = (num_pixels * 3) as f32;
    color_data
        .chunks(4)
        .zip(target.chunks(4))
        .flat_map(|(r, t)| {
            [
                2.0 * (r[0] - t[0]) / n_rgb,
                2.0 * (r[1] - t[1]) / n_rgb,
                2.0 * (r[2] - t[2]) / n_rgb,
                0.0,
            ]
        })
        .collect()
}

/// Compute error metric that handles near-zero gradients.
///
/// Uses relative error when gradients are significant, and absolute error
/// when both gradients are near zero.
fn gradient_error(analytical: f32, numerical: f32) -> f32 {
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

/// Compute median of a (mutable) error vector.
///
/// The median is naturally robust to outliers regardless of sample size.
/// The vector is sorted in-place.
fn median_error(errors: &mut [f32]) -> f32 {
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

// ---------------------------------------------------------------------------
// Core GPU gradient verification driver
// ---------------------------------------------------------------------------

/// Result of a single parameter-type gradient verification run.
struct GpuGradVerifyResult {
    max_error: f32,
    mean_error: f32,
    median_error: f32,
    num_gradients: usize,
    num_outliers: usize,
}

/// Compute GPU forward pass on a model and return the loss value.
///
/// This re-uploads the model to the rasterizer and runs the forward pass.
fn gpu_forward_loss(
    rasterizer: &mut Rasterizer,
    model: &GaussianModel,
    camera: &RenderCamera,
    target: &[f32],
) -> Result<f32, RenderError> {
    rasterizer.upload_gaussians(model);
    let output = rasterizer.forward(model, camera)?;
    Ok(mse_loss(&output.color_data, target))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_self_consistent_opacity() {
    pollster::block_on(async {
        let resolution = (64, 64);
        let num_gaussians = 5;
        let sh_degree = 0;
        let epsilon = 5e-3f32;

        let config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree);
        let scene = create_test_scene(num_gaussians, sh_degree, 42);
        let camera = create_test_camera(resolution);
        let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
        let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];

        // -- Step 1: Analytical gradients via GPU backward --
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer init");
        rasterizer.upload_gaussians(&scene);
        let base_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("base forward");
        let base_loss = mse_loss(&base_output.color_data, &target);
        let grad_image = mse_grad_image(&base_output.color_data, &target);
        let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

        println!("=== GPU Self-Consistent Opacity Gradient Test ===");
        println!("Base loss: {:.6e}", base_loss);

        // -- Step 2: Numerical gradients via GPU forward --
        let mut max_error = 0.0f32;
        let mut sum_error = 0.0f32;
        let mut count = 0usize;
        let mut all_errors = Vec::new();

        for i in 0..scene.len() {
            // Forward perturbation
            let mut perturbed_plus = scene.clone();
            perturbed_plus.gaussians[i].opacity += epsilon;
            let loss_plus =
                gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                    .expect("fwd+");

            // Backward perturbation
            let mut perturbed_minus = scene.clone();
            perturbed_minus.gaussians[i].opacity -= epsilon;
            let loss_minus =
                gpu_forward_loss(&mut rasterizer, &perturbed_minus, &render_camera, &target)
                    .expect("fwd-");

            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytic = analytical.grad_opacities[i];
            let err = gradient_error(analytic, numerical);

            println!(
                "  Opacity[{}]: numerical={:.6e}, analytical={:.6e}, error={:.6e}",
                i, numerical, analytic, err
            );
            if err > 0.1 {
                println!(
                    "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                    loss_plus,
                    loss_minus,
                    loss_plus - loss_minus
                );
            }

            all_errors.push(err);
            max_error = max_error.max(err);
            sum_error += err;
            count += 1;
        }

        let mean_error = if count > 0 {
            sum_error / count as f32
        } else {
            0.0
        };
        let median_err = median_error(&mut all_errors);
        let num_outliers = all_errors.iter().filter(|&&e| e > 0.5).count();
        println!(
            "Opacity: max_error={:.6e}, median_error={:.6e}, mean_error={:.6e}, n={}, outliers={}",
            max_error, median_err, mean_error, count, num_outliers
        );
        assert!(
            median_err < 5e-2,
            "Opacity gradient median error too high: {:.6e} (max={:.6e}, outliers={})",
            median_err,
            max_error,
            num_outliers
        );
        let max_outlier_count = (all_errors.len() as f32 * 0.3).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Opacity too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            all_errors.len(),
            max_outlier_count,
            median_err
        );
    });
}

#[test]
fn test_gpu_self_consistent_position() {
    pollster::block_on(async {
        let resolution = (64, 64);
        let num_gaussians = 5;
        let sh_degree = 0;
        let epsilon = 5e-3f32;

        let config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree);
        let scene = create_test_scene(num_gaussians, sh_degree, 42);
        let camera = create_test_camera(resolution);
        let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
        let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];

        // -- Analytical gradients --
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer init");
        rasterizer.upload_gaussians(&scene);
        let base_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("base forward");
        let base_loss = mse_loss(&base_output.color_data, &target);
        let grad_image = mse_grad_image(&base_output.color_data, &target);
        let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

        println!("=== GPU Self-Consistent Position Gradient Test ===");
        println!("Base loss: {:.6e}", base_loss);

        // -- Numerical gradients --
        let mut max_error = 0.0f32;
        let mut sum_error = 0.0f32;
        let mut count = 0usize;
        let mut all_errors = Vec::new();
        let axis_names = ["x", "y", "z"];

        for i in 0..scene.len() {
            for (axis, axis_name) in axis_names.iter().enumerate() {
                // Forward perturbation
                let mut perturbed_plus = scene.clone();
                perturbed_plus.gaussians[i].position[axis] += epsilon;
                let loss_plus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                        .expect("fwd+");

                // Backward perturbation
                let mut perturbed_minus = scene.clone();
                perturbed_minus.gaussians[i].position[axis] -= epsilon;
                let loss_minus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_minus, &render_camera, &target)
                        .expect("fwd-");

                let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                let analytic = analytical.grad_positions[i][axis];
                let err = gradient_error(analytic, numerical);

                println!(
                    "  Position[{}].{}: numerical={:.6e}, analytical={:.6e}, error={:.6e}",
                    i, axis_name, numerical, analytic, err
                );
                if err > 0.1 {
                    println!(
                        "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                        loss_plus,
                        loss_minus,
                        loss_plus - loss_minus
                    );
                }

                all_errors.push(err);
                max_error = max_error.max(err);
                sum_error += err;
                count += 1;
            }
        }

        let mean_error = if count > 0 {
            sum_error / count as f32
        } else {
            0.0
        };
        let median_err = median_error(&mut all_errors);
        let num_outliers = all_errors.iter().filter(|&&e| e > 0.5).count();
        println!(
            "Position: max_error={:.6e}, median_error={:.6e}, mean_error={:.6e}, n={}, outliers={}",
            max_error, median_err, mean_error, count, num_outliers
        );
        // Position gradients through a tiled rasterizer have higher finite-difference error
        // because position perturbation directly affects tile assignment, causing discontinuities
        // in the forward pass that the backward pass (correctly) doesn't model.
        assert!(
            median_err < 2.5e-1, // See POSITION_MEDIAN_ERROR_THRESHOLD in mod.rs
            "Position gradient median error too high: {:.6e} (max={:.6e}, outliers={})",
            median_err,
            max_error,
            num_outliers
        );
        let max_outlier_count = (all_errors.len() as f32 * 0.3).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Position too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            all_errors.len(),
            max_outlier_count,
            median_err
        );
    });
}

#[test]
fn test_gpu_self_consistent_scale() {
    pollster::block_on(async {
        let resolution = (64, 64);
        let num_gaussians = 5;
        let sh_degree = 0;
        let epsilon = 5e-3f32;

        let config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree);
        let scene = create_test_scene(num_gaussians, sh_degree, 42);
        let camera = create_test_camera(resolution);
        let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
        let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];

        // -- Analytical gradients --
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer init");
        rasterizer.upload_gaussians(&scene);
        let base_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("base forward");
        let base_loss = mse_loss(&base_output.color_data, &target);
        let grad_image = mse_grad_image(&base_output.color_data, &target);
        let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

        println!("=== GPU Self-Consistent Scale Gradient Test ===");
        println!("Base loss: {:.6e}", base_loss);

        // -- Numerical gradients --
        let mut max_error = 0.0f32;
        let mut sum_error = 0.0f32;
        let mut count = 0usize;
        let mut all_errors = Vec::new();
        let axis_names = ["sx", "sy", "sz"];

        for i in 0..scene.len() {
            for (axis, axis_name) in axis_names.iter().enumerate() {
                // Forward perturbation
                let mut perturbed_plus = scene.clone();
                perturbed_plus.gaussians[i].scale[axis] += epsilon;
                let loss_plus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                        .expect("fwd+");

                // Backward perturbation
                let mut perturbed_minus = scene.clone();
                perturbed_minus.gaussians[i].scale[axis] -= epsilon;
                let loss_minus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_minus, &render_camera, &target)
                        .expect("fwd-");

                let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                let analytic = analytical.grad_scales[i][axis];
                let err = gradient_error(analytic, numerical);

                println!(
                    "  Scale[{}].{}: numerical={:.6e}, analytical={:.6e}, error={:.6e}",
                    i, axis_name, numerical, analytic, err
                );
                if err > 0.1 {
                    println!(
                        "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                        loss_plus,
                        loss_minus,
                        loss_plus - loss_minus
                    );
                }

                all_errors.push(err);
                max_error = max_error.max(err);
                sum_error += err;
                count += 1;
            }
        }

        let mean_error = if count > 0 {
            sum_error / count as f32
        } else {
            0.0
        };
        let median_err = median_error(&mut all_errors);
        let num_outliers = all_errors.iter().filter(|&&e| e > 0.5).count();
        println!(
            "Scale: max_error={:.6e}, median_error={:.6e}, mean_error={:.6e}, n={}, outliers={}",
            max_error, median_err, mean_error, count, num_outliers
        );
        assert!(
            median_err < 5e-2,
            "Scale gradient median error too high: {:.6e} (max={:.6e}, outliers={})",
            median_err,
            max_error,
            num_outliers
        );
        let max_outlier_count = (all_errors.len() as f32 * 0.3).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Scale too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            all_errors.len(),
            max_outlier_count,
            median_err
        );
    });
}

#[test]
fn test_gpu_self_consistent_rotation() {
    pollster::block_on(async {
        let resolution = (64, 64);
        let num_gaussians = 5;
        let sh_degree = 0;
        let epsilon = 5e-3f32;

        let config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree);
        let scene = create_test_scene(num_gaussians, sh_degree, 42);
        let camera = create_test_camera(resolution);
        let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
        let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];

        // -- Analytical gradients --
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer init");
        rasterizer.upload_gaussians(&scene);
        let base_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("base forward");
        let base_loss = mse_loss(&base_output.color_data, &target);
        let grad_image = mse_grad_image(&base_output.color_data, &target);
        let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

        println!("=== GPU Self-Consistent Rotation Gradient Test ===");
        println!("Base loss: {:.6e}", base_loss);

        // -- Numerical gradients --
        // Note: We perturb raw quaternion components (x, y, z, w) WITHOUT
        // renormalization, matching the GPU's quat_to_mat() which uses the
        // raw quaternion.
        let mut max_error = 0.0f32;
        let mut sum_error = 0.0f32;
        let mut count = 0usize;
        let mut all_errors = Vec::new();
        let comp_names = ["qx", "qy", "qz", "qw"];

        for i in 0..scene.len() {
            for (comp, comp_name) in comp_names.iter().enumerate() {
                // Forward perturbation
                let mut perturbed_plus = scene.clone();
                perturbed_plus.gaussians[i].rotation[comp] += epsilon;
                let loss_plus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                        .expect("fwd+");

                // Backward perturbation
                let mut perturbed_minus = scene.clone();
                perturbed_minus.gaussians[i].rotation[comp] -= epsilon;
                let loss_minus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_minus, &render_camera, &target)
                        .expect("fwd-");

                let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                let analytic = analytical.grad_rotations[i][comp];
                let err = gradient_error(analytic, numerical);

                println!(
                    "  Rotation[{}].{}: numerical={:.6e}, analytical={:.6e}, error={:.6e}",
                    i, comp_name, numerical, analytic, err
                );
                if err > 0.1 {
                    println!(
                        "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                        loss_plus,
                        loss_minus,
                        loss_plus - loss_minus
                    );
                }

                all_errors.push(err);
                max_error = max_error.max(err);
                sum_error += err;
                count += 1;
            }
        }

        let mean_error = if count > 0 {
            sum_error / count as f32
        } else {
            0.0
        };
        let median_err = median_error(&mut all_errors);
        let num_outliers = all_errors.iter().filter(|&&e| e > 0.5).count();
        println!(
            "Rotation: max_error={:.6e}, median_error={:.6e}, mean_error={:.6e}, n={}, outliers={}",
            max_error, median_err, mean_error, count, num_outliers
        );
        assert!(
            median_err < 5e-2,
            "Rotation gradient median error too high: {:.6e} (max={:.6e}, outliers={})",
            median_err,
            max_error,
            num_outliers
        );
        let max_outlier_count = (all_errors.len() as f32 * 0.3).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Rotation too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            all_errors.len(),
            max_outlier_count,
            median_err
        );
    });
}

#[test]
fn test_gpu_self_consistent_sh0() {
    pollster::block_on(async {
        let resolution = (64, 64);
        let num_gaussians = 5;
        let sh_degree = 0;
        let epsilon = 5e-3f32;

        let config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree);
        let scene = create_test_scene(num_gaussians, sh_degree, 42);
        let camera = create_test_camera(resolution);
        let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
        let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];

        let sh_coeffs_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let total_sh_coeffs = sh_coeffs_per_gaussian * num_gaussians;

        // -- Analytical gradients --
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer init");
        rasterizer.upload_gaussians(&scene);
        let base_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("base forward");
        let base_loss = mse_loss(&base_output.color_data, &target);
        let grad_image = mse_grad_image(&base_output.color_data, &target);
        let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

        println!("=== GPU Self-Consistent SH0 Gradient Test ===");
        println!("Base loss: {:.6e}", base_loss);
        println!(
            "SH coeffs per gaussian: {}, total: {}",
            sh_coeffs_per_gaussian, total_sh_coeffs
        );

        // -- Numerical gradients --
        let mut max_error = 0.0f32;
        let mut sum_error = 0.0f32;
        let mut count = 0usize;
        let mut all_errors = Vec::new();

        for coeff_idx in 0..total_sh_coeffs {
            let gaussian_idx = coeff_idx / sh_coeffs_per_gaussian;
            let local_idx = coeff_idx % sh_coeffs_per_gaussian;
            let channel = ["R", "G", "B"][local_idx % 3];

            // Forward perturbation
            let mut perturbed_plus = scene.clone();
            perturbed_plus.sh_coeffs[coeff_idx] += epsilon;
            let loss_plus =
                gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                    .expect("fwd+");

            // Backward perturbation
            let mut perturbed_minus = scene.clone();
            perturbed_minus.sh_coeffs[coeff_idx] -= epsilon;
            let loss_minus =
                gpu_forward_loss(&mut rasterizer, &perturbed_minus, &render_camera, &target)
                    .expect("fwd-");

            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytic = analytical.grad_sh_coeffs[coeff_idx];
            let err = gradient_error(analytic, numerical);

            println!(
                "  SH[g={}, l={}, ch={}]: numerical={:.6e}, analytical={:.6e}, error={:.6e}",
                gaussian_idx,
                local_idx / 3,
                channel,
                numerical,
                analytic,
                err
            );
            if err > 0.1 {
                println!(
                    "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                    loss_plus,
                    loss_minus,
                    loss_plus - loss_minus
                );
            }

            all_errors.push(err);
            max_error = max_error.max(err);
            sum_error += err;
            count += 1;
        }

        let mean_error = if count > 0 {
            sum_error / count as f32
        } else {
            0.0
        };
        let median_err = median_error(&mut all_errors);
        let num_outliers = all_errors.iter().filter(|&&e| e > 0.5).count();
        println!(
            "SH0: max_error={:.6e}, median_error={:.6e}, mean_error={:.6e}, n={}, outliers={}",
            max_error, median_err, mean_error, count, num_outliers
        );
        assert!(
            median_err < 5e-2,
            "SH0 gradient median error too high: {:.6e} (max={:.6e}, outliers={})",
            median_err,
            max_error,
            num_outliers
        );
        let max_outlier_count = (all_errors.len() as f32 * 0.3).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "SH0 too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            all_errors.len(),
            max_outlier_count,
            median_err
        );
    });
}

#[test]
fn test_gpu_self_consistent_sh1() {
    pollster::block_on(async {
        let resolution = (64, 64);
        let num_gaussians = 5;
        let sh_degree = 1;
        let epsilon = 5e-3f32;

        let config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree);
        let scene = create_test_scene(num_gaussians, sh_degree, 42);
        let camera = create_test_camera(resolution);
        let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
        let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];

        let sh_coeffs_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let total_sh_coeffs = sh_coeffs_per_gaussian * num_gaussians;

        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer init");
        rasterizer.upload_gaussians(&scene);
        let base_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("base forward");
        let base_loss = mse_loss(&base_output.color_data, &target);
        let grad_image = mse_grad_image(&base_output.color_data, &target);
        let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

        println!("=== GPU Self-Consistent SH1 Gradient Test ===");
        println!("Base loss: {:.6e}", base_loss);
        println!(
            "SH coeffs per gaussian: {}, total: {}",
            sh_coeffs_per_gaussian, total_sh_coeffs
        );

        let mut max_error = 0.0f32;
        let mut sum_error = 0.0f32;
        let mut count = 0usize;
        let mut all_errors = Vec::new();

        for coeff_idx in 0..total_sh_coeffs {
            let gaussian_idx = coeff_idx / sh_coeffs_per_gaussian;
            let local_idx = coeff_idx % sh_coeffs_per_gaussian;
            let channel = ["R", "G", "B"][local_idx % 3];

            let mut perturbed_plus = scene.clone();
            perturbed_plus.sh_coeffs[coeff_idx] += epsilon;
            let loss_plus =
                gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                    .expect("fwd+");

            let mut perturbed_minus = scene.clone();
            perturbed_minus.sh_coeffs[coeff_idx] -= epsilon;
            let loss_minus =
                gpu_forward_loss(&mut rasterizer, &perturbed_minus, &render_camera, &target)
                    .expect("fwd-");

            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytic = analytical.grad_sh_coeffs[coeff_idx];
            let err = gradient_error(analytic, numerical);

            println!(
                "  SH[g={}, l={}, ch={}]: numerical={:.6e}, analytical={:.6e}, error={:.6e}",
                gaussian_idx,
                local_idx / 3,
                channel,
                numerical,
                analytic,
                err
            );
            if err > 0.1 {
                println!(
                    "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                    loss_plus,
                    loss_minus,
                    loss_plus - loss_minus
                );
            }

            all_errors.push(err);
            max_error = max_error.max(err);
            sum_error += err;
            count += 1;
        }

        let mean_error = if count > 0 {
            sum_error / count as f32
        } else {
            0.0
        };
        let median_err = median_error(&mut all_errors);
        let num_outliers = all_errors.iter().filter(|&&e| e > 0.5).count();
        println!(
            "SH1: max_error={:.6e}, median_error={:.6e}, mean_error={:.6e}, n={}, outliers={}",
            max_error, median_err, mean_error, count, num_outliers
        );
        assert!(
            median_err < 5e-2,
            "SH1 gradient median error too high: {:.6e} (max={:.6e}, outliers={})",
            median_err,
            max_error,
            num_outliers
        );
        let max_outlier_count = (all_errors.len() as f32 * 0.3).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "SH1 too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            all_errors.len(),
            max_outlier_count,
            median_err
        );
    });
}

#[test]
fn test_gpu_self_consistent_sh2() {
    let max_retries = 3;
    let mut last_error = String::new();

    for attempt in 1..=max_retries {
        let result = std::panic::catch_unwind(|| {
            pollster::block_on(async {
                let resolution = (64, 64);
                let num_gaussians = 5;
                let sh_degree = 2;
                let epsilon = 5e-3f32;

                let config = RasterConfig::new()
                    .with_resolution(resolution.0, resolution.1)
                    .with_sh_degree(sh_degree);
                let scene = create_test_scene(num_gaussians, sh_degree, 42);
                let camera = create_test_camera(resolution);
                let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
                let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];

                let sh_coeffs_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
                let total_sh_coeffs = sh_coeffs_per_gaussian * num_gaussians;

                let mut rasterizer = Rasterizer::new(config.clone())
                    .await
                    .expect("GPU rasterizer init");
                rasterizer.upload_gaussians(&scene);
                let base_output = rasterizer
                    .forward(&scene, &render_camera)
                    .expect("base forward");
                let base_loss = mse_loss(&base_output.color_data, &target);
                let grad_image = mse_grad_image(&base_output.color_data, &target);
                let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

                println!("=== GPU Self-Consistent SH2 Gradient Test ===");
                println!("Base loss: {:.6e}", base_loss);
                println!(
                    "SH coeffs per gaussian: {}, total: {}",
                    sh_coeffs_per_gaussian, total_sh_coeffs
                );

                let mut max_error = 0.0f32;
                let mut sum_error = 0.0f32;
                let mut count = 0usize;
                let mut all_errors = Vec::new();

                for coeff_idx in 0..total_sh_coeffs {
                    let gaussian_idx = coeff_idx / sh_coeffs_per_gaussian;
                    let local_idx = coeff_idx % sh_coeffs_per_gaussian;
                    let channel = ["R", "G", "B"][local_idx % 3];

                    let mut perturbed_plus = scene.clone();
                    perturbed_plus.sh_coeffs[coeff_idx] += epsilon;
                    let loss_plus =
                        gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                            .expect("fwd+");

                    let mut perturbed_minus = scene.clone();
                    perturbed_minus.sh_coeffs[coeff_idx] -= epsilon;
                    let loss_minus = gpu_forward_loss(
                        &mut rasterizer,
                        &perturbed_minus,
                        &render_camera,
                        &target,
                    )
                    .expect("fwd-");

                    let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                    let analytic = analytical.grad_sh_coeffs[coeff_idx];
                    let err = gradient_error(analytic, numerical);

                    println!(
                        "  SH[g={}, l={}, ch={}]: numerical={:.6e}, analytical={:.6e}, error={:.6e}",
                        gaussian_idx,
                        local_idx / 3,
                        channel,
                        numerical,
                        analytic,
                        err
                    );
                    if err > 0.1 {
                        println!(
                            "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                            loss_plus,
                            loss_minus,
                            loss_plus - loss_minus
                        );
                    }

                    all_errors.push(err);
                    max_error = max_error.max(err);
                    sum_error += err;
                    count += 1;
                }

                let mean_error = if count > 0 {
                    sum_error / count as f32
                } else {
                    0.0
                };
                let median_err = median_error(&mut all_errors);
                let num_outliers = all_errors.iter().filter(|&&e| e > 0.5).count();
                println!(
                    "SH2: max_error={:.6e}, median_error={:.6e}, mean_error={:.6e}, n={}, outliers={}",
                    max_error, median_err, mean_error, count, num_outliers
                );
                assert!(
                    median_err < 5e-2,
                    "SH2 gradient median error too high: {:.6e} (max={:.6e}, outliers={})",
                    median_err,
                    max_error,
                    num_outliers
                );
                let max_outlier_count = (all_errors.len() as f32 * 0.3).ceil() as usize;
                assert!(
                    num_outliers <= max_outlier_count,
                    "SH2 too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
                    num_outliers,
                    all_errors.len(),
                    max_outlier_count,
                    median_err
                );
            });
        });

        match result {
            Ok(()) => return,
            Err(e) => {
                last_error = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    format!("{:?}", e)
                };
                if attempt < max_retries {
                    eprintln!("SH2 gradient test attempt {attempt}/{max_retries} failed (GPU non-determinism), retrying...");
                }
            }
        }
    }

    panic!("SH2 gradient test failed after {max_retries} attempts: {last_error}");
}

#[test]
fn test_gpu_self_consistent_sh3() {
    pollster::block_on(async {
        let resolution = (64, 64);
        let num_gaussians = 5;
        let sh_degree = 3;
        let epsilon = 5e-3f32;

        let config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree);
        let scene = create_test_scene(num_gaussians, sh_degree, 42);
        let camera = create_test_camera(resolution);
        let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
        let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];

        let sh_coeffs_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let total_sh_coeffs = sh_coeffs_per_gaussian * num_gaussians;

        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer init");
        rasterizer.upload_gaussians(&scene);
        let base_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("base forward");
        let base_loss = mse_loss(&base_output.color_data, &target);
        let grad_image = mse_grad_image(&base_output.color_data, &target);
        let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

        println!("=== GPU Self-Consistent SH3 Gradient Test ===");
        println!("Base loss: {:.6e}", base_loss);
        println!(
            "SH coeffs per gaussian: {}, total: {}",
            sh_coeffs_per_gaussian, total_sh_coeffs
        );

        let mut max_error = 0.0f32;
        let mut sum_error = 0.0f32;
        let mut count = 0usize;
        let mut all_errors = Vec::new();

        for coeff_idx in 0..total_sh_coeffs {
            let gaussian_idx = coeff_idx / sh_coeffs_per_gaussian;
            let local_idx = coeff_idx % sh_coeffs_per_gaussian;
            let channel = ["R", "G", "B"][local_idx % 3];

            let mut perturbed_plus = scene.clone();
            perturbed_plus.sh_coeffs[coeff_idx] += epsilon;
            let loss_plus =
                gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                    .expect("fwd+");

            let mut perturbed_minus = scene.clone();
            perturbed_minus.sh_coeffs[coeff_idx] -= epsilon;
            let loss_minus =
                gpu_forward_loss(&mut rasterizer, &perturbed_minus, &render_camera, &target)
                    .expect("fwd-");

            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytic = analytical.grad_sh_coeffs[coeff_idx];
            let err = gradient_error(analytic, numerical);

            println!(
                "  SH[g={}, l={}, ch={}]: numerical={:.6e}, analytical={:.6e}, error={:.6e}",
                gaussian_idx,
                local_idx / 3,
                channel,
                numerical,
                analytic,
                err
            );
            if err > 0.1 {
                println!(
                    "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                    loss_plus,
                    loss_minus,
                    loss_plus - loss_minus
                );
            }

            all_errors.push(err);
            max_error = max_error.max(err);
            sum_error += err;
            count += 1;
        }

        let mean_error = if count > 0 {
            sum_error / count as f32
        } else {
            0.0
        };
        let median_err = median_error(&mut all_errors);
        let num_outliers = all_errors.iter().filter(|&&e| e > 0.5).count();
        println!(
            "SH3: max_error={:.6e}, median_error={:.6e}, mean_error={:.6e}, n={}, outliers={}",
            max_error, median_err, mean_error, count, num_outliers
        );
        assert!(
            median_err < 5e-2,
            "SH3 gradient median error too high: {:.6e} (max={:.6e}, outliers={})",
            median_err,
            max_error,
            num_outliers
        );
        let max_outlier_count = (all_errors.len() as f32 * 0.3).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "SH3 too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            all_errors.len(),
            max_outlier_count,
            median_err
        );
    });
}

/// Combined test that verifies all parameter types in a single GPU session.
///
/// This is more efficient than running individual tests because it only
/// initializes the GPU rasterizer once and shares the base forward pass.
#[test]
fn test_gpu_self_consistent_all_params() {
    pollster::block_on(async {
        let resolution = (64, 64);
        let num_gaussians = 5;
        let sh_degree = 0;
        let epsilon = 5e-3f32;
        let median_threshold = 5e-2f32;

        let config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree);
        let scene = create_test_scene(num_gaussians, sh_degree, 42);
        let camera = create_test_camera(resolution);
        let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
        let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];
        let sh_coeffs_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;

        // -- Analytical gradients (single backward pass) --
        // Note: The first forward pass on a fresh rasterizer may produce slightly
        // different results from subsequent passes (GPU pipeline warm-up).
        // We run a warm-up pass first, then use the stabilized output as our base.
        let mut rasterizer = Rasterizer::new(config.clone())
            .await
            .expect("GPU rasterizer init");

        // Warm-up pass to stabilize GPU pipeline state
        rasterizer.upload_gaussians(&scene);
        let _warmup_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("warmup forward");

        // Stabilized base forward pass (matches subsequent numerical forward passes)
        rasterizer.upload_gaussians(&scene);
        let base_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("base forward");
        let base_loss = mse_loss(&base_output.color_data, &target);

        // Determinism check: verify subsequent passes match the stabilized base
        rasterizer.upload_gaussians(&scene);
        let check_output = rasterizer.forward(&scene, &render_camera).expect("check");
        let check_loss = mse_loss(&check_output.color_data, &target);
        println!(
            "Determinism check: base={:.10e}, check={:.10e}",
            base_loss, check_loss
        );
        println!("  delta={:.6e}", (check_loss - base_loss).abs());

        let grad_image = mse_grad_image(&base_output.color_data, &target);
        let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

        println!("=== GPU Self-Consistent ALL PARAMS Gradient Test ===");
        println!("Base loss: {:.6e}", base_loss);
        println!();

        let mut results: Vec<(&str, GpuGradVerifyResult)> = Vec::new();

        // --- Opacity ---
        {
            let mut max_err = 0.0f32;
            let mut sum_err = 0.0f32;
            let mut cnt = 0usize;
            let mut errors_vec = Vec::new();
            println!("--- Opacity ---");
            for i in 0..scene.len() {
                // Forward perturbation
                let mut perturbed_plus = scene.clone();
                perturbed_plus.gaussians[i].opacity += epsilon;
                let loss_plus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                        .expect("fwd+");

                // Backward perturbation
                let mut perturbed_minus = scene.clone();
                perturbed_minus.gaussians[i].opacity -= epsilon;
                let loss_minus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_minus, &render_camera, &target)
                        .expect("fwd-");

                let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                let analytic = analytical.grad_opacities[i];
                let err = gradient_error(analytic, numerical);
                println!(
                    "  [{}] num={:.6e} ana={:.6e} err={:.6e}",
                    i, numerical, analytic, err
                );
                if err > 0.1 {
                    println!(
                        "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                        loss_plus,
                        loss_minus,
                        loss_plus - loss_minus
                    );
                }
                errors_vec.push(err);
                max_err = max_err.max(err);
                sum_err += err;
                cnt += 1;
            }
            let median_err = median_error(&mut errors_vec);
            let num_outliers = errors_vec.iter().filter(|&&e| e > 0.5).count();
            results.push((
                "opacity",
                GpuGradVerifyResult {
                    max_error: max_err,
                    mean_error: if cnt > 0 { sum_err / cnt as f32 } else { 0.0 },
                    median_error: median_err,
                    num_gradients: cnt,
                    num_outliers,
                },
            ));
        }

        // --- Position ---
        {
            let mut max_err = 0.0f32;
            let mut sum_err = 0.0f32;
            let mut cnt = 0usize;
            let mut errors_vec = Vec::new();
            let axes = ["x", "y", "z"];
            println!("--- Position ---");
            for i in 0..scene.len() {
                for (axis, axis_name) in axes.iter().enumerate() {
                    // Forward perturbation
                    let mut perturbed_plus = scene.clone();
                    perturbed_plus.gaussians[i].position[axis] += epsilon;
                    let loss_plus =
                        gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                            .expect("fwd+");

                    // Backward perturbation
                    let mut perturbed_minus = scene.clone();
                    perturbed_minus.gaussians[i].position[axis] -= epsilon;
                    let loss_minus = gpu_forward_loss(
                        &mut rasterizer,
                        &perturbed_minus,
                        &render_camera,
                        &target,
                    )
                    .expect("fwd-");

                    let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                    let analytic = analytical.grad_positions[i][axis];
                    let err = gradient_error(analytic, numerical);
                    println!(
                        "  [{}.{}] num={:.6e} ana={:.6e} err={:.6e}",
                        i, axis_name, numerical, analytic, err
                    );
                    if err > 0.1 {
                        println!(
                            "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                            loss_plus,
                            loss_minus,
                            loss_plus - loss_minus
                        );
                    }
                    errors_vec.push(err);
                    max_err = max_err.max(err);
                    sum_err += err;
                    cnt += 1;
                }
            }
            let median_err = median_error(&mut errors_vec);
            let num_outliers = errors_vec.iter().filter(|&&e| e > 0.5).count();
            results.push((
                "position",
                GpuGradVerifyResult {
                    max_error: max_err,
                    mean_error: if cnt > 0 { sum_err / cnt as f32 } else { 0.0 },
                    median_error: median_err,
                    num_gradients: cnt,
                    num_outliers,
                },
            ));
        }

        // --- Scale ---
        {
            let mut max_err = 0.0f32;
            let mut sum_err = 0.0f32;
            let mut cnt = 0usize;
            let mut errors_vec = Vec::new();
            let axes = ["sx", "sy", "sz"];
            println!("--- Scale ---");
            for i in 0..scene.len() {
                for (axis, axis_name) in axes.iter().enumerate() {
                    // Forward perturbation
                    let mut perturbed_plus = scene.clone();
                    perturbed_plus.gaussians[i].scale[axis] += epsilon;
                    let loss_plus =
                        gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                            .expect("fwd+");

                    // Backward perturbation
                    let mut perturbed_minus = scene.clone();
                    perturbed_minus.gaussians[i].scale[axis] -= epsilon;
                    let loss_minus = gpu_forward_loss(
                        &mut rasterizer,
                        &perturbed_minus,
                        &render_camera,
                        &target,
                    )
                    .expect("fwd-");

                    let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                    let analytic = analytical.grad_scales[i][axis];
                    let err = gradient_error(analytic, numerical);
                    println!(
                        "  [{}.{}] num={:.6e} ana={:.6e} err={:.6e}",
                        i, axis_name, numerical, analytic, err
                    );
                    if err > 0.1 {
                        println!(
                            "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                            loss_plus,
                            loss_minus,
                            loss_plus - loss_minus
                        );
                    }
                    errors_vec.push(err);
                    max_err = max_err.max(err);
                    sum_err += err;
                    cnt += 1;
                }
            }
            let median_err = median_error(&mut errors_vec);
            let num_outliers = errors_vec.iter().filter(|&&e| e > 0.5).count();
            results.push((
                "scale",
                GpuGradVerifyResult {
                    max_error: max_err,
                    mean_error: if cnt > 0 { sum_err / cnt as f32 } else { 0.0 },
                    median_error: median_err,
                    num_gradients: cnt,
                    num_outliers,
                },
            ));
        }

        // --- Rotation ---
        {
            let mut max_err = 0.0f32;
            let mut sum_err = 0.0f32;
            let mut cnt = 0usize;
            let mut errors_vec = Vec::new();
            let comps = ["qx", "qy", "qz", "qw"];
            println!("--- Rotation ---");
            for i in 0..scene.len() {
                for (comp, comp_name) in comps.iter().enumerate() {
                    // Forward perturbation
                    let mut perturbed_plus = scene.clone();
                    perturbed_plus.gaussians[i].rotation[comp] += epsilon;
                    let loss_plus =
                        gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                            .expect("fwd+");

                    // Backward perturbation
                    let mut perturbed_minus = scene.clone();
                    perturbed_minus.gaussians[i].rotation[comp] -= epsilon;
                    let loss_minus = gpu_forward_loss(
                        &mut rasterizer,
                        &perturbed_minus,
                        &render_camera,
                        &target,
                    )
                    .expect("fwd-");

                    let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                    let analytic = analytical.grad_rotations[i][comp];
                    let err = gradient_error(analytic, numerical);
                    println!(
                        "  [{}.{}] num={:.6e} ana={:.6e} err={:.6e}",
                        i, comp_name, numerical, analytic, err
                    );
                    if err > 0.1 {
                        println!(
                            "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                            loss_plus,
                            loss_minus,
                            loss_plus - loss_minus
                        );
                    }
                    errors_vec.push(err);
                    max_err = max_err.max(err);
                    sum_err += err;
                    cnt += 1;
                }
            }
            let median_err = median_error(&mut errors_vec);
            let num_outliers = errors_vec.iter().filter(|&&e| e > 0.5).count();
            results.push((
                "rotation",
                GpuGradVerifyResult {
                    max_error: max_err,
                    mean_error: if cnt > 0 { sum_err / cnt as f32 } else { 0.0 },
                    median_error: median_err,
                    num_gradients: cnt,
                    num_outliers,
                },
            ));
        }

        // --- SH coefficients ---
        {
            let total_sh = sh_coeffs_per_gaussian * num_gaussians;
            let mut max_err = 0.0f32;
            let mut sum_err = 0.0f32;
            let mut cnt = 0usize;
            let mut errors_vec = Vec::new();
            println!("--- SH0 ---");
            for coeff_idx in 0..total_sh {
                let g_idx = coeff_idx / sh_coeffs_per_gaussian;
                let l_idx = coeff_idx % sh_coeffs_per_gaussian;
                let ch = ["R", "G", "B"][l_idx % 3];

                // Forward perturbation
                let mut perturbed_plus = scene.clone();
                perturbed_plus.sh_coeffs[coeff_idx] += epsilon;
                let loss_plus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_plus, &render_camera, &target)
                        .expect("fwd+");

                // Backward perturbation
                let mut perturbed_minus = scene.clone();
                perturbed_minus.sh_coeffs[coeff_idx] -= epsilon;
                let loss_minus =
                    gpu_forward_loss(&mut rasterizer, &perturbed_minus, &render_camera, &target)
                        .expect("fwd-");

                let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                let analytic = analytical.grad_sh_coeffs[coeff_idx];
                let err = gradient_error(analytic, numerical);
                println!(
                    "  [g={},l={},ch={}] num={:.6e} ana={:.6e} err={:.6e}",
                    g_idx,
                    l_idx / 3,
                    ch,
                    numerical,
                    analytic,
                    err
                );
                if err > 0.1 {
                    println!(
                        "    OUTLIER: loss_plus={:.10e}, loss_minus={:.10e}, delta={:.6e}",
                        loss_plus,
                        loss_minus,
                        loss_plus - loss_minus
                    );
                }
                errors_vec.push(err);
                max_err = max_err.max(err);
                sum_err += err;
                cnt += 1;
            }
            let median_err = median_error(&mut errors_vec);
            let num_outliers = errors_vec.iter().filter(|&&e| e > 0.5).count();
            results.push((
                "sh0",
                GpuGradVerifyResult {
                    max_error: max_err,
                    mean_error: if cnt > 0 { sum_err / cnt as f32 } else { 0.0 },
                    median_error: median_err,
                    num_gradients: cnt,
                    num_outliers,
                },
            ));
        }

        // --- Summary ---
        println!();
        println!("=== SUMMARY ===");
        let mut any_failed = false;
        for (name, result) in &results {
            let max_outliers = (result.num_gradients as f32 * 0.3).ceil() as usize;
            // Position gradients through a tiled rasterizer have higher finite-difference error
            // because position perturbation directly affects tile assignment, causing discontinuities
            // in the forward pass that the backward pass (correctly) doesn't model.
            let param_threshold = match *name {
                "position" => 2.5e-1,
                _ => median_threshold,
            };
            let status =
                if result.median_error < param_threshold && result.num_outliers <= max_outliers {
                    "PASS"
                } else {
                    any_failed = true;
                    "FAIL"
                };
            println!(
                "  {:<10} max_err={:.6e}  median_err={:.6e}  mean_err={:.6e}  n={:3}  outliers={:2}/{:2}  [{}]",
                name, result.max_error, result.median_error, result.mean_error, result.num_gradients, result.num_outliers, max_outliers, status
            );
        }
        println!();

        assert!(
            !any_failed,
            "One or more parameter types failed GPU self-consistent gradient verification (median + outlier metric)"
        );
    });
}
