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

    // Every `GaussianModel` invariant is that the FLAME binding arrays are
    // parallel to `gaussians` (one entry per Gaussian) - code that indexes
    // them in lockstep (FLAME deform, density-control clone/split) panics or
    // reads misaligned data otherwise. These synthetic scenes have no real
    // mesh binding, so use the same "no binding" defaults as
    // `GaussianModel::load_ply` and `gradient_verification::create_test_scene`
    // rather than leaving the vectors empty.
    let n = gaussians.len();
    let third = 1.0_f32 / 3.0_f32;

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: vec![0u32; n],
        barycentric: vec![[third, third, third]; n],
        local_offsets: vec![[0.0, 0.0, 0.0]; n],
        is_rigid: vec![false; n],
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

/// Cached result of probing for a usable GPU adapter (see [`gpu_available`]).
static GPU_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether a compatible GPU adapter is available in this environment.
///
/// Every test in this file constructs a real [`Rasterizer`], which fails with
/// `RenderError::GpuInit`/`AdapterNotFound` on a machine with no compatible
/// adapter (e.g. many headless CI runners). Blanket-`#[ignore]`ing them for
/// that reason let the whole suite report green without a single backward
/// shader ever running, so each test calls this at the top and returns early
/// when it is `false` instead - the tests then actually execute (and gate CI)
/// on any machine that does have a GPU.
///
/// The probe constructs a full `Rasterizer`, so the result is cached in a
/// [`std::sync::OnceLock`], which is safe under the harness's default parallel
/// execution.
fn gpu_available() -> bool {
    *GPU_AVAILABLE.get_or_init(|| {
        match pollster::block_on(Rasterizer::new(RasterConfig::new())) {
            Ok(_) => true,
            Err(err) => {
                eprintln!(
                    "skipping GPU-dependent gradient test: no compatible GPU adapter available ({err})"
                );
                false
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Loss & error utilities
// ---------------------------------------------------------------------------

/// Compute RGB-only MSE loss (skip alpha channel).
///
/// `loss = sum_pixels( (R-Rt)^2 + (G-Gt)^2 + (B-Bt)^2 ) / (num_pixels * 3)`
///
/// The reduction accumulates in `f64` even though the inputs and the result
/// are `f32`. Every numerical gradient in this file is a central difference
/// `(loss_plus - loss_minus) / (2 * epsilon)`, i.e. a *catastrophic
/// cancellation* of two nearly equal losses; a naive `sum::<f32>()` over the
/// tens of thousands of pixel terms accumulates enough rounding error to
/// swamp that difference. The effect is invisible against a black background
/// (the loss is tiny) but dominates once a non-zero background lifts the
/// whole image - see `test_gpu_self_consistent_nonzero_background`.
fn mse_loss(color_data: &[f32], target: &[f32]) -> f32 {
    let num_pixels = color_data.len() / 4;
    let rgb_count = (num_pixels * 3) as f64;
    let sum: f64 = color_data
        .chunks(4)
        .zip(target.chunks(4))
        .map(|(c, t)| {
            ((c[0] - t[0]) as f64).powi(2)
                + ((c[1] - t[1]) as f64).powi(2)
                + ((c[2] - t[2]) as f64).powi(2)
        })
        .sum();
    (sum / rgb_count) as f32
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
/// On success the vector is left sorted in-place. Returns `f32::NAN` -
/// without sorting - if `errors` is empty or contains any non-finite entry,
/// so a downstream `median_err < THRESHOLD` assertion always fails on either
/// condition instead of silently reporting a passing median.
///
/// This mirrors `gradient_verification::median_error`; the two suites live in
/// separate test binaries and cannot share a module.
fn median_error(errors: &mut [f32]) -> f32 {
    if errors.is_empty() {
        // Nothing was actually compared. Returning a "clean" 0.0 here would
        // let every `median_err < THRESHOLD` assertion silently pass on an
        // empty comparison, so surface NaN instead (NaN compares `false`
        // against everything, including `<`).
        return f32::NAN;
    }
    if errors.iter().any(|e| e.is_nan()) {
        // `partial_cmp` returns `None` for any comparison involving NaN, so
        // the `unwrap_or(Ordering::Equal)` below treats a NaN as tied with
        // every other value and lets `sort_by` place it anywhere - silently
        // picking an arbitrary "median" instead of surfacing that a backward
        // shader produced a non-finite gradient.
        return f32::NAN;
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

/// Summarize a batch of per-gradient errors into a [`GpuGradVerifyResult`].
///
/// Every statistic is deliberately NaN-poisoning, because a backward shader
/// that emits a non-finite gradient is a bug rather than an out-of-tolerance
/// value, and because comparing *nothing* must not look like a perfect match:
///
/// * `errors` empty, or any entry non-finite -> all three error statistics are
///   `f32::NAN`, so every `< THRESHOLD` assertion evaluates to `false`.
/// * a NaN entry also counts as an outlier: `e > 0.5` is `false` for NaN, so
///   the plain predicate used to under-count them.
///
/// The hand-rolled accumulators this replaces did the opposite on all three
/// counts: `max_error.max(err)` discards NaN (`f32::max` returns the non-NaN
/// operand), `median_error` returned `0.0` for an empty vector, and
/// `if count > 0 { sum / count } else { 0.0 }` reported a mean of zero for an
/// empty comparison - so a run that compared nothing, or that produced NaN
/// gradients, reported PASS.
fn summarize_errors(errors: &mut [f32]) -> GpuGradVerifyResult {
    let num_gradients = errors.len();
    let num_outliers = errors.iter().filter(|e| e.is_nan() || **e > 0.5).count();

    if num_gradients == 0 || errors.iter().any(|e| !e.is_finite()) {
        return GpuGradVerifyResult {
            max_error: f32::NAN,
            mean_error: f32::NAN,
            median_error: f32::NAN,
            num_gradients,
            num_outliers,
        };
    }

    let max_error = errors.iter().copied().fold(0.0f32, f32::max);
    let mean_error = errors.iter().sum::<f32>() / num_gradients as f32;
    let median = median_error(errors);

    GpuGradVerifyResult {
        max_error,
        mean_error,
        median_error: median,
        num_gradients,
        num_outliers,
    }
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

/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present, so it still runs (and gates CI) on any
/// machine that does have one.
#[test]
fn test_gpu_self_consistent_opacity() {
    if !gpu_available() {
        return;
    }

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
        let mut all_errors: Vec<f32> = Vec::new();

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
        }

        // Shared, NaN-poisoning summary: an empty comparison or a
        // non-finite gradient must fail every threshold below, not
        // report a clean 0.0 (see `summarize_errors`).
        let summary = summarize_errors(&mut all_errors);
        let max_error = summary.max_error;
        let mean_error = summary.mean_error;
        let median_err = summary.median_error;
        let count = summary.num_gradients;
        let num_outliers = summary.num_outliers;
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

/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present, so it still runs (and gates CI) on any
/// machine that does have one.
#[test]
fn test_gpu_self_consistent_position() {
    if !gpu_available() {
        return;
    }

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
        let mut all_errors: Vec<f32> = Vec::new();
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
            }
        }

        // Shared, NaN-poisoning summary: an empty comparison or a
        // non-finite gradient must fail every threshold below, not
        // report a clean 0.0 (see `summarize_errors`).
        let summary = summarize_errors(&mut all_errors);
        let max_error = summary.max_error;
        let mean_error = summary.mean_error;
        let median_err = summary.median_error;
        let count = summary.num_gradients;
        let num_outliers = summary.num_outliers;
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

/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present, so it still runs (and gates CI) on any
/// machine that does have one.
#[test]
fn test_gpu_self_consistent_scale() {
    if !gpu_available() {
        return;
    }

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
        let mut all_errors: Vec<f32> = Vec::new();
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
            }
        }

        // Shared, NaN-poisoning summary: an empty comparison or a
        // non-finite gradient must fail every threshold below, not
        // report a clean 0.0 (see `summarize_errors`).
        let summary = summarize_errors(&mut all_errors);
        let max_error = summary.max_error;
        let mean_error = summary.mean_error;
        let median_err = summary.median_error;
        let count = summary.num_gradients;
        let num_outliers = summary.num_outliers;
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

/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present, so it still runs (and gates CI) on any
/// machine that does have one.
#[test]
fn test_gpu_self_consistent_rotation() {
    if !gpu_available() {
        return;
    }

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
        let mut all_errors: Vec<f32> = Vec::new();
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
            }
        }

        // Shared, NaN-poisoning summary: an empty comparison or a
        // non-finite gradient must fail every threshold below, not
        // report a clean 0.0 (see `summarize_errors`).
        let summary = summarize_errors(&mut all_errors);
        let max_error = summary.max_error;
        let mean_error = summary.mean_error;
        let median_err = summary.median_error;
        let count = summary.num_gradients;
        let num_outliers = summary.num_outliers;
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

/// GPU self-consistent gradient verification for the SH coefficients
/// (degrees 0-3). Kept in its own file so this one stays under the 2000
/// line limit.
///
/// `#[path]` is required: this file is an integration-test *crate root*, so a
/// bare `mod sh;` would resolve to `tests/sh.rs` - which Cargo would then
/// auto-discover as a separate test target - rather than to a subdirectory
/// named after this file.
#[path = "gpu_gradient_verify/sh.rs"]
mod sh;

/// Combined test that verifies all parameter types in a single GPU session.
///
/// This is more efficient than running individual tests because it only
/// initializes the GPU rasterizer once and shares the base forward pass.
/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present, so it still runs (and gates CI) on any
/// machine that does have one.
#[test]
fn test_gpu_self_consistent_all_params() {
    if !gpu_available() {
        return;
    }

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
            let mut errors_vec: Vec<f32> = Vec::new();
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
            }
            // Shared, NaN-poisoning summary (see `summarize_errors`).
            results.push(("opacity", summarize_errors(&mut errors_vec)));
        }

        // --- Position ---
        {
            let mut errors_vec: Vec<f32> = Vec::new();
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
                }
            }
            // Shared, NaN-poisoning summary (see `summarize_errors`).
            results.push(("position", summarize_errors(&mut errors_vec)));
        }

        // --- Scale ---
        {
            let mut errors_vec: Vec<f32> = Vec::new();
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
                }
            }
            // Shared, NaN-poisoning summary (see `summarize_errors`).
            results.push(("scale", summarize_errors(&mut errors_vec)));
        }

        // --- Rotation ---
        {
            let mut errors_vec: Vec<f32> = Vec::new();
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
                }
            }
            // Shared, NaN-poisoning summary (see `summarize_errors`).
            results.push(("rotation", summarize_errors(&mut errors_vec)));
        }

        // --- SH coefficients ---
        {
            let total_sh = sh_coeffs_per_gaussian * num_gaussians;
            let mut errors_vec: Vec<f32> = Vec::new();
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
            }
            // Shared, NaN-poisoning summary (see `summarize_errors`).
            results.push(("sh0", summarize_errors(&mut errors_vec)));
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

// ---------------------------------------------------------------------------
// Non-zero background
// ---------------------------------------------------------------------------

/// A bright, deliberately non-grey background for the tests below.
///
/// `rasterize_bwd.wgsl` adds `(-T_final / (1 - alpha)) * dot(background, dL_dcolor)`
/// to `dL/dalpha`, because the forward pass finishes with
/// `color += T_final * background`. That whole term is **identically zero**
/// when `background == [0, 0, 0]`, which is `RasterConfig`'s default and what
/// every other test in this file uses - so without a non-zero background the
/// term ships unexercised.
const TEST_BACKGROUND: [f32; 3] = [0.4, 0.6, 0.8];

/// Central-difference numerical gradient of the GPU forward loss with respect
/// to one scalar parameter, selected by `perturb`.
///
/// Both the plus and the minus render go through the *same* `rasterizer`, so
/// they share its `RasterConfig` - including its background - with the
/// analytical backward pass being checked.
fn numerical_gpu_grad<F>(
    rasterizer: &mut Rasterizer,
    scene: &GaussianModel,
    camera: &RenderCamera,
    target: &[f32],
    epsilon: f32,
    mut perturb: F,
) -> f32
where
    F: FnMut(&mut GaussianModel, f32),
{
    let mut plus = scene.clone();
    perturb(&mut plus, epsilon);
    let loss_plus = gpu_forward_loss(rasterizer, &plus, camera, target).expect("fwd+");

    let mut minus = scene.clone();
    perturb(&mut minus, -epsilon);
    let loss_minus = gpu_forward_loss(rasterizer, &minus, camera, target).expect("fwd-");

    (loss_plus - loss_minus) / (2.0 * epsilon)
}

/// Scene for the non-zero-background test: a few large, soft Gaussians.
///
/// The Gaussians are deliberately wide and low-opacity. `rasterize_fwd.wgsl`
/// truncates every Gaussian hard (it skips a pixel once `power < -4` or
/// `alpha < 1/255`), so any parameter that changes a Gaussian's *extent* -
/// scale, and depth through `position.z` - moves that truncation ring. Against
/// a black background the pixels on the ring are near-black either way and the
/// jump is invisible; against a bright background each ring pixel abruptly
/// attenuates `T_final * background`, so the finite difference picks up a
/// discontinuity the (correct) backward pass does not model. Wide discs keep
/// that O(circumference) boundary term small next to the O(area) smooth term,
/// and a low opacity shrinks the size of each individual jump.
fn background_test_scene() -> GaussianModel {
    let positions = [[-0.5, 0.35, -4.0], [0.6, -0.4, -4.3], [0.05, 0.0, -3.8]];

    let mut gaussians = Vec::with_capacity(positions.len());
    let mut sh_coeffs = Vec::with_capacity(positions.len() * 3);

    for position in positions {
        gaussians.push(GaussianAttributes {
            position,
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            // exp(0) = 1.0 world units: a wide disc at this depth.
            scale: [0.0, 0.0, 0.0],
            // sigmoid(-2) ~ 0.12: soft enough that the truncation ring jump is
            // small, opaque enough to move the loss.
            opacity: -2.0,
        });
        // Degree-0 colour ~0.2, clearly darker than TEST_BACKGROUND, so
        // covering a pixel changes the loss in a well-defined direction.
        sh_coeffs.extend_from_slice(&[-1.06, -1.06, -1.06]);
    }

    let n = gaussians.len();
    let third = 1.0_f32 / 3.0_f32;

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree: 0,
        face_indices: vec![0u32; n],
        barycentric: vec![[third, third, third]; n],
        local_offsets: vec![[0.0, 0.0, 0.0]; n],
        is_rigid: vec![false; n],
    }
}

/// GPU self-consistent gradient verification with a **non-zero background**.
///
/// Covers the `dL/dalpha` background term in `rasterize_bwd.wgsl` (which is
/// identically zero for the default black background) and, through the same
/// backward pass, the `mean2d` / `conic` chains under a background-shifted
/// `dL/dcolor`.
///
/// The test is built not to pass vacuously:
///
/// 1. It first asserts the forward pass really applied the background, by
///    checking an uncovered corner pixel against [`TEST_BACKGROUND`]. If the
///    background never reached the image, `T_final * background` is zero and
///    the backward term under test cannot be exercised at all.
/// 2. It then asserts the analytical opacity gradients differ *materially*
///    from the same scene's gradients with a black background. With a black
///    target and a bright background the term dominates `dL/dalpha`, so a
///    backward pass that dropped it would not merely be slightly off - it
///    would fail the comparison in step 3 outright.
/// 3. Finally it compares analytical against numerical (both from the GPU, so
///    no CPU/GPU forward discrepancy is folded in) for opacity, position and
///    scale.
///
/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present.
#[test]
fn test_gpu_self_consistent_nonzero_background() {
    if !gpu_available() {
        return;
    }

    pollster::block_on(async {
        let resolution = (128u32, 128u32);
        let sh_degree = 0u32;
        let epsilon = 5e-3f32;

        let bg_config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree)
            .with_background(TEST_BACKGROUND);
        let black_config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(sh_degree);

        let scene = background_test_scene();
        let num_gaussians = scene.len();
        let camera = create_test_camera(resolution);
        let render_camera = cpu_to_render_camera(&camera).expect("camera conversion");
        let target = vec![0.0f32; (resolution.0 * resolution.1 * 4) as usize];

        // -- Analytical gradients under the bright background --
        let mut rasterizer = Rasterizer::new(bg_config.clone())
            .await
            .expect("GPU rasterizer init");
        rasterizer.upload_gaussians(&scene);
        let base_output = rasterizer
            .forward(&scene, &render_camera)
            .expect("base forward");

        println!("=== GPU Self-Consistent NON-ZERO BACKGROUND Gradient Test ===");
        println!("Background: {:?}", TEST_BACKGROUND);
        println!(
            "Base loss:  {:.6e}",
            mse_loss(&base_output.color_data, &target)
        );

        let grad_image = mse_grad_image(&base_output.color_data, &target);
        let analytical = rasterizer.backward(&scene, &grad_image).expect("backward");

        // -- The same scene rendered with the default black background --
        let (black_output, black_analytical) = {
            let mut black_rasterizer = Rasterizer::new(black_config)
                .await
                .expect("GPU rasterizer init (black background)");
            black_rasterizer.upload_gaussians(&scene);
            let out = black_rasterizer
                .forward(&scene, &render_camera)
                .expect("base forward (black background)");
            let g = mse_grad_image(&out.color_data, &target);
            let grads = black_rasterizer
                .backward(&scene, &g)
                .expect("backward (black background)");
            (out, grads)
        };

        // (1) The forward pass must actually have applied the background.
        //
        // The forward finishes with `color += T_final * background`, so the
        // difference between the two renders is exactly `T_final * background`
        // per channel. Pixel (0, 0) is a frame corner where the Gaussians are
        // faint, so its transmittance is close to (but not exactly) 1 - hence
        // a *relative* check rather than an exact one. The black render's
        // corner must also be near zero, or the difference would not be
        // attributable to the background at all.
        for (channel, background) in TEST_BACKGROUND.iter().enumerate() {
            let bright = base_output.color_data[channel];
            let black = black_output.color_data[channel];
            let contributed = bright - black;
            println!(
                "  corner ch{channel}: bright={bright:.6} black={black:.6} \
                 background contribution={contributed:.6} (expected ~{background})"
            );
            assert!(
                black.abs() < 0.05,
                "corner pixel channel {channel} is {black} with a black background; \
                 it is not background-dominated, so the check below proves nothing"
            );
            assert!(
                (contributed - background).abs() < 0.05 * background,
                "the background contributed {contributed} to corner channel {channel}, \
                 expected ~{background}; the backward background term cannot be \
                 exercised if the forward never applied it"
            );
        }

        // (2) Materiality of the backward term.
        let mut bg_vs_black: Vec<f32> = analytical
            .grad_opacities
            .iter()
            .zip(black_analytical.grad_opacities.iter())
            .map(|(bright, black)| gradient_error(*bright, *black))
            .collect();
        let bg_influence = median_error(&mut bg_vs_black);
        println!(
            "Background influence on dL/dalpha (median relative difference vs black bg): {:.6e}",
            bg_influence
        );
        assert!(
            bg_influence > 0.5,
            "the background barely changes dL/dalpha (median difference {bg_influence:.6e}); \
             this scene does not exercise the background term meaningfully"
        );

        // -- (3) Analytical vs numerical, all through the bright-background config --
        let mut opacity_errors = Vec::with_capacity(num_gaussians);
        for i in 0..scene.len() {
            let numerical = numerical_gpu_grad(
                &mut rasterizer,
                &scene,
                &render_camera,
                &target,
                epsilon,
                |m, d| m.gaussians[i].opacity += d,
            );
            let err = gradient_error(analytical.grad_opacities[i], numerical);
            println!(
                "  opacity[{}] num={:.6e} ana={:.6e} err={:.6e}",
                i, numerical, analytical.grad_opacities[i], err
            );
            opacity_errors.push(err);
        }
        let opacity = summarize_errors(&mut opacity_errors);

        let mut position_errors = Vec::with_capacity(num_gaussians * 3);
        let mut scale_errors = Vec::with_capacity(num_gaussians * 3);
        for i in 0..scene.len() {
            for axis in 0..3usize {
                let numerical = numerical_gpu_grad(
                    &mut rasterizer,
                    &scene,
                    &render_camera,
                    &target,
                    epsilon,
                    |m, d| m.gaussians[i].position[axis] += d,
                );
                let err = gradient_error(analytical.grad_positions[i][axis], numerical);
                println!(
                    "  position[{}][{}] num={:.6e} ana={:.6e} err={:.6e}",
                    i, axis, numerical, analytical.grad_positions[i][axis], err
                );
                position_errors.push(err);

                let numerical = numerical_gpu_grad(
                    &mut rasterizer,
                    &scene,
                    &render_camera,
                    &target,
                    epsilon,
                    |m, d| m.gaussians[i].scale[axis] += d,
                );
                let err = gradient_error(analytical.grad_scales[i][axis], numerical);
                println!(
                    "  scale[{}][{}]    num={:.6e} ana={:.6e} err={:.6e}",
                    i, axis, numerical, analytical.grad_scales[i][axis], err
                );
                scale_errors.push(err);
            }
        }
        let position = summarize_errors(&mut position_errors);
        let scale = summarize_errors(&mut scale_errors);

        for (name, result, threshold) in [
            // Measured on this scene: 1.9e-3 median. The background term is
            // the dominant contribution to dL/dalpha here, so this number is
            // a direct check on it.
            ("opacity", &opacity, 5e-2f32),
            // Position gradients through a tiled rasterizer carry higher
            // finite-difference error (tile-assignment discontinuities), so
            // they get more headroom than the other two - but far less than
            // the 2.5e-1 the black-background position tests need, because
            // this scene's Gaussians are wide enough that the boundary terms
            // are a small fraction of the smooth interior term. Measured
            // median: 1.6e-2.
            ("position", &position, 1e-1f32),
            // Measured: 8.4e-3 median. The `max_error` on this row can still
            // approach 1.0 for a single near-zero component (the depth axis of
            // a screen-facing Gaussian), which is why the assertion is on the
            // median plus an outlier budget rather than on the maximum.
            ("scale", &scale, 5e-2f32),
        ] {
            let max_outliers = (result.num_gradients as f32 * 0.3).ceil() as usize;
            println!(
                "  {:<8} max_err={:.6e} median_err={:.6e} mean_err={:.6e} n={} outliers={}/{}",
                name,
                result.max_error,
                result.median_error,
                result.mean_error,
                result.num_gradients,
                result.num_outliers,
                max_outliers
            );
            assert!(
                result.median_error < threshold,
                "{name} gradient median error too high with a non-zero background: \
                 {:.6e} (max={:.6e})",
                result.median_error,
                result.max_error
            );
            assert!(
                result.num_outliers <= max_outliers,
                "{name} too many gradient outliers with a non-zero background: \
                 {} out of {} (limit={})",
                result.num_outliers,
                result.num_gradients,
                max_outliers
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Regression tests for the error-summary helpers (no GPU required)
// ---------------------------------------------------------------------------

/// Whether `value < threshold` evaluates to `true`.
///
/// Written with `partial_cmp` rather than `!(value < threshold)` so clippy's
/// `neg_cmp_op_on_partial_ord` lint stays satisfied: the whole point of these
/// regression tests is that a NaN makes the comparison *incomparable*, and
/// `partial_cmp` says so explicitly.
fn passes_threshold(value: f32, threshold: f32) -> bool {
    matches!(
        value.partial_cmp(&threshold),
        Some(std::cmp::Ordering::Less)
    )
}

/// Regression test: `median_error` must not let a NaN entry sort to an
/// arbitrary position and return a misleadingly "passing" median. It used to
/// call `partial_cmp(..).unwrap_or(Ordering::Equal)`, which treats NaN as tied
/// with everything.
#[test]
fn test_median_error_nan_propagates() {
    let mut errors = vec![0.1f32, f32::NAN, 0.2];
    assert!(median_error(&mut errors).is_nan());
}

/// Regression test: an empty error list must not report a "passing" median of
/// `0.0` - that let a run which compared nothing report PASS.
#[test]
fn test_median_error_empty_is_not_a_silent_pass() {
    let mut errors: Vec<f32> = Vec::new();
    assert!(median_error(&mut errors).is_nan());
}

#[test]
fn test_summarize_errors_normal_case() {
    let mut errors = vec![0.1f32, 0.2, 0.3, 0.9];
    let summary = summarize_errors(&mut errors);

    assert_eq!(summary.num_gradients, 4);
    assert_eq!(summary.num_outliers, 1);
    assert!((summary.max_error - 0.9).abs() < 1e-6);
    assert!((summary.mean_error - 0.375).abs() < 1e-6);
    assert!((summary.median_error - 0.25).abs() < 1e-6);
}

/// Regression test: `max_error.max(err)` silently discarded NaN (`f32::max`
/// returns the non-NaN operand), so a backward shader emitting a non-finite
/// gradient left `max_error` at the largest *finite* value and every
/// `< THRESHOLD` assertion passed.
#[test]
fn test_summarize_errors_nan_fails_every_threshold() {
    let mut errors = vec![1e-6f32, f32::NAN, 2e-6];
    let summary = summarize_errors(&mut errors);

    assert!(summary.max_error.is_nan());
    assert!(summary.mean_error.is_nan());
    assert!(summary.median_error.is_nan());
    assert!(
        !passes_threshold(summary.median_error, 5e-2),
        "a NaN gradient must fail the median threshold, not pass it"
    );
    // `e > 0.5` is `false` for NaN, so the old outlier predicate under-counted
    // non-finite gradients as well.
    assert_eq!(summary.num_outliers, 1);
}

/// Regression test: an infinite error must fail just like a NaN one.
#[test]
fn test_summarize_errors_infinity_fails() {
    let mut errors = vec![1e-6f32, f32::INFINITY];
    let summary = summarize_errors(&mut errors);

    assert!(summary.max_error.is_nan());
    assert!(!passes_threshold(summary.median_error, 5e-2));
}

/// Regression test: comparing nothing must not look like a perfect match.
/// `mean_error = if count > 0 { .. } else { 0.0 }` and a `0.0` median both
/// reported PASS for an empty run.
#[test]
fn test_summarize_errors_empty_is_not_a_silent_pass() {
    let mut errors: Vec<f32> = Vec::new();
    let summary = summarize_errors(&mut errors);

    assert_eq!(summary.num_gradients, 0);
    assert!(summary.max_error.is_nan());
    assert!(summary.mean_error.is_nan());
    assert!(summary.median_error.is_nan());
    assert!(!passes_threshold(summary.median_error, 5e-2));
}

/// Regression test: `create_test_scene` in this file must keep the FLAME
/// binding arrays parallel to `gaussians`; they used to be left empty, which
/// breaks the `GaussianModel` invariant that code indexing them in lockstep
/// (FLAME deform, density-control clone/split) relies on.
#[test]
fn test_create_test_scene_binding_arrays_match_gaussian_count() {
    let scene = create_test_scene(4, 1, 42);

    assert_eq!(scene.gaussians.len(), 4);
    assert_eq!(scene.face_indices.len(), 4);
    assert_eq!(scene.barycentric.len(), 4);
    assert_eq!(scene.local_offsets.len(), 4);
    assert_eq!(scene.is_rigid.len(), 4);
    // sh_degree 1 -> (1+1)^2 * 3 = 12 coefficients per Gaussian.
    assert_eq!(scene.sh_coeffs.len(), 4 * 12);
}
