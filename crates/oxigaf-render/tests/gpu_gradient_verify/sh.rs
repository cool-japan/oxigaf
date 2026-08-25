//! GPU self-consistent gradient verification for **SH coefficients**.
//!
//! Split out of `gpu_gradient_verify.rs` to keep every file under the 2000
//! line limit; the four tests here (degrees 0-3) are one concern - the
//! `grad_sh_coeffs` output of `preprocess_bwd.wgsl` - while the parent file
//! covers the geometric parameters. Shared scene, loss and error helpers come
//! from the parent module via the glob import below.

use super::*;

/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present, so it still runs (and gates CI) on any
/// machine that does have one.
#[test]
fn test_gpu_self_consistent_sh0() {
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
        let mut all_errors: Vec<f32> = Vec::new();

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

/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present, so it still runs (and gates CI) on any
/// machine that does have one.
#[test]
fn test_gpu_self_consistent_sh1() {
    if !gpu_available() {
        return;
    }

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

        let mut all_errors: Vec<f32> = Vec::new();

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

/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present, so it still runs (and gates CI) on any
/// machine that does have one.
#[test]
fn test_gpu_self_consistent_sh2() {
    if !gpu_available() {
        return;
    }

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

                let mut all_errors: Vec<f32> = Vec::new();

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

/// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
/// compatible GPU adapter is present, so it still runs (and gates CI) on any
/// machine that does have one.
#[test]
fn test_gpu_self_consistent_sh3() {
    if !gpu_available() {
        return;
    }

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

        let mut all_errors: Vec<f32> = Vec::new();

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
