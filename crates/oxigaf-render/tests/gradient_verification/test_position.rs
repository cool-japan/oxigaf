//! Position gradient verification tests.
//!
//! This module tests gradients with respect to Gaussian positions (x, y, z).
//!
//! Test strategy:
//! - Compute numerical gradients via finite-difference
//! - Compute analytical gradients from GPU backward pass
//! - Assert relative error < 1e-3

#[cfg(test)]
mod tests {
    use crate::gradient_verification::*;
    use oxigaf_render::config::RasterConfig;

    /// Helper function to verify position gradients match between numerical and analytical.
    fn verify_position_gradients(
        model: &oxigaf_render::gaussian::GaussianModel,
        camera: &CpuCamera,
        target: &[f32],
        config: &RasterConfig,
        fd_config: &FiniteDiffConfig,
    ) -> (GradientVerificationResult, f32, usize) {
        let loss_fn = MseLoss;

        // Compute numerical gradients
        let numerical_grads =
            compute_position_gradients(model, config, camera, target, &loss_fn, fd_config)
                .expect("Failed to compute numerical gradients");

        // Compute analytical gradients from GPU
        let analytical_grads_struct =
            compute_analytical_gradients_sync(model, camera, target, config)
                .expect("Failed to compute analytical gradients");
        let analytical_grads = analytical_grads_struct.grad_positions;

        // DEBUG: Print first 3 gradients
        println!("Numerical gradients (first 3):");
        for (i, grad) in numerical_grads.iter().take(3).enumerate() {
            println!("  [{i}]: [{:.6}, {:.6}, {:.6}]", grad[0], grad[1], grad[2]);
        }
        println!("Analytical gradients (first 3):");
        for (i, grad) in analytical_grads.iter().take(3).enumerate() {
            println!("  [{i}]: [{:.6}, {:.6}, {:.6}]", grad[0], grad[1], grad[2]);
        }

        // Compare gradients
        assert_eq!(
            numerical_grads.len(),
            analytical_grads.len(),
            "Gradient count mismatch"
        );

        let mut errors = compare_gradients_3d(&analytical_grads, &numerical_grads);
        let median = median_error(&mut errors);
        let num_outliers = errors.iter().filter(|&&e| e > 0.5).count();
        let result = GradientVerificationResult::new(&errors, POSITION_MEDIAN_ERROR_THRESHOLD);
        (result, median, num_outliers)
    }

    /// Verify finite-difference position gradients are well-defined for a
    /// simple scene. This is a fast, GPU-free sanity check on the numerical
    /// estimator only - it does not verify the GPU backward pass against it.
    /// See `test_position_analytical_vs_numerical` for that comparison,
    /// which is skipped (not `#[ignore]`d) via `gpu_available()` when no
    /// GPU is present rather than being permanently disabled.
    #[test]
    fn test_position_numerical_gradients_are_finite() {
        let scene_config = TestSceneConfig {
            num_gaussians: 3,
            resolution: (64, 64),
            sh_degree: 0,
            seed: 42,
        };

        let model = create_test_scene(&scene_config).expect("Failed to create test scene");
        let camera = create_test_camera(scene_config.resolution);
        let target = create_target_image(scene_config.resolution);

        let config = RasterConfig::new()
            .with_resolution(scene_config.resolution.0, scene_config.resolution.1)
            .with_sh_degree(scene_config.sh_degree);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        // Compute numerical gradients
        let numerical_grads =
            compute_position_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute position gradients");

        // Verify we got the right number of gradients
        assert_eq!(numerical_grads.len(), model.len());

        // This only checks that the finite-difference estimator itself
        // produces well-defined numbers, not that the GPU backward pass
        // agrees with it.
        for (i, grad) in numerical_grads.iter().enumerate() {
            assert!(
                grad[0].is_finite(),
                "Gradient {i} x-component is not finite"
            );
            assert!(
                grad[1].is_finite(),
                "Gradient {i} y-component is not finite"
            );
            assert!(
                grad[2].is_finite(),
                "Gradient {i} z-component is not finite"
            );
        }
    }

    /// Test position gradients with a single Gaussian.
    #[test]
    fn test_position_gradients_single_gaussian() {
        let scene_config = TestSceneConfig {
            num_gaussians: 1,
            resolution: (64, 64),
            sh_degree: 0,
            seed: 123,
        };

        let model = create_test_scene(&scene_config).expect("Failed to create test scene");
        let camera = create_test_camera(scene_config.resolution);
        let target = create_target_image(scene_config.resolution);

        let config = RasterConfig::new()
            .with_resolution(scene_config.resolution.0, scene_config.resolution.1)
            .with_sh_degree(scene_config.sh_degree);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        let numerical_grads =
            compute_position_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute position gradients");

        assert_eq!(numerical_grads.len(), 1);

        // Gradients should be non-zero for a visible Gaussian
        let grad = numerical_grads[0];
        let grad_norm = (grad[0].powi(2) + grad[1].powi(2) + grad[2].powi(2)).sqrt();
        assert!(grad_norm > 1e-8, "Gradient is too small: {grad_norm}");
    }

    /// Test position gradients with Gaussian behind camera (should be culled).
    #[test]
    fn test_position_gradients_behind_camera() {
        use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

        // Create a Gaussian behind the camera (positive Z)
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, 1.0], // Behind camera
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0], // Identity
            scale: [-1.0, -1.0, -1.0],      // exp(-1) ≈ 0.37
            opacity: 0.0,
        };

        let model = GaussianModel {
            gaussians: vec![gaussian],
            sh_coeffs: vec![0.5, 0.5, 0.5], // Gray
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0; 3]],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![false],
        };

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));

        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(0);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        let numerical_grads =
            compute_position_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute position gradients");

        // Gradient should be very small (Gaussian is culled)
        let grad = numerical_grads[0];
        let grad_norm = (grad[0].powi(2) + grad[1].powi(2) + grad[2].powi(2)).sqrt();
        assert!(
            grad_norm < 0.1,
            "Gradient should be small for culled Gaussian: {grad_norm}"
        );
    }

    /// Test position gradients with multiple overlapping Gaussians.
    #[test]
    fn test_position_gradients_overlapping() {
        use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

        // Create three overlapping Gaussians at the same position
        let base_gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, -1.0, -1.0],
            opacity: 0.0,
        };

        let model = GaussianModel {
            gaussians: vec![base_gaussian, base_gaussian, base_gaussian],
            sh_coeffs: vec![
                0.5, 0.5, 0.5, // G1: gray
                0.5, 0.5, 0.5, // G2: gray
                0.5, 0.5, 0.5, // G3: gray
            ],
            sh_degree: 0,
            face_indices: vec![0; 3],
            barycentric: vec![[1.0 / 3.0; 3]; 3],
            local_offsets: vec![[0.0; 3]; 3],
            is_rigid: vec![false; 3],
        };

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));

        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(0);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        let numerical_grads =
            compute_position_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute position gradients");

        assert_eq!(numerical_grads.len(), 3);

        // All gradients should be similar (same position)
        for grad in &numerical_grads {
            assert!(grad[0].is_finite());
            assert!(grad[1].is_finite());
            assert!(grad[2].is_finite());
        }
    }

    /// Test position gradients at image boundaries.
    #[test]
    fn test_position_gradients_at_boundaries() {
        use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

        // Create Gaussians at corners of the image
        let positions = [
            [-2.0, -2.0, -3.0], // Top-left
            [2.0, -2.0, -3.0],  // Top-right
            [-2.0, 2.0, -3.0],  // Bottom-left
            [2.0, 2.0, -3.0],   // Bottom-right
        ];

        let gaussians: Vec<_> = positions
            .iter()
            .map(|&pos| GaussianAttributes {
                position: pos,
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-1.0, -1.0, -1.0],
                opacity: 0.0,
            })
            .collect();

        let sh_coeffs = vec![0.5; 12]; // 4 Gaussians × 3 channels
        let n = gaussians.len();

        let model = GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree: 0,
            face_indices: vec![0; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        };

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));

        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(0);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        let numerical_grads =
            compute_position_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute position gradients");

        assert_eq!(numerical_grads.len(), 4);

        // All gradients should be finite
        for (i, grad) in numerical_grads.iter().enumerate() {
            assert!(grad[0].is_finite(), "Gradient {i} x is not finite");
            assert!(grad[1].is_finite(), "Gradient {i} y is not finite");
            assert!(grad[2].is_finite(), "Gradient {i} z is not finite");
        }
    }

    /// GRADIENT VERIFICATION TEST: Compare analytical vs numerical position gradients.
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when
    /// no compatible GPU adapter is present, so it still runs (and gates
    /// CI) on any machine that does have one.
    #[test]
    fn test_position_analytical_vs_numerical() {
        if !gpu_available() {
            return;
        }

        let scene_config = TestSceneConfig {
            num_gaussians: 5,
            resolution: (64, 64),
            sh_degree: 0,
            seed: 42,
        };

        let model = create_test_scene(&scene_config).expect("Failed to create test scene");
        let camera = create_test_camera(scene_config.resolution);
        let target = create_target_image(scene_config.resolution);

        let config = RasterConfig::new()
            .with_resolution(scene_config.resolution.0, scene_config.resolution.1)
            .with_sh_degree(scene_config.sh_degree);

        let fd_config = FiniteDiffConfig::default();

        let (result, median_err, num_outliers) =
            verify_position_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < POSITION_MEDIAN_ERROR_THRESHOLD,
            "Position gradient median error too high: {:.6e}, max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Position too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );

        println!("Position Gradient Verification:");
        println!("  Median Err: {:.6e}", median_err);
        println!("  Max Error:  {:.6e}", result.max_error);
        println!("  Mean Error: {:.6e}", result.mean_error);
        println!("  Outliers:   {}/{}", num_outliers, max_outlier_count);
        println!("  Status:     PASS");
    }

    /// Test position gradients with 10 Gaussians.
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when
    /// no compatible GPU adapter is present.
    #[test]
    fn test_position_gradients_10_gaussians() {
        if !gpu_available() {
            return;
        }

        let scene_config = TestSceneConfig {
            num_gaussians: 10,
            resolution: (128, 128),
            sh_degree: 0,
            seed: 12345,
        };

        let model = create_test_scene(&scene_config).expect("Failed to create test scene");
        let camera = create_test_camera(scene_config.resolution);
        let target = create_target_image(scene_config.resolution);

        let config = RasterConfig::new()
            .with_resolution(scene_config.resolution.0, scene_config.resolution.1)
            .with_sh_degree(scene_config.sh_degree);

        let fd_config = FiniteDiffConfig::default();

        let (result, median_err, num_outliers) =
            verify_position_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < POSITION_MEDIAN_ERROR_THRESHOLD,
            "Position gradient verification (10 Gaussians) failed: median_error={:.6e}",
            median_err
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Position (10 Gaussians) too many outliers: {} out of {} (limit={})",
            num_outliers,
            result.num_gradients,
            max_outlier_count
        );
    }

    /// Test position gradients with 100 Gaussians (stress test).
    ///
    /// Kept `#[ignore]`d for speed (unlike the other analytical-vs-numerical
    /// tests in this file), not merely because it needs a GPU; the reason
    /// string now says both, and it also self-skips via `gpu_available()`
    /// so `--ignored` runs on a machine without a GPU fail cleanly instead
    /// of hard-erroring out of `Rasterizer::new`.
    #[test]
    #[ignore = "slow test (100 Gaussians) and requires GPU hardware; run with --ignored"]
    fn test_position_gradients_100_gaussians() {
        if !gpu_available() {
            return;
        }

        let scene_config = TestSceneConfig {
            num_gaussians: 100,
            resolution: (128, 128),
            sh_degree: 0,
            seed: 99999,
        };

        let model = create_test_scene(&scene_config).expect("Failed to create test scene");
        let camera = create_test_camera(scene_config.resolution);
        let target = create_target_image(scene_config.resolution);

        let config = RasterConfig::new()
            .with_resolution(scene_config.resolution.0, scene_config.resolution.1)
            .with_sh_degree(scene_config.sh_degree);

        let fd_config = FiniteDiffConfig::default();

        let (result, median_err, num_outliers) =
            verify_position_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < POSITION_MEDIAN_ERROR_THRESHOLD,
            "Position gradient verification (100 Gaussians) failed: median_error={:.6e}",
            median_err
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Position (100 Gaussians) too many outliers: {} out of {} (limit={})",
            num_outliers,
            result.num_gradients,
            max_outlier_count
        );

        println!("Position Gradient Verification (100 Gaussians):");
        println!("  Median Err: {:.6e}", median_err);
        println!("  Max Error:  {:.6e}", result.max_error);
        println!("  Mean Error: {:.6e}", result.mean_error);
        println!("  Outliers:   {}/{}", num_outliers, max_outlier_count);
    }

    // -----------------------------------------------------------------
    // View-dependent (sh_degree >= 1) position gradients
    // -----------------------------------------------------------------

    /// `SH_C0` from `shaders/sh_eval.wgsl` / `src/cpu_reference.rs`.
    const SH_C0: f32 = 0.282_094_79;
    /// `SH_C1` from `shaders/sh_eval.wgsl` / `src/cpu_reference.rs`.
    const SH_C1: f32 = 0.488_602_52;

    /// Evaluate the degree-1 SH colour exactly as `cpu_reference::eval_sh_degree1`
    /// and `preprocess_sh1.wgsl` do: basis `(-y, z, -x)`, `+0.5` offset,
    /// clamped at zero.
    fn eval_sh1_color(dir: [f32; 3], sh: &[f32]) -> [f32; 3] {
        let [x, y, z] = dir;
        let mut out = [0.0f32; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = SH_C0 * sh[c]
                + SH_C1 * (-y) * sh[3 + c]
                + SH_C1 * z * sh[6 + c]
                + SH_C1 * (-x) * sh[9 + c]
                + 0.5;
            *o = o.max(0.0);
        }
        out
    }

    /// Unit view direction from the camera (at the origin in
    /// `create_test_camera`) towards `position`.
    fn view_dir_from_origin(position: [f32; 3]) -> [f32; 3] {
        let len =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        [position[0] / len, position[1] / len, position[2] / len]
    }

    /// Build a view-dependent scene with large degree-1 SH coefficients.
    ///
    /// The DC term stays bright enough that the degree-1 swing never drives a
    /// channel through the forward `max(color, 0)` clamp, which would zero the
    /// direction derivative and make the test vacuous.
    fn view_dependent_sh1_scene() -> oxigaf_render::gaussian::GaussianModel {
        use oxigaf_render::gaussian::GaussianAttributes;

        let positions = [
            [-0.8, 0.4, -3.0],
            [0.9, -0.5, -3.2],
            [0.2, 0.8, -2.8],
            [-0.6, -0.7, -3.4],
            [0.5, 0.1, -3.1],
        ];

        let mut gaussians = Vec::with_capacity(positions.len());
        let mut sh_coeffs = Vec::with_capacity(positions.len() * 12);

        for (i, position) in positions.iter().enumerate() {
            gaussians.push(GaussianAttributes {
                position: *position,
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-1.2, -1.2, -1.2],
                opacity: 0.5,
            });

            // DC: a bright, safely positive base colour.
            sh_coeffs.extend_from_slice(&[1.0, 0.8, 0.6]);
            // Degree 1: large and sign-varied, so d(colour)/d(dir) is far from
            // zero on every axis and differs per Gaussian.
            for k in 0..9usize {
                let sign = if (i + k).is_multiple_of(2) { 1.0 } else { -1.0 };
                sh_coeffs.push(sign * (0.5 + 0.05 * ((i * 9 + k) % 5) as f32));
            }
        }

        model_from_gaussians(gaussians, sh_coeffs, 1)
    }

    /// Build the *view-independent* scene that renders identically to
    /// `sh1_model` at its unperturbed positions.
    ///
    /// Each Gaussian gets a `sh_degree = 0` DC coefficient chosen so that
    /// `SH_C0 * dc + 0.5` equals the degree-1 colour evaluated at that
    /// Gaussian's actual view direction. The two scenes therefore produce the
    /// *same image* and the same loss, but only the degree-1 one has a colour
    /// that moves when the position moves.
    fn view_independent_twin(
        sh1_model: &oxigaf_render::gaussian::GaussianModel,
    ) -> oxigaf_render::gaussian::GaussianModel {
        let mut dc_coeffs = Vec::with_capacity(sh1_model.len() * 3);

        for (i, gaussian) in sh1_model.gaussians.iter().enumerate() {
            let dir = view_dir_from_origin(gaussian.position);
            let color = eval_sh1_color(dir, &sh1_model.sh_coeffs[i * 12..(i + 1) * 12]);
            for channel in color {
                // Invert `color = max(SH_C0 * dc + 0.5, 0)`.
                dc_coeffs.push((channel - 0.5) / SH_C0);
            }
        }

        model_from_gaussians(sh1_model.gaussians.clone(), dc_coeffs, 0)
    }

    /// GRADIENT VERIFICATION TEST: position gradients at `sh_degree = 1`.
    ///
    /// At `sh_degree == 0` the SH colour is view-independent, so the
    /// `dir = normalize(pos - cam_pos)` -> `pos` chain in `preprocess_bwd.wgsl`
    /// (the "5d. SH view-direction gradient" block) contributes *exactly zero*
    /// and every other position test in this file leaves it unexercised. This
    /// test drives it with deliberately large degree-1 coefficients.
    ///
    /// It is self-proving rather than merely "green": it also builds the
    /// view-independent twin scene that renders to the same image, and asserts
    /// that the twin's numerical position gradient differs materially from the
    /// degree-1 one. That difference *is* the direction chain, so the
    /// analytical-vs-numerical comparison below cannot pass vacuously - a
    /// backward pass that omitted the term would match the twin, not the
    /// degree-1 gradient.
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
    /// compatible GPU adapter is present.
    #[test]
    fn test_position_analytical_vs_numerical_sh1_view_dependent() {
        if !gpu_available() {
            return;
        }

        let resolution = (64u32, 64u32);
        let model = view_dependent_sh1_scene();
        let twin = view_independent_twin(&model);
        let camera = create_test_camera(resolution);
        let target = create_target_image(resolution);

        let sh1_config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(1);
        let sh0_config = RasterConfig::new()
            .with_resolution(resolution.0, resolution.1)
            .with_sh_degree(0);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        // -- Materiality: the two scenes must render (near) identically ... --
        let cpu = oxigaf_render::CpuRasterizer::new(sh1_config.clone());
        let sh1_image = cpu
            .render(&model, &camera)
            .expect("degree-1 CPU render failed");
        let cpu_twin = oxigaf_render::CpuRasterizer::new(sh0_config.clone());
        let twin_image = cpu_twin
            .render(&twin, &camera)
            .expect("view-independent CPU render failed");
        let image_mse: f64 = sh1_image
            .color_data
            .iter()
            .zip(twin_image.color_data.iter())
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / sh1_image.color_data.len() as f64;
        assert!(
            image_mse < 1e-10,
            "twin scene must reproduce the degree-1 image (MSE={image_mse:.3e}); \
             the materiality check below is meaningless otherwise"
        );

        // -- ... but their position gradients must NOT agree. --
        let numerical_sh1 =
            compute_position_gradients(&model, &sh1_config, &camera, &target, &loss_fn, &fd_config)
                .expect("degree-1 numerical position gradients");
        let numerical_twin =
            compute_position_gradients(&twin, &sh0_config, &camera, &target, &loss_fn, &fd_config)
                .expect("view-independent numerical position gradients");

        let mut dir_term_errors = compare_gradients_3d(&numerical_sh1, &numerical_twin);
        let dir_term_median = median_error(&mut dir_term_errors);
        println!(
            "SH-direction chain materiality: median relative difference between \
             view-dependent and view-independent numerical position gradients = {dir_term_median:.6e}"
        );
        assert!(
            dir_term_median > 5e-2,
            "the degree-1 SH view-direction term is not material in this scene \
             (median difference {dir_term_median:.6e}); the verification below \
             would pass even with the dir->pos chain missing"
        );

        // -- The actual gradient verification at sh_degree = 1. --
        let (result, median_err, num_outliers) =
            verify_position_gradients(&model, &camera, &target, &sh1_config, &fd_config);

        assert!(
            median_err < POSITION_MEDIAN_ERROR_THRESHOLD,
            "Degree-1 position gradient median error too high: {:.6e}, \
             max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Degree-1 position too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );

        println!("Position Gradient Verification (sh_degree=1, view-dependent):");
        println!("  Median Err: {:.6e}", median_err);
        println!("  Max Error:  {:.6e}", result.max_error);
        println!("  Mean Error: {:.6e}", result.mean_error);
        println!("  Outliers:   {}/{}", num_outliers, max_outlier_count);
    }

    /// Regression test for the helpers above: the view-independent twin must
    /// reproduce the degree-1 colour exactly at the unperturbed position, or
    /// the materiality assertion in the test above compares two different
    /// images instead of isolating the direction chain.
    #[test]
    fn test_view_independent_twin_reproduces_sh1_color() {
        let model = view_dependent_sh1_scene();
        let twin = view_independent_twin(&model);

        assert_eq!(twin.sh_degree, 0);
        assert_eq!(twin.sh_coeffs.len(), model.len() * 3);

        for (i, gaussian) in model.gaussians.iter().enumerate() {
            let dir = view_dir_from_origin(gaussian.position);
            let sh1_color = eval_sh1_color(dir, &model.sh_coeffs[i * 12..(i + 1) * 12]);
            for (c, expected) in sh1_color.iter().enumerate() {
                // Degree-0 forward: color = max(SH_C0 * dc + 0.5, 0).
                let dc = twin.sh_coeffs[i * 3 + c];
                let actual = (SH_C0 * dc + 0.5).max(0.0);
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "Gaussian {i} channel {c}: twin colour {actual} != degree-1 colour {expected}"
                );
                // A clamped channel would zero the direction derivative and
                // make the whole test vacuous.
                assert!(
                    *expected > 1e-3,
                    "Gaussian {i} channel {c} is clamped at zero ({expected}); \
                     the degree-1 coefficients must stay inside the positive range"
                );
            }
        }
    }

    /// Test position gradients at different resolutions.
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when
    /// no compatible GPU adapter is present.
    #[test]
    fn test_position_gradients_different_resolutions() {
        if !gpu_available() {
            return;
        }

        let resolutions = vec![(64, 64), (128, 128), (256, 256)];

        for resolution in resolutions {
            let scene_config = TestSceneConfig {
                num_gaussians: 5,
                resolution,
                sh_degree: 0,
                seed: 42,
            };

            let model = create_test_scene(&scene_config).expect("Failed to create test scene");
            let camera = create_test_camera(scene_config.resolution);
            let target = create_target_image(scene_config.resolution);

            let config = RasterConfig::new()
                .with_resolution(scene_config.resolution.0, scene_config.resolution.1)
                .with_sh_degree(scene_config.sh_degree);

            let fd_config = FiniteDiffConfig::default();

            let (result, median_err, num_outliers) =
                verify_position_gradients(&model, &camera, &target, &config, &fd_config);

            assert!(
                median_err < POSITION_MEDIAN_ERROR_THRESHOLD,
                "Position gradient verification failed at resolution {:?}: median_error={:.6e}",
                resolution,
                median_err
            );
            let max_outlier_count =
                (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
            assert!(
                num_outliers <= max_outlier_count,
                "Position at resolution {:?} too many outliers: {} out of {} (limit={})",
                resolution,
                num_outliers,
                result.num_gradients,
                max_outlier_count
            );
        }
    }
}
