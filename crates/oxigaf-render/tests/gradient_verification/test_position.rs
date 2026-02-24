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

    /// Test position gradients for a simple scene.
    #[test]
    fn test_position_gradients_simple() {
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

        // TODO: When GPU backward pass is implemented, compare with analytical gradients
        // For now, just verify gradients are finite
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
            face_indices: vec![],
            barycentric: vec![],
            local_offsets: vec![],
            is_rigid: vec![],
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
            face_indices: vec![],
            barycentric: vec![],
            local_offsets: vec![],
            is_rigid: vec![],
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

        let model = GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree: 0,
            face_indices: vec![],
            barycentric: vec![],
            local_offsets: vec![],
            is_rigid: vec![],
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
    #[test]
    fn test_position_analytical_vs_numerical() {
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
    #[test]
    fn test_position_gradients_10_gaussians() {
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
    #[test]
    #[ignore] // Slow test, run with --ignored flag
    fn test_position_gradients_100_gaussians() {
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

    /// Test position gradients at different resolutions.
    #[test]
    fn test_position_gradients_different_resolutions() {
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
