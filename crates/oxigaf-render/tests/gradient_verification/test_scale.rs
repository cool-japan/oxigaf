//! Scale gradient verification tests.
//!
//! This module tests gradients with respect to Gaussian scales (log-space).
//!
//! Test strategy:
//! - Compute numerical gradients via finite-difference
//! - Compute analytical gradients from GPU backward pass
//! - Assert relative error < 1e-3

#[cfg(test)]
mod tests {
    use crate::gradient_verification::*;
    use oxigaf_render::config::RasterConfig;
    use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

    /// Helper function to verify scale gradients match between numerical and analytical.
    fn verify_scale_gradients(
        model: &GaussianModel,
        camera: &CpuCamera,
        target: &[f32],
        config: &RasterConfig,
        fd_config: &FiniteDiffConfig,
    ) -> (GradientVerificationResult, f32, usize) {
        let loss_fn = MseLoss;

        // Compute numerical gradients
        let numerical_grads =
            compute_scale_gradients(model, config, camera, target, &loss_fn, fd_config)
                .expect("Failed to compute numerical gradients");

        // Compute analytical gradients from GPU
        let analytical_grads_struct =
            compute_analytical_gradients_sync(model, camera, target, config)
                .expect("Failed to compute analytical gradients");
        let analytical_grads = analytical_grads_struct.grad_scales;

        // Compare gradients
        assert_eq!(
            numerical_grads.len(),
            analytical_grads.len(),
            "Gradient count mismatch"
        );

        let mut errors = compare_gradients_3d(&analytical_grads, &numerical_grads);
        let median = median_error(&mut errors);
        let num_outliers = errors.iter().filter(|&&e| e > 0.5).count();
        let result = GradientVerificationResult::new(&errors, MEDIAN_ERROR_THRESHOLD);
        (result, median, num_outliers)
    }

    /// Test scale gradients for a simple scene.
    #[test]
    fn test_scale_gradients_simple() {
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
            compute_scale_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute scale gradients");

        // Verify we got the right number of gradients
        assert_eq!(numerical_grads.len(), model.len());

        // Verify gradients are finite
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

    /// Test scale gradients with isotropic scaling (sx = sy = sz).
    #[test]
    fn test_scale_gradients_isotropic() {
        // Create a Gaussian with isotropic scaling
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, -1.0, -1.0], // exp(-1) ≈ 0.37 for all axes
            opacity: 0.0,
        };

        // `model_from_gaussians` sizes the FLAME binding arrays to the
        // Gaussian count; a hand-written literal used to leave them empty.
        let model = model_from_gaussians(vec![gaussian], vec![0.5, 0.5, 0.5], 0);

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));

        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(0);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        let numerical_grads =
            compute_scale_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute scale gradients");

        assert_eq!(numerical_grads.len(), 1);

        // Gradients should be finite
        let grad = numerical_grads[0];
        assert!(grad[0].is_finite());
        assert!(grad[1].is_finite());
        assert!(grad[2].is_finite());
    }

    /// Test scale gradients with anisotropic scaling (sx >> sy, sz).
    #[test]
    fn test_scale_gradients_anisotropic() {
        // Create a Gaussian with highly anisotropic scaling
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.5, -2.0, -2.0], // sx >> sy, sz
            opacity: 0.0,
        };

        // `model_from_gaussians` sizes the FLAME binding arrays to the
        // Gaussian count; a hand-written literal used to leave them empty.
        let model = model_from_gaussians(vec![gaussian], vec![0.5, 0.5, 0.5], 0);

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));

        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(0);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        let numerical_grads =
            compute_scale_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute scale gradients");

        assert_eq!(numerical_grads.len(), 1);

        // Gradients should be finite even with anisotropic scaling
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            assert!(
                numerical_grads[0][i].is_finite(),
                "Gradient component {i} is not finite"
            );
        }
    }

    /// Test scale gradients with very small scales.
    #[test]
    fn test_scale_gradients_small_scales() {
        // Create a Gaussian with very small scales (log-space)
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-3.0, -3.0, -3.0], // exp(-3) ≈ 0.05
            opacity: 0.0,
        };

        // `model_from_gaussians` sizes the FLAME binding arrays to the
        // Gaussian count; a hand-written literal used to leave them empty.
        let model = model_from_gaussians(vec![gaussian], vec![0.5, 0.5, 0.5], 0);

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));

        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(0);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        let numerical_grads =
            compute_scale_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute scale gradients");

        assert_eq!(numerical_grads.len(), 1);

        // Gradients should be finite (may be small due to small scale)
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            assert!(
                numerical_grads[0][i].is_finite(),
                "Gradient component {i} is not finite"
            );
        }
    }

    /// Test scale gradients with large scales.
    #[test]
    fn test_scale_gradients_large_scales() {
        // Create a Gaussian with large scales
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0], // exp(1) ≈ 2.72
            opacity: 0.0,
        };

        // `model_from_gaussians` sizes the FLAME binding arrays to the
        // Gaussian count; a hand-written literal used to leave them empty.
        let model = model_from_gaussians(vec![gaussian], vec![0.5, 0.5, 0.5], 0);

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));

        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(0);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        let numerical_grads =
            compute_scale_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute scale gradients");

        assert_eq!(numerical_grads.len(), 1);

        // Gradients should be finite
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            assert!(
                numerical_grads[0][i].is_finite(),
                "Gradient component {i} is not finite"
            );
        }
    }

    /// Test scale gradients with multiple Gaussians at different scales.
    #[test]
    fn test_scale_gradients_varying_scales() {
        // Create Gaussians with different scales
        let scales = [
            [-2.0, -2.0, -2.0], // Small
            [-1.0, -1.0, -1.0], // Medium
            [0.0, 0.0, 0.0],    // exp(0) = 1.0
        ];

        let gaussians: Vec<_> = scales
            .iter()
            .enumerate()
            .map(|(i, &scale)| GaussianAttributes {
                position: [i as f32 - 1.0, 0.0, -3.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale,
                opacity: 0.0,
            })
            .collect();

        let sh_coeffs = vec![0.5; 9]; // 3 Gaussians × 3 channels

        let model = model_from_gaussians(gaussians, sh_coeffs, 0);

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));

        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(0);

        let fd_config = FiniteDiffConfig::default();
        let loss_fn = MseLoss;

        let numerical_grads =
            compute_scale_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute scale gradients");

        assert_eq!(numerical_grads.len(), 3);

        // All gradients should be finite
        for (i, grad) in numerical_grads.iter().enumerate() {
            #[allow(clippy::needless_range_loop)]
            for j in 0..3 {
                assert!(
                    grad[j].is_finite(),
                    "Gradient {i} component {j} is not finite"
                );
            }
        }
    }

    /// GRADIENT VERIFICATION TEST: Compare analytical vs numerical scale gradients.
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
    /// compatible GPU adapter is present, so it still runs (and gates CI) on
    /// any machine that does have one. A blanket `#[ignore]` let the whole
    /// suite report green without a single backward shader ever running.
    #[test]
    fn test_scale_analytical_vs_numerical() {
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
            verify_scale_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < MEDIAN_ERROR_THRESHOLD,
            "Scale gradient median error too high: {:.6e}, max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Scale too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );

        println!("Scale Gradient Verification:");
        println!("  Median Err: {:.6e}", median_err);
        println!("  Max Error:  {:.6e}", result.max_error);
        println!("  Mean Error: {:.6e}", result.mean_error);
        println!("  Outliers:   {}/{}", num_outliers, max_outlier_count);
        println!("  Status:     PASS");
    }

    /// Test scale gradients with extreme anisotropic configurations.
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when no
    /// compatible GPU adapter is present.
    #[test]
    fn test_scale_gradients_extreme_anisotropic() {
        if !gpu_available() {
            return;
        }

        let test_cases = [
            [1.0, -2.0, -2.0], // Very elongated in X
            [-2.0, 1.0, -2.0], // Very elongated in Y
            [-2.0, -2.0, 1.0], // Very elongated in Z
            [0.5, -1.5, -0.5], // Mixed scales
        ];

        for (idx, scale) in test_cases.iter().enumerate() {
            let gaussian = GaussianAttributes {
                position: [0.0, 0.0, -3.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: *scale,
                opacity: 0.0,
            };

            let model = model_from_gaussians(vec![gaussian], vec![0.5, 0.5, 0.5], 0);

            let camera = create_test_camera((64, 64));
            let target = create_target_image((64, 64));

            let config = RasterConfig::new()
                .with_resolution(64, 64)
                .with_sh_degree(0);

            let fd_config = FiniteDiffConfig::default();

            let (result, median_err, num_outliers) =
                verify_scale_gradients(&model, &camera, &target, &config, &fd_config);

            assert!(
                median_err < MEDIAN_ERROR_THRESHOLD,
                "Scale gradient verification failed for test case {}: scale={:?}, median_error={:.6e}",
                idx,
                scale,
                median_err
            );
            let max_outlier_count =
                (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
            assert!(
                num_outliers <= max_outlier_count,
                "Scale test case {} too many outliers: {} out of {} (limit={})",
                idx,
                num_outliers,
                result.num_gradients,
                max_outlier_count
            );
        }
    }
}
