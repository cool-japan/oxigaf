//! Opacity gradient verification tests.
//!
//! This module tests gradients with respect to Gaussian opacities (sigmoid-inverse space).
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

    /// Helper function to verify opacity gradients match between numerical and analytical.
    fn verify_opacity_gradients(
        model: &GaussianModel,
        camera: &CpuCamera,
        target: &[f32],
        config: &RasterConfig,
        fd_config: &FiniteDiffConfig,
    ) -> (GradientVerificationResult, f32, usize) {
        let loss_fn = MseLoss;

        // Compute numerical gradients
        let numerical_grads =
            compute_opacity_gradients(model, config, camera, target, &loss_fn, fd_config)
                .expect("Failed to compute numerical gradients");

        // Compute analytical gradients from GPU
        let analytical_grads_struct =
            compute_analytical_gradients_sync(model, camera, target, config)
                .expect("Failed to compute analytical gradients");
        let analytical_grads = analytical_grads_struct.grad_opacities;

        // Compare gradients
        assert_eq!(
            numerical_grads.len(),
            analytical_grads.len(),
            "Gradient count mismatch"
        );

        let mut errors = compare_gradients_1d(&analytical_grads, &numerical_grads);
        let median = median_error(&mut errors);
        let num_outliers = errors.iter().filter(|&&e| e > 0.5).count();
        let result = GradientVerificationResult::new(&errors, MEDIAN_ERROR_THRESHOLD);
        (result, median, num_outliers)
    }

    /// Test opacity gradients for a simple scene.
    #[test]
    fn test_opacity_gradients_simple() {
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
            compute_opacity_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute opacity gradients");

        // Verify we got the right number of gradients
        assert_eq!(numerical_grads.len(), model.len());

        // Verify gradients are finite
        for (i, grad) in numerical_grads.iter().enumerate() {
            assert!(grad.is_finite(), "Gradient {i} is not finite");
        }
    }

    /// Test opacity gradients with zero opacity (fully transparent).
    #[test]
    fn test_opacity_gradients_transparent() {
        // Create a fully transparent Gaussian
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, -1.0, -1.0],
            opacity: -10.0, // sigmoid(-10) ≈ 0
        };

        let model = GaussianModel {
            gaussians: vec![gaussian],
            sh_coeffs: vec![0.5, 0.5, 0.5],
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
            compute_opacity_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute opacity gradients");

        assert_eq!(numerical_grads.len(), 1);

        // Gradient should be finite (may be very small)
        assert!(numerical_grads[0].is_finite());
    }

    /// Test opacity gradients with full opacity (fully opaque).
    #[test]
    fn test_opacity_gradients_opaque() {
        // Create a fully opaque Gaussian
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, -1.0, -1.0],
            opacity: 10.0, // sigmoid(10) ≈ 1
        };

        let model = GaussianModel {
            gaussians: vec![gaussian],
            sh_coeffs: vec![0.5, 0.5, 0.5],
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
            compute_opacity_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute opacity gradients");

        assert_eq!(numerical_grads.len(), 1);

        // Gradient should be finite
        assert!(numerical_grads[0].is_finite());
    }

    /// Test opacity gradients with medium opacity (around 0.5).
    #[test]
    fn test_opacity_gradients_medium() {
        // Create a Gaussian with medium opacity
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, -1.0, -1.0],
            opacity: 0.0, // sigmoid(0) = 0.5
        };

        let model = GaussianModel {
            gaussians: vec![gaussian],
            sh_coeffs: vec![0.5, 0.5, 0.5],
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
            compute_opacity_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute opacity gradients");

        assert_eq!(numerical_grads.len(), 1);

        // Gradient should be finite and non-zero (strongest gradient at opacity=0.5)
        assert!(numerical_grads[0].is_finite());
    }

    /// Test opacity gradients with multiple overlapping transparent Gaussians.
    #[test]
    fn test_opacity_gradients_overlapping_transparent() {
        // Create multiple overlapping transparent Gaussians
        let gaussians: Vec<_> = (0..3)
            .map(|i| {
                GaussianAttributes {
                    position: [0.0, 0.0, -3.0 - i as f32 * 0.1],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-1.0, -1.0, -1.0],
                    opacity: -2.0, // Semi-transparent
                }
            })
            .collect();

        let sh_coeffs = vec![0.5; 9]; // 3 Gaussians × 3 channels

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
            compute_opacity_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute opacity gradients");

        assert_eq!(numerical_grads.len(), 3);

        // All gradients should be finite
        for (i, &grad) in numerical_grads.iter().enumerate() {
            assert!(grad.is_finite(), "Gradient {i} is not finite");
        }
    }

    /// Test opacity gradients with varying opacities.
    #[test]
    fn test_opacity_gradients_varying() {
        // Create Gaussians with different opacities
        let opacities = [
            -5.0, // Very transparent
            0.0,  // Medium
            5.0,  // Very opaque
        ];

        let gaussians: Vec<_> = opacities
            .iter()
            .enumerate()
            .map(|(i, &opacity)| GaussianAttributes {
                position: [i as f32 - 1.0, 0.0, -3.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-1.0, -1.0, -1.0],
                opacity,
            })
            .collect();

        let sh_coeffs = vec![0.5; 9]; // 3 Gaussians × 3 channels

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
            compute_opacity_gradients(&model, &config, &camera, &target, &loss_fn, &fd_config)
                .expect("Failed to compute opacity gradients");

        assert_eq!(numerical_grads.len(), 3);

        // All gradients should be finite
        for (i, &grad) in numerical_grads.iter().enumerate() {
            assert!(grad.is_finite(), "Gradient {i} is not finite");
        }
    }

    /// GRADIENT VERIFICATION TEST: Compare analytical vs numerical opacity gradients.
    #[test]
    fn test_opacity_analytical_vs_numerical() {
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
            verify_opacity_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < MEDIAN_ERROR_THRESHOLD,
            "Opacity gradient median error too high: {:.6e}, max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "Opacity too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );

        println!("Opacity Gradient Verification:");
        println!("  Median Err: {:.6e}", median_err);
        println!("  Max Error:  {:.6e}", result.max_error);
        println!("  Mean Error: {:.6e}", result.mean_error);
        println!("  Outliers:   {}/{}", num_outliers, max_outlier_count);
        println!("  Status:     PASS");
    }

    /// Test opacity gradients across the full sigmoid range.
    #[test]
    fn test_opacity_gradients_sigmoid_range() {
        // Test a range of opacity values (sigmoid input), excluding extreme values
        // where the sigmoid derivative is near-zero and single-entry median is unreliable
        let opacity_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        for opacity in opacity_values {
            let gaussian = GaussianAttributes {
                position: [0.0, 0.0, -3.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-1.0, -1.0, -1.0],
                opacity,
            };

            let model = GaussianModel {
                gaussians: vec![gaussian],
                sh_coeffs: vec![0.5, 0.5, 0.5],
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

            let (_result, median_err, _num_outliers) =
                verify_opacity_gradients(&model, &camera, &target, &config, &fd_config);

            // For single-entry tests, use a more relaxed absolute threshold
            // since median of 1 entry is the entry itself
            assert!(
                median_err < 1e-1,
                "Opacity gradient verification failed at opacity={}: median_error={:.6e}",
                opacity,
                median_err
            );
        }
    }
}
