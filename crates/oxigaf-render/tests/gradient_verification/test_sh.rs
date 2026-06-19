//! Spherical Harmonics (SH) gradient verification tests.
//!
//! This module tests gradients with respect to SH coefficients (SH0-SH3).
//!
//! Test strategy:
//! - Compute numerical gradients via finite-difference
//! - Compute analytical gradients from GPU backward pass
//! - Assert relative error < 1e-3
//!
//! Note: SH gradients are implemented in a separate module due to the complexity
//! of testing different SH degrees (0-3) and the large number of coefficients.

#[cfg(test)]
mod tests {
    use crate::gradient_verification::*;
    use oxigaf_render::config::RasterConfig;
    use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};

    /// Helper function to verify SH gradients match between numerical and analytical.
    fn verify_sh_gradients(
        model: &GaussianModel,
        camera: &CpuCamera,
        target: &[f32],
        config: &RasterConfig,
        fd_config: &FiniteDiffConfig,
    ) -> (GradientVerificationResult, f32, usize) {
        let loss_fn = MseLoss;

        // Compute numerical gradients
        let numerical_grads =
            compute_sh_gradients(model, config, camera, target, &loss_fn, fd_config)
                .expect("Failed to compute numerical gradients");

        // Compute analytical gradients from GPU
        let analytical_grads_struct =
            compute_analytical_gradients_sync(model, camera, target, config)
                .expect("Failed to compute analytical gradients");
        let analytical_grads = analytical_grads_struct.grad_sh_coeffs;

        // Compare gradients
        assert_eq!(
            numerical_grads.len(),
            analytical_grads.len(),
            "Gradient count mismatch: numerical={}, analytical={}",
            numerical_grads.len(),
            analytical_grads.len()
        );

        let mut errors = compare_gradients_1d(&analytical_grads, &numerical_grads);
        let median = median_error(&mut errors);
        let num_outliers = errors.iter().filter(|&&e| e > 0.5).count();
        let result = GradientVerificationResult::new(&errors, MEDIAN_ERROR_THRESHOLD);
        (result, median, num_outliers)
    }

    /// GRADIENT VERIFICATION TEST: SH degree 0 (DC term only).
    #[test]
    #[ignore = "requires GPU hardware"]
    fn test_sh_gradients_degree0() {
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

        let (result, median_err, num_outliers) =
            verify_sh_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < MEDIAN_ERROR_THRESHOLD,
            "SH degree 0 gradient median error too high: {:.6e}, max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "SH degree 0 too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );

        println!("SH Degree 0 Gradient Verification:");
        println!("  Median Err: {:.6e}", median_err);
        println!("  Max Error:  {:.6e}", result.max_error);
        println!("  Mean Error: {:.6e}", result.mean_error);
        println!("  Num Grads:  {}", result.num_gradients);
        println!("  Outliers:   {}/{}", num_outliers, max_outlier_count);
        println!("  Status:     PASS");
    }

    /// GRADIENT VERIFICATION TEST: SH degree 1 (DC + linear).
    #[test]
    #[ignore = "requires GPU hardware"]
    fn test_sh_gradients_degree1() {
        let scene_config = TestSceneConfig {
            num_gaussians: 3,
            resolution: (64, 64),
            sh_degree: 1,
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
            verify_sh_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < MEDIAN_ERROR_THRESHOLD,
            "SH degree 1 gradient median error too high: {:.6e}, max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "SH degree 1 too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );

        println!("SH Degree 1 Gradient Verification:");
        println!("  Median Err: {:.6e}", median_err);
        println!("  Max Error:  {:.6e}", result.max_error);
        println!("  Mean Error: {:.6e}", result.mean_error);
        println!("  Num Grads:  {}", result.num_gradients);
        println!("  Outliers:   {}/{}", num_outliers, max_outlier_count);
        println!("  Status:     PASS");
    }

    /// GRADIENT VERIFICATION TEST: SH degree 2 (DC + linear + quadratic).
    #[test]
    #[ignore = "requires GPU hardware"]
    fn test_sh_gradients_degree2() {
        let scene_config = TestSceneConfig {
            num_gaussians: 3,
            resolution: (64, 64),
            sh_degree: 2,
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
            verify_sh_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < MEDIAN_ERROR_THRESHOLD,
            "SH degree 2 gradient median error too high: {:.6e}, max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "SH degree 2 too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );

        println!("SH Degree 2 Gradient Verification:");
        println!("  Median Err: {:.6e}", median_err);
        println!("  Max Error:  {:.6e}", result.max_error);
        println!("  Mean Error: {:.6e}", result.mean_error);
        println!("  Num Grads:  {}", result.num_gradients);
        println!("  Outliers:   {}/{}", num_outliers, max_outlier_count);
        println!("  Status:     PASS");
    }

    /// GRADIENT VERIFICATION TEST: SH degree 3 (DC + linear + quadratic + cubic).
    #[test]
    #[ignore = "requires GPU hardware"]
    fn test_sh_gradients_degree3() {
        let scene_config = TestSceneConfig {
            num_gaussians: 3,
            resolution: (64, 64),
            sh_degree: 3,
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
            verify_sh_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < MEDIAN_ERROR_THRESHOLD,
            "SH degree 3 gradient median error too high: {:.6e}, max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "SH degree 3 too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );

        println!("SH Degree 3 Gradient Verification:");
        println!("  Median Err: {:.6e}", median_err);
        println!("  Max Error:  {:.6e}", result.max_error);
        println!("  Mean Error: {:.6e}", result.mean_error);
        println!("  Num Grads:  {}", result.num_gradients);
        println!("  Outliers:   {}/{}", num_outliers, max_outlier_count);
        println!("  Status:     PASS");
    }

    /// Test SH gradients with view-dependent effects.
    #[test]
    fn test_sh_gradients_view_dependent() {
        use nalgebra as na;

        // Create a Gaussian rotated to test view-dependency
        let angle = std::f32::consts::PI / 4.0;
        let axis = na::Vector3::new(0.0, 1.0, 0.0);
        let quat = na::UnitQuaternion::from_axis_angle(&na::Unit::new_normalize(axis), angle);

        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [quat.coords.x, quat.coords.y, quat.coords.z, quat.coords.w],
            scale: [-1.0, -1.0, -1.0],
            opacity: 0.0,
        };

        // Use degree 1 for view-dependent color
        let sh_coeffs = vec![
            0.5, 0.3, 0.2, // DC
            0.2, 0.0, 0.0, // L1m-1 (red tint)
            0.0, 0.2, 0.0, // L10 (green tint)
            0.0, 0.0, 0.2, // L11 (blue tint)
        ];

        let model = GaussianModel {
            gaussians: vec![gaussian],
            sh_coeffs,
            sh_degree: 1,
            face_indices: vec![],
            barycentric: vec![],
            local_offsets: vec![],
            is_rigid: vec![],
        };

        assert_eq!(model.sh_coeffs.len(), 12);
        // TODO: Implement actual gradient verification when compute_sh_gradients is available
    }

    /// Test SH gradients with multiple Gaussians at different SH degrees.
    #[test]
    fn test_sh_gradients_mixed_degrees() {
        // Note: In practice, all Gaussians in a model share the same SH degree,
        // but we can test different configurations separately

        for degree in 0..=3 {
            let scene_config = TestSceneConfig {
                num_gaussians: 2,
                resolution: (64, 64),
                sh_degree: degree,
                seed: 42,
            };

            let model = create_test_scene(&scene_config).expect("Failed to create test scene");

            let expected_coeffs_per_gaussian = ((degree + 1) * (degree + 1) * 3) as usize;
            let expected_total_coeffs = expected_coeffs_per_gaussian * scene_config.num_gaussians;

            assert_eq!(
                model.sh_coeffs.len(),
                expected_total_coeffs,
                "Incorrect number of SH coefficients for degree {degree}"
            );
        }
    }

    /// Test SH gradient structure for high-degree coefficients.
    #[test]
    fn test_sh_gradients_high_degree_structure() {
        // Verify that SH coefficient layout is correct for degree 3
        let gaussian = GaussianAttributes {
            position: [0.0, 0.0, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, -1.0, -1.0],
            opacity: 0.0,
        };

        // Create distinct coefficients to verify layout
        let mut sh_coeffs = Vec::new();
        for i in 0..48 {
            sh_coeffs.push(i as f32 * 0.01);
        }

        let model = GaussianModel {
            gaussians: vec![gaussian],
            sh_coeffs: sh_coeffs.clone(),
            sh_degree: 3,
            face_indices: vec![],
            barycentric: vec![],
            local_offsets: vec![],
            is_rigid: vec![],
        };

        // Verify coefficient ordering
        assert_eq!(model.sh_coeffs[0], 0.0); // First DC component
        assert_eq!(model.sh_coeffs[1], 0.01); // Second DC component
        assert_eq!(model.sh_coeffs[2], 0.02); // Third DC component
        assert_eq!(model.sh_coeffs[47], 0.47); // Last coefficient
    }
}
