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
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when
    /// no compatible GPU adapter is present, so it still runs (and gates
    /// CI) on any machine that does have one.
    #[test]
    fn test_sh_gradients_degree0() {
        if !gpu_available() {
            return;
        }

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
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when
    /// no compatible GPU adapter is present.
    #[test]
    fn test_sh_gradients_degree1() {
        if !gpu_available() {
            return;
        }

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
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when
    /// no compatible GPU adapter is present.
    #[test]
    fn test_sh_gradients_degree2() {
        if !gpu_available() {
            return;
        }

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
    ///
    /// Not `#[ignore]`d: skips itself at runtime via `gpu_available()` when
    /// no compatible GPU adapter is present.
    #[test]
    fn test_sh_gradients_degree3() {
        if !gpu_available() {
            return;
        }

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

    /// GRADIENT VERIFICATION TEST: SH gradients with view-dependent effects.
    ///
    /// Compares the GPU analytical SH-coefficient gradients against the
    /// finite-difference numerical estimate for a Gaussian with non-trivial
    /// rotation and degree-1 (view-dependent) SH coefficients, using the
    /// same `compute_sh_gradients` + `compute_analytical_gradients_sync`
    /// comparison as `verify_sh_gradients`. Not `#[ignore]`d: skips itself
    /// at runtime via `gpu_available()` when no compatible GPU adapter is
    /// present.
    ///
    /// The Gaussian is deliberately placed off the camera's optical axis
    /// (not at `x = y = 0`): the camera looks down `-Z` from the origin, so
    /// an on-axis Gaussian has a camera->Gaussian direction of exactly
    /// `(0, 0, -1)`, at which the `Y_1^{-1}` (`~y`) and `Y_1^{1}` (`~x`)
    /// degree-1 basis functions both evaluate to exactly zero. That would
    /// make 6 of the 12 coefficients' analytical *and* numerical gradients
    /// trivially zero (`compute_relative_error` short-circuits an exact
    /// zero/zero match), so a coefficient-ordering bug swapping those two
    /// basis slots would go undetected - the same class of "verifies
    /// nothing" gap as finding 4 itself, just for two-thirds of the
    /// view-dependent term instead of all of it. Off-axis, every basis
    /// direction is non-zero and actually exercised.
    #[test]
    fn test_sh_gradients_view_dependent() {
        use nalgebra as na;

        if !gpu_available() {
            return;
        }

        // Create a Gaussian rotated to test view-dependency
        let angle = std::f32::consts::PI / 4.0;
        let axis = na::Vector3::new(0.0, 1.0, 0.0);
        let quat = na::UnitQuaternion::from_axis_angle(&na::Unit::new_normalize(axis), angle);

        let gaussian = GaussianAttributes {
            // Off-axis in both x and y - see the function doc for why
            // `[0.0, 0.0, -3.0]` (dead-center on the optical axis) would
            // silently fail to exercise 6 of the 12 coefficients.
            position: [0.3, -0.25, -3.0],
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
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0; 3]],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![false],
        };

        assert_eq!(model.sh_coeffs.len(), 12);

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));
        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(1);
        let fd_config = FiniteDiffConfig::default();

        let (result, median_err, num_outliers) =
            verify_sh_gradients(&model, &camera, &target, &config, &fd_config);

        assert!(
            median_err < MEDIAN_ERROR_THRESHOLD,
            "View-dependent SH gradient median error too high: {:.6e}, max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "View-dependent SH too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );
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

    /// GRADIENT VERIFICATION TEST: SH gradient structure for high-degree
    /// coefficients.
    ///
    /// Verifies that all 48 degree-3 SH coefficients (16 basis functions x
    /// 3 RGB channels) actually participate in rendering with the
    /// coefficient ordering the backward shader expects, by comparing the
    /// GPU analytical gradient for every coefficient against an independent
    /// finite-difference estimate - not merely reading back the raw `Vec`
    /// this test itself just wrote. Not `#[ignore]`d: skips itself at
    /// runtime via `gpu_available()` when no compatible GPU adapter is
    /// present.
    ///
    /// The Gaussian is placed off the camera's optical axis, at an x/y
    /// offset of different, non-simply-related magnitude - the same
    /// degeneracy documented on `test_sh_gradients_view_dependent`, but
    /// total here instead of partial. Per `environment::sh_basis_up_to_l3`
    /// (whose doc comment states it matches the shader's own basis
    /// evaluation exactly), every `m != 0` real SH basis function through
    /// degree 3 is a monomial in `x` and `y` (`y`, `x`, `xy`, `xz`,
    /// `x^2-y^2`, `y(3x^2-y^2)`, `xyz`, `y(4z^2-x^2-y^2)`,
    /// `x(4z^2-x^2-y^2)`, `z(x^2-y^2)`, `x(x^2-3y^2)`) that is identically
    /// zero whenever `x = y = 0`. The dead-center camera->Gaussian direction
    /// `(0, 0, -1)` used before this fix put exactly that: 12 of the 16
    /// basis functions (2 of 3 at degree 1, 4 of 5 at degree 2, 6 of 7 at
    /// degree 3 - everything except each degree's `m = 0` term) collapsed
    /// to zero at once, so 36 of the 48 coefficients had an analytical
    /// *and* numerical gradient of exactly `0.0`. `compute_relative_error
    /// (0.0, 0.0)` short-circuits to a "perfect" `0.0` regardless of what
    /// the shader does with those indices, so a coefficient-ordering bug
    /// confined to that 36-entry dead set (e.g. swapping two of them) would
    /// have been invisible, even though the test compares real gradients
    /// throughout. Moving off-axis makes every one of those monomials
    /// nonzero - verified numerically for the position below: the smallest
    /// live basis value is still comfortably above zero, not a near-miss -
    /// so all 48 coefficients are now actually exercised.
    #[test]
    fn test_sh_gradients_high_degree_structure() {
        if !gpu_available() {
            return;
        }

        let gaussian = GaussianAttributes {
            // Off-axis with `x` and `y` offsets of different magnitude and
            // no simple ratio between them - see the function doc. Neither
            // component is small relative to the other or to `z`, so every
            // basis monomial listed above stays comfortably clear of zero
            // rather than merely nonzero-by-a-hair.
            position: [0.9, -0.2, -3.0],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, -1.0, -1.0],
            opacity: 0.0,
        };

        // Distinct, non-degenerate coefficient *values* so a
        // coefficient-ordering mismatch shows up as a per-index gradient
        // mismatch instead of numerically cancelling out. Necessary but not
        // sufficient on its own - it's the off-axis `position` above that
        // keeps every coefficient's *basis function* away from zero; see
        // the function doc.
        let sh_coeffs: Vec<f32> = (0..48).map(|i| i as f32 * 0.01).collect();

        // Sanity-check the raw layout before it goes through the pipeline.
        assert_eq!(sh_coeffs[0], 0.0); // First DC component
        assert_eq!(sh_coeffs[1], 0.01); // Second DC component
        assert_eq!(sh_coeffs[2], 0.02); // Third DC component
        assert_eq!(sh_coeffs[47], 0.47); // Last coefficient

        let model = GaussianModel {
            gaussians: vec![gaussian],
            sh_coeffs,
            sh_degree: 3,
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0; 3]],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![false],
        };

        let camera = create_test_camera((64, 64));
        let target = create_target_image((64, 64));
        let config = RasterConfig::new()
            .with_resolution(64, 64)
            .with_sh_degree(3);
        let fd_config = FiniteDiffConfig::default();

        let (result, median_err, num_outliers) =
            verify_sh_gradients(&model, &camera, &target, &config, &fd_config);

        assert_eq!(
            result.num_gradients, 48,
            "Expected one gradient per degree-3 SH coefficient (16 basis fns x 3 channels)"
        );
        assert!(
            median_err < MEDIAN_ERROR_THRESHOLD,
            "High-degree SH gradient median error too high: {:.6e}, max_error={:.6e}, mean_error={:.6e}",
            median_err,
            result.max_error,
            result.mean_error
        );
        let max_outlier_count =
            (result.num_gradients as f32 * MAX_OUTLIER_FRACTION).ceil() as usize;
        assert!(
            num_outliers <= max_outlier_count,
            "High-degree SH too many gradient outliers: {} out of {} (limit={}, median={:.6e})",
            num_outliers,
            result.num_gradients,
            max_outlier_count,
            median_err
        );
    }

    /// Regression test for the geometry `test_sh_gradients_high_degree_structure`
    /// relies on to be a meaningful check at all - and, unlike that test, this
    /// one needs no GPU, so it still runs (and can still fail) on a machine
    /// where `gpu_available()` makes the comparison test above silently skip.
    ///
    /// Pins two things: (1) the on-axis direction this suite used before this
    /// fix made exactly the 12 `m != 0` SH basis functions (of 16, through
    /// degree 3) vanish at once - the reason the old comparison passed
    /// regardless of shader correctness on 36 of 48 coefficients, see that
    /// test's doc comment - and (2) that the off-axis direction it now uses
    /// leaves every one of the 16 basis functions clearly nonzero. If a
    /// future edit moves that test's Gaussian back on-axis, or onto any
    /// other direction where some degree-0..3 basis monomial happens to
    /// vanish (each has its own zero set - e.g. `x = 0`, `y = 0`, `|x| =
    /// |y|`, `|y| = sqrt(3)|x|`, `|x| = sqrt(3)|y|` all zero at least one),
    /// this test - not just the GPU-only one - has a chance of catching it.
    #[test]
    fn test_sh_basis_degenerate_on_axis_vs_live_off_axis() {
        use oxigaf_render::environment::sh_basis_up_to_l3;

        // On-axis: camera at the origin looking down -Z, Gaussian directly
        // ahead (direction `(0, 0, -1)`). Every `m != 0` basis function
        // through degree 3 is a monomial in x and y (see
        // `test_sh_gradients_high_degree_structure`'s doc comment for the
        // full list), so all of them vanish here - this is the historical
        // bug's root cause, reproduced directly rather than through a render.
        let on_axis = sh_basis_up_to_l3([0.0, 0.0, -1.0]);
        let on_axis_zero_count = on_axis.iter().filter(|v| **v == 0.0).count();
        assert_eq!(
            on_axis_zero_count, 12,
            "expected exactly the 12 `m != 0` basis functions to vanish on-axis, got \
             {on_axis_zero_count} zero of {on_axis:?}"
        );

        // Off-axis: the exact position `test_sh_gradients_high_degree_structure`
        // now uses (`[0.9, -0.2, -3.0]`, normalized). None of the 16 basis
        // functions should be exactly zero, or even close to it - either
        // would silently defeat that test's per-coefficient gradient
        // comparison the same way the on-axis position did.
        let (px, py, pz) = (0.9_f32, -0.2_f32, -3.0_f32);
        let len = (px * px + py * py + pz * pz).sqrt();
        let off_axis = sh_basis_up_to_l3([px / len, py / len, pz / len]);
        for (i, v) in off_axis.iter().enumerate() {
            assert!(
                v.abs() > 1e-4,
                "basis function {i} is degenerate off-axis ({v}), which would silently defeat \
                 test_sh_gradients_high_degree_structure's per-coefficient comparison: {off_axis:?}"
            );
        }
    }
}
