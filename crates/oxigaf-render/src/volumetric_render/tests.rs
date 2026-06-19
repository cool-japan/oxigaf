//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn unit_grid() -> VolumeGrid {
        VolumeGrid::new(4, 4, 4, [0.0; 3], [0.25; 3])
    }
    fn constant_grid(value: f32) -> VolumeGrid {
        let mut g = unit_grid();
        for v in g.data.iter_mut() {
            *v = value;
        }
        g
    }
    fn sphere_grid() -> VolumeGrid {
        VolumeGrid::from_fn(16, 16, 16, [-1.0; 3], [2.0 / 16.0; 3], |x, y, z| {
            let r2 = x * x + y * y + z * z;
            if r2 < 0.25 {
                1.0
            } else {
                0.0
            }
        })
    }
    fn default_tf() -> TransferFunction {
        TransferFunction::grayscale(1.0)
    }
    fn default_cam(w: u32, h: u32) -> VolumetricCamera {
        VolumetricCamera::default_front(w, h)
    }
    fn default_cfg() -> VolumetricRenderConfig {
        VolumetricRenderConfig {
            step_size: 0.05,
            max_steps: 200,
            early_termination_alpha: 0.99,
            integration: VolumetricIntegration::FrontToBack,
            jitter: false,
            jitter_seed: 42,
        }
    }
    #[test]
    fn test_volume_grid_new_dimensions() {
        let g = VolumeGrid::new(4, 5, 6, [0.0; 3], [1.0; 3]);
        assert_eq!(g.nx, 4);
        assert_eq!(g.ny, 5);
        assert_eq!(g.nz, 6);
        assert_eq!(g.data.len(), 4 * 5 * 6);
        assert!(g.data.iter().all(|&v| v == 0.0));
    }
    #[test]
    fn test_volume_grid_from_fn_constant() {
        let g = VolumeGrid::from_fn(3, 3, 3, [0.0; 3], [1.0; 3], |_, _, _| 7.0);
        assert!(g.data.iter().all(|&v| (v - 7.0).abs() < 1e-6));
    }
    #[test]
    fn test_volume_grid_from_fn_positional() {
        let vs = 0.5;
        let g = VolumeGrid::from_fn(2, 1, 1, [0.0; 3], [vs; 3], |x, _, _| x);
        assert!((g.data[0] - 0.25).abs() < 1e-5);
        assert!((g.data[1] - 0.75).abs() < 1e-5);
    }
    #[test]
    fn test_world_voxel_round_trip() {
        let g = VolumeGrid::new(8, 8, 8, [-1.0; 3], [0.25; 3]);
        let p = [0.3, -0.1, 0.7_f32];
        let vi = g.world_to_voxel(p);
        let p2 = g.voxel_to_world(vi);
        for i in 0..3 {
            assert!((p[i] - p2[i]).abs() < 1e-5, "axis {i}: {p2:?}");
        }
    }
    #[test]
    fn test_world_to_voxel_origin() {
        let g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        let vi = g.world_to_voxel([0.0; 3]);
        for (i, &v) in vi.iter().enumerate() {
            assert!((v - (-0.5)).abs() < 1e-5, "axis {i}");
        }
    }
    #[test]
    fn test_in_bounds_inside() {
        let g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        assert!(g.in_bounds([0.5, 0.5, 0.5]));
        assert!(g.in_bounds([2.0, 2.0, 2.0]));
    }
    #[test]
    fn test_in_bounds_corners() {
        let g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        assert!(g.in_bounds([0.0; 3]));
        assert!(g.in_bounds([4.0; 3]));
    }
    #[test]
    fn test_in_bounds_outside() {
        let g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        assert!(!g.in_bounds([-0.01, 0.0, 0.0]));
        assert!(!g.in_bounds([4.01, 0.0, 0.0]));
    }
    #[test]
    fn test_density_at_indexing() {
        let mut g = VolumeGrid::new(3, 3, 3, [0.0; 3], [1.0; 3]);
        g.data[3 * 3 + 2 * 3] = 42.0;
        assert_eq!(g.density_at(0, 2, 1), 42.0);
    }
    #[test]
    fn test_density_at_clamping() {
        let mut g = VolumeGrid::new(2, 2, 2, [0.0; 3], [1.0; 3]);
        g.data[7] = 99.0;
        assert_eq!(g.density_at(10, 10, 10), 99.0);
    }
    #[test]
    fn test_sample_trilinear_at_voxel_centres() {
        let mut g = VolumeGrid::new(2, 2, 2, [0.0; 3], [1.0; 3]);
        g.data[0] = 3.0;
        let v = g.sample_trilinear(0.5, 0.5, 0.5);
        assert!((v - 3.0).abs() < 1e-5, "got {v}");
    }
    #[test]
    fn test_sample_trilinear_midpoint() {
        let mut g = VolumeGrid::new(2, 1, 1, [0.0; 3], [1.0; 3]);
        g.data[0] = 0.0;
        g.data[1] = 4.0;
        let v = g.sample_trilinear(1.0, 0.5, 0.5);
        assert!((v - 2.0).abs() < 0.1, "got {v}");
    }
    #[test]
    fn test_sample_trilinear_constant_grid() {
        let g = constant_grid(5.0);
        let v = g.sample_trilinear(0.4, 0.3, 0.5);
        assert!((v - 5.0).abs() < 1e-5);
    }
    #[test]
    fn test_sample_trilinear_boundary_clamp() {
        let mut g = VolumeGrid::new(4, 4, 4, [0.0; 3], [0.25; 3]);
        for v in g.data.iter_mut() {
            *v = 1.0;
        }
        let v = g.sample_trilinear(-1.0, -1.0, -1.0);
        assert!((v - 1.0).abs() < 1e-4, "got {v}");
    }
    #[test]
    fn test_sample_nearest_centre() {
        let mut g = VolumeGrid::new(3, 3, 3, [0.0; 3], [1.0; 3]);
        g.data[3 * 3 + 3 + 1] = 7.0;
        let v = g.sample_nearest(1.5, 1.5, 1.5);
        assert!((v - 7.0).abs() < 1e-5, "got {v}");
    }
    #[test]
    fn test_gradient_constant_volume() {
        let g = constant_grid(3.0);
        let grad = g.gradient(0.5, 0.5, 0.5);
        for (i, &g_val) in grad.iter().enumerate() {
            assert!(g_val.abs() < 1e-4, "grad[{i}] = {}", g_val);
        }
    }
    #[test]
    fn test_gradient_linear_x() {
        let g = VolumeGrid::from_fn(16, 4, 4, [0.0; 3], [0.25; 3], |x, _, _| x);
        let grad = g.gradient(1.0, 0.5, 0.5);
        assert!((grad[0] - 1.0).abs() < 0.1, "dx={}", grad[0]);
        assert!(grad[1].abs() < 0.1, "dy={}", grad[1]);
    }
    #[test]
    fn test_ray_at() {
        let ray = VolumetricRay::new_normalized([0.0; 3], [1.0, 0.0, 0.0]);
        let p = ray.at(3.0);
        assert!((p[0] - 3.0).abs() < 1e-5);
        assert!(p[1].abs() < 1e-5);
        assert!(p[2].abs() < 1e-5);
    }
    #[test]
    fn test_ray_direction_normalized() {
        let ray = VolumetricRay::new_normalized([0.0; 3], [3.0, 4.0, 0.0]);
        let len = ray.direction[0].powi(2) + ray.direction[1].powi(2) + ray.direction[2].powi(2);
        assert!((len.sqrt() - 1.0).abs() < 1e-5, "len={len}");
    }
    #[test]
    fn test_aabb_ray_through_centre() {
        let g = VolumeGrid::new(10, 10, 10, [0.0; 3], [0.1; 3]);
        let ray = VolumetricRay::new_normalized([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let hit = vr_ray_aabb_intersect(&ray, &g);
        assert!(hit.is_some(), "should hit");
        let (t0, t1) = hit.unwrap_or((0.0, 0.0));
        assert!(t1 > t0, "t_far={t1} <= t_near={t0}");
    }
    #[test]
    fn test_aabb_ray_miss() {
        let g = VolumeGrid::new(10, 10, 10, [0.0; 3], [0.1; 3]);
        let ray = VolumetricRay::new_normalized([-2.0, 0.5, 0.5], [-1.0, 0.0, 0.0]);
        assert!(vr_ray_aabb_intersect(&ray, &g).is_none());
    }
    #[test]
    fn test_aabb_ray_starting_inside() {
        let g = VolumeGrid::new(10, 10, 10, [0.0; 3], [0.1; 3]);
        let ray = VolumetricRay::new_normalized([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let hit = vr_ray_aabb_intersect(&ray, &g);
        assert!(hit.is_some());
        let (t0, _t1) = hit.unwrap_or((1.0, 0.0));
        assert!(t0 < 1e-5, "t_near should be 0 for ray inside box, got {t0}");
    }
    #[test]
    fn test_aabb_ray_parallel_inside() {
        let g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        let ray = VolumetricRay::new_normalized([2.0, 2.0, -1.0], [0.0, 0.0, 1.0]);
        assert!(vr_ray_aabb_intersect(&ray, &g).is_some());
    }
    #[test]
    fn test_aabb_ray_parallel_outside() {
        let g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        let ray = VolumetricRay::new_normalized([5.0, 5.0, -1.0], [0.0, 0.0, 1.0]);
        assert!(vr_ray_aabb_intersect(&ray, &g).is_none());
    }
    #[test]
    fn test_transfer_function_evaluate_endpoints() {
        let tf = TransferFunction::grayscale(1.0);
        let (c0, a0) = tf.evaluate(0.0);
        assert!(c0.iter().all(|&v| v == 0.0));
        assert_eq!(a0, 0.0);
        let (c1, a1) = tf.evaluate(1.0);
        assert!(c1.iter().all(|&v| (v - 1.0).abs() < 1e-5));
        assert!((a1 - 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_transfer_function_evaluate_midpoint() {
        let tf = TransferFunction::grayscale(1.0);
        let (c, a) = tf.evaluate(0.5);
        assert!((c[0] - 0.5).abs() < 1e-5, "color mid={}", c[0]);
        assert!((a - 0.5).abs() < 1e-5, "alpha mid={a}");
    }
    #[test]
    fn test_transfer_function_evaluate_below_range() {
        let tf = TransferFunction::grayscale(1.0);
        let (c, a) = tf.evaluate(-5.0);
        assert!(c.iter().all(|&v| v == 0.0));
        assert_eq!(a, 0.0);
    }
    #[test]
    fn test_transfer_function_evaluate_above_range() {
        let tf = TransferFunction::grayscale(1.0);
        let (c, a) = tf.evaluate(100.0);
        assert!(c.iter().all(|&v| (v - 1.0).abs() < 1e-5));
        assert!((a - 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_transfer_function_unsorted_error() {
        let pts = vec![
            TransferPoint {
                density: 1.0,
                color: [1.0; 3],
                opacity: 1.0,
            },
            TransferPoint {
                density: 0.0,
                color: [0.0; 3],
                opacity: 0.0,
            },
        ];
        assert!(TransferFunction::new(pts).is_err());
    }
    #[test]
    fn test_transfer_function_empty_error() {
        assert!(TransferFunction::new(vec![]).is_err());
    }
    #[test]
    fn test_grayscale_endpoints() {
        let tf = TransferFunction::grayscale(2.0);
        let (c0, a0) = tf.evaluate(0.0);
        assert_eq!(a0, 0.0);
        assert!(c0.iter().all(|&v| v == 0.0));
        let (c1, a1) = tf.evaluate(2.0);
        assert!((a1 - 1.0).abs() < 1e-5);
        assert!(c1.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }
    #[test]
    fn test_heat_endpoints() {
        let tf = TransferFunction::heat(1.0);
        let (c0, a0) = tf.evaluate(0.0);
        assert!(a0.abs() < 1e-5);
        assert!((c0[2] - 1.0).abs() < 1e-5, "expected blue, got {c0:?}");
        let (c1, a1) = tf.evaluate(1.0);
        assert!((a1 - 1.0).abs() < 1e-5);
        assert!((c1[0] - 1.0).abs() < 1e-5, "expected red, got {c1:?}");
    }
    #[test]
    fn test_heat_midpoint() {
        let tf = TransferFunction::heat(1.0);
        let (_c, a) = tf.evaluate(0.5);
        assert!((a - 0.5).abs() < 0.01, "heat mid alpha={a}");
    }
    #[test]
    fn test_camera_generate_ray_points_toward_target() {
        let cam = VolumetricCamera::default_front(64, 64);
        let ray = cam.generate_ray(32.0, 32.0);
        assert!(ray.direction[2] < 0.0, "dir.z={}", ray.direction[2]);
    }
    #[test]
    fn test_camera_ray_is_unit_length() {
        let cam = VolumetricCamera::default_front(64, 64);
        for (px, py) in [(0.0, 0.0), (63.0, 63.0), (32.0, 0.0)] {
            let ray = cam.generate_ray(px, py);
            let len =
                (ray.direction[0].powi(2) + ray.direction[1].powi(2) + ray.direction[2].powi(2))
                    .sqrt();
            assert!((len - 1.0).abs() < 1e-4, "px={px} py={py} len={len}");
        }
    }
    #[test]
    fn test_camera_different_pixels_different_directions() {
        let cam = VolumetricCamera::default_front(64, 64);
        let r1 = cam.generate_ray(0.0, 0.0);
        let r2 = cam.generate_ray(63.0, 63.0);
        let same = r1
            .direction
            .iter()
            .zip(r2.direction.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);
        assert!(!same, "corner rays should differ");
    }
    #[test]
    fn test_march_ray_empty_volume_zero_alpha() {
        let g = unit_grid();
        let tf = default_tf();
        let cfg = default_cfg();
        let ray = VolumetricRay::new_normalized([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!(res.alpha.abs() < 1e-5, "alpha={}", res.alpha);
        assert!(res.color.iter().all(|&v| v.abs() < 1e-5));
    }
    #[test]
    fn test_march_ray_constant_density_alpha_increases() {
        let g = constant_grid(1.0);
        let tf = default_tf();
        let mut cfg = default_cfg();
        cfg.step_size = 0.1;
        cfg.max_steps = 500;
        let ray = VolumetricRay::new_normalized([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!(res.alpha > 0.0, "alpha should be > 0 for dense volume");
    }
    #[test]
    fn test_march_ray_front_to_back_alpha_monotone() {
        let g = VolumeGrid::from_fn(16, 4, 4, [0.0; 3], [0.0625; 3], |x, _, _| x);
        let tf = TransferFunction::grayscale(1.0);
        let mut cfg = default_cfg();
        cfg.step_size = 0.02;
        cfg.max_steps = 100;
        let ray = VolumetricRay::new_normalized([-0.1, 0.125, 0.125], [1.0, 0.0, 0.0]);
        let res1 = vr_march_ray(&ray, &g, &tf, &cfg);
        let mut cfg2 = cfg.clone();
        cfg2.max_steps = 10;
        let res2 = vr_march_ray(&ray, &g, &tf, &cfg2);
        assert!(
            res1.alpha >= res2.alpha - 1e-4,
            "monotone failure: {} < {}",
            res1.alpha,
            res2.alpha
        );
    }
    #[test]
    fn test_march_ray_mip_max_density() {
        let mut g = VolumeGrid::new(8, 8, 8, [0.0; 3], [0.125; 3]);
        g.data[4 * 8 * 8 + 4 * 8 + 4] = 0.9;
        let tf = TransferFunction::grayscale(1.0);
        let mut cfg = default_cfg();
        cfg.integration = VolumetricIntegration::Mip;
        cfg.step_size = 0.05;
        let ray = VolumetricRay::new_normalized([0.5625, 0.5625, -0.5], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!(res.alpha > 0.0, "MIP should capture the bright voxel");
    }
    #[test]
    fn test_march_ray_early_termination() {
        let g = constant_grid(1.0);
        let tf = TransferFunction::grayscale(1.0);
        let mut cfg = default_cfg();
        cfg.step_size = 0.1;
        cfg.max_steps = 10000;
        cfg.early_termination_alpha = 0.5;
        let ray = VolumetricRay::new_normalized([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!(
            res.n_steps < 10000,
            "early term failed, steps={}",
            res.n_steps
        );
        assert!(res.alpha >= 0.5 - 0.1);
    }
    #[test]
    fn test_march_ray_back_to_front() {
        let g = sphere_grid();
        let tf = TransferFunction::grayscale(1.0);
        let mut cfg = default_cfg();
        cfg.integration = VolumetricIntegration::BackToFront;
        cfg.step_size = 0.1;
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!(res.alpha >= 0.0);
    }
    #[test]
    fn test_march_ray_avg() {
        let g = constant_grid(0.5);
        let tf = TransferFunction::grayscale(1.0);
        let mut cfg = default_cfg();
        cfg.integration = VolumetricIntegration::Avg;
        cfg.step_size = 0.1;
        let ray = VolumetricRay::new_normalized([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!((res.alpha - 0.5).abs() < 0.05, "avg alpha={}", res.alpha);
    }
    #[test]
    fn test_march_ray_max_steps_limit() {
        let g = constant_grid(0.0);
        let tf = default_tf();
        let mut cfg = default_cfg();
        cfg.max_steps = 1;
        cfg.step_size = 10.0;
        let ray = VolumetricRay::new_normalized([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!(res.n_steps <= 1);
    }
    #[test]
    fn test_march_ray_miss_no_intersection() {
        let g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        let tf = default_tf();
        let cfg = default_cfg();
        let ray = VolumetricRay::new_normalized([10.0, 10.0, 10.0], [1.0, 1.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert_eq!(res.n_steps, 0);
        assert_eq!(res.alpha, 0.0);
    }
    #[test]
    fn test_render_image_buffer_size() {
        let g = sphere_grid();
        let tf = default_tf();
        let cam = default_cam(8, 6);
        let cfg = default_cfg();
        let img = vr_render_image(&g, &tf, &cam, &cfg);
        assert!(img.is_ok());
        assert_eq!(img.unwrap_or_default().len(), 8 * 6);
    }
    #[test]
    fn test_render_image_u8_buffer_size() {
        let g = sphere_grid();
        let tf = default_tf();
        let cam = default_cam(8, 6);
        let cfg = default_cfg();
        let img = vr_render_image_u8(&g, &tf, &cam, &cfg);
        assert!(img.is_ok());
        assert_eq!(img.unwrap_or_default().len(), 8 * 6 * 4);
    }
    #[test]
    fn test_render_image_zero_size_error() {
        let g = VolumeGrid::new(0, 4, 4, [0.0; 3], [1.0; 3]);
        let tf = default_tf();
        let cam = default_cam(4, 4);
        let cfg = default_cfg();
        assert!(vr_render_image(&g, &tf, &cam, &cfg).is_err());
    }
    #[test]
    fn test_render_image_rgba_range() {
        let g = sphere_grid();
        let tf = TransferFunction::heat(1.0);
        let cam = default_cam(4, 4);
        let cfg = default_cfg();
        let img = vr_render_image(&g, &tf, &cam, &cfg).unwrap_or_default();
        for px in &img {
            for &ch in px.iter() {
                assert!((0.0..=1.0 + 1e-4).contains(&ch), "out of range: {ch}");
            }
        }
    }
    #[test]
    fn test_render_image_u8_range() {
        let g = sphere_grid();
        let tf = default_tf();
        let cam = default_cam(4, 4);
        let cfg = default_cfg();
        let img = vr_render_image_u8(&g, &tf, &cam, &cfg).unwrap_or_default();
        assert_eq!(img.len(), 4 * 4 * 4);
    }
    #[test]
    fn test_occupancy_threshold_zero_all_occupied() {
        let mut g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        g.data[0] = 0.001;
        let occ = vr_build_occupancy_grid(&g, 0.0, 2);
        assert!(occ.data.iter().any(|&v| v > 0.5));
    }
    #[test]
    fn test_occupancy_threshold_large_all_empty() {
        let g = constant_grid(0.5);
        let occ = vr_build_occupancy_grid(&g, 1e6, 2);
        assert!(occ.data.iter().all(|&v| v < 0.5));
    }
    #[test]
    fn test_occupancy_grid_downsampling() {
        let g = VolumeGrid::new(8, 8, 8, [0.0; 3], [0.5; 3]);
        let occ = vr_build_occupancy_grid(&g, 0.0, 4);
        assert_eq!(occ.nx, 2);
        assert_eq!(occ.ny, 2);
        assert_eq!(occ.nz, 2);
    }
    #[test]
    fn test_occupancy_selectively_occupied() {
        let mut g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        g.data[0] = 1.0;
        let occ = vr_build_occupancy_grid(&g, 0.5, 2);
        assert!(occ.data[0] > 0.5, "first occ voxel should be occupied");
        let last = occ.data.len() - 1;
        assert!(occ.data[last] < 0.5, "last occ voxel should be empty");
    }
    #[test]
    fn test_can_skip_empty_region() {
        let g = VolumeGrid::new(4, 4, 4, [0.0; 3], [1.0; 3]);
        let ray = VolumetricRay::new_normalized([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]);
        assert!(vr_can_skip(&g, &ray, 0.5, 0.1));
    }
    #[test]
    fn test_can_skip_occupied_region() {
        let g = constant_grid(1.0);
        let ray = VolumetricRay::new_normalized([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]);
        assert!(!vr_can_skip(&g, &ray, 0.5, 0.1));
    }
    #[test]
    fn test_gaussians_to_volume_increases_density() {
        let mut g = VolumeGrid::new(8, 8, 8, [-1.0; 3], [0.25; 3]);
        let positions = [0.0_f32, 0.0, 0.0];
        let scales = [0.2_f32, 0.2, 0.2];
        let opacities = [1.0_f32];
        vr_gaussians_to_volume(&positions, &scales, &opacities, 1, &mut g).unwrap_or(());
        let centre_density = g.sample_trilinear(0.0, 0.0, 0.0);
        assert!(centre_density > 0.0, "density at origin={centre_density}");
    }
    #[test]
    fn test_gaussians_to_volume_density_falloff() {
        let mut g = VolumeGrid::new(16, 16, 16, [-2.0; 3], [0.25; 3]);
        let positions = [0.0_f32, 0.0, 0.0];
        let scales = [0.3_f32, 0.3, 0.3];
        let opacities = [1.0_f32];
        vr_gaussians_to_volume(&positions, &scales, &opacities, 1, &mut g).unwrap_or(());
        let near = g.sample_trilinear(0.0, 0.0, 0.0);
        let far = g.sample_trilinear(1.5, 1.5, 1.5);
        assert!(near > far, "near={near} far={far}");
    }
    #[test]
    fn test_gaussians_to_volume_wrong_positions_length() {
        let mut g = unit_grid();
        let result = vr_gaussians_to_volume(&[0.0, 0.0], &[1.0, 1.0, 1.0], &[1.0], 1, &mut g);
        assert!(result.is_err());
    }
    #[test]
    fn test_gaussians_to_volume_wrong_scales_length() {
        let mut g = unit_grid();
        let result = vr_gaussians_to_volume(&[0.0, 0.0, 0.0], &[1.0, 1.0], &[1.0], 1, &mut g);
        assert!(result.is_err());
    }
    #[test]
    fn test_gaussians_to_volume_wrong_opacities_length() {
        let mut g = unit_grid();
        let result = vr_gaussians_to_volume(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], &[], 1, &mut g);
        assert!(result.is_err());
    }
    #[test]
    fn test_gaussians_to_volume_zero_gaussians() {
        let mut g = unit_grid();
        let before = g.data.clone();
        vr_gaussians_to_volume(&[], &[], &[], 0, &mut g).unwrap_or(());
        assert_eq!(g.data, before);
    }
    #[test]
    fn test_compute_stats_empty_slice() {
        let stats = vr_compute_stats(&[]);
        assert_eq!(stats.n_rays, 0);
        assert_eq!(stats.mean_steps_per_ray, 0.0);
    }
    #[test]
    fn test_compute_stats_mean_steps() {
        let results = vec![
            RayMarchResult {
                n_steps: 10,
                alpha: 0.5,
                ..Default::default()
            },
            RayMarchResult {
                n_steps: 20,
                alpha: 0.9,
                ..Default::default()
            },
        ];
        let stats = vr_compute_stats(&results);
        assert!((stats.mean_steps_per_ray - 15.0).abs() < 1e-3);
        assert_eq!(stats.n_rays, 2);
    }
    #[test]
    fn test_compute_stats_fully_opaque() {
        let results = vec![
            RayMarchResult {
                n_steps: 5,
                alpha: 1.0,
                ..Default::default()
            },
            RayMarchResult {
                n_steps: 5,
                alpha: 0.1,
                ..Default::default()
            },
        ];
        let stats = vr_compute_stats(&results);
        assert_eq!(stats.fully_opaque_rays, 1);
    }
    #[test]
    fn test_compute_stats_empty_rays() {
        let results = vec![
            RayMarchResult {
                n_steps: 0,
                alpha: 0.0,
                ..Default::default()
            },
            RayMarchResult {
                n_steps: 5,
                alpha: 0.5,
                ..Default::default()
            },
        ];
        let stats = vr_compute_stats(&results);
        assert_eq!(stats.empty_rays, 1);
    }
    #[test]
    fn test_compute_stats_max_steps() {
        let results = vec![
            RayMarchResult {
                n_steps: 100,
                alpha: 0.5,
                ..Default::default()
            },
            RayMarchResult {
                n_steps: 50,
                alpha: 0.5,
                ..Default::default()
            },
        ];
        let stats = vr_compute_stats(&results);
        assert_eq!(stats.max_steps_per_ray, 100);
    }
    #[test]
    fn test_compute_stats_mean_alpha() {
        let results = vec![
            RayMarchResult {
                n_steps: 1,
                alpha: 0.4,
                ..Default::default()
            },
            RayMarchResult {
                n_steps: 1,
                alpha: 0.8,
                ..Default::default()
            },
        ];
        let stats = vr_compute_stats(&results);
        assert!((stats.mean_alpha - 0.6).abs() < 1e-5);
    }
    #[test]
    fn test_format_stats_nonempty() {
        let results = vec![RayMarchResult {
            n_steps: 5,
            alpha: 0.5,
            ..Default::default()
        }];
        let stats = vr_compute_stats(&results);
        let s = vr_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("VolumetricStats"));
    }
    #[test]
    fn test_format_config_nonempty() {
        let cfg = default_cfg();
        let s = vr_format_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("VolumetricRenderConfig"));
    }
    #[test]
    fn test_large_step_size_few_steps() {
        let g = sphere_grid();
        let tf = default_tf();
        let mut cfg = default_cfg();
        cfg.step_size = 100.0;
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!(res.n_steps <= 1);
    }
    #[test]
    fn test_step_size_greater_than_volume_few_steps() {
        let g = VolumeGrid::new(4, 4, 4, [0.0; 3], [0.1; 3]);
        let tf = default_tf();
        let mut cfg = default_cfg();
        cfg.step_size = 10.0;
        let ray = VolumetricRay::new_normalized([0.2, 0.2, -0.5], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!(res.n_steps <= 1);
    }
    #[test]
    fn test_max_steps_one() {
        let g = constant_grid(1.0);
        let tf = default_tf();
        let mut cfg = default_cfg();
        cfg.max_steps = 1;
        cfg.step_size = 0.01;
        let ray = VolumetricRay::new_normalized([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]);
        let res = vr_march_ray(&ray, &g, &tf, &cfg);
        assert!(res.n_steps <= 1);
    }
    #[test]
    fn test_jitter_enabled_still_produces_valid_result() {
        let g = sphere_grid();
        let tf = default_tf();
        let mut cfg = default_cfg();
        cfg.jitter = true;
        cfg.jitter_seed = 12345;
        let cam = default_cam(4, 4);
        let img = vr_render_image(&g, &tf, &cam, &cfg);
        assert!(img.is_ok());
        assert_eq!(img.unwrap_or_default().len(), 16);
    }
    #[test]
    fn test_jitter_zero_seed_does_not_hang() {
        let g = sphere_grid();
        let tf = default_tf();
        let mut cfg = default_cfg();
        cfg.jitter = true;
        cfg.jitter_seed = 0;
        let ray = VolumetricRay::new_normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]);
        let _res = vr_march_ray(&ray, &g, &tf, &cfg);
    }
    #[test]
    fn test_vr_format_config_contains_step_size() {
        let mut cfg = default_cfg();
        cfg.step_size = 0.123;
        let s = vr_format_config(&cfg);
        assert!(s.contains("0.123"), "format={s}");
    }
    #[test]
    fn test_transfer_function_single_point() {
        let pts = vec![TransferPoint {
            density: 0.5,
            color: [0.3, 0.5, 0.7],
            opacity: 0.8,
        }];
        let tf = TransferFunction::new(pts);
        assert!(tf.is_ok());
        let tf = tf.unwrap_or_else(|_| TransferFunction::grayscale(1.0));
        let (c, a) = tf.evaluate(0.0);
        assert!((c[0] - 0.3).abs() < 1e-5);
        assert!((a - 0.8).abs() < 1e-5);
        let (c2, a2) = tf.evaluate(1.0);
        assert!((c2[0] - 0.3).abs() < 1e-5);
        assert!((a2 - 0.8).abs() < 1e-5);
    }
}
