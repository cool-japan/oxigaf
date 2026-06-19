//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::{dot3, normalize3, quat_inverse, quat_rotate};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }
    fn approx_eq3(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        approx_eq(a[0], b[0], eps) && approx_eq(a[1], b[1], eps) && approx_eq(a[2], b[2], eps)
    }
    #[test]
    fn test_bbox_center() {
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [2.0; 3],
        };
        assert_eq!(bbox.center(), [1.0, 1.0, 1.0]);
    }
    #[test]
    fn test_bbox_size() {
        let bbox = GaussianBBox {
            min: [-1.0; 3],
            max: [1.0; 3],
        };
        assert_eq!(bbox.size(), [2.0, 2.0, 2.0]);
    }
    #[test]
    fn test_bbox_volume() {
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [2.0, 3.0, 4.0],
        };
        assert!(approx_eq(bbox.volume(), 24.0, 1e-5));
    }
    #[test]
    fn test_bbox_diagonal() {
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        assert!(approx_eq(bbox.diagonal(), 3.0_f32.sqrt(), 1e-5));
    }
    #[test]
    fn test_bbox_contains() {
        let bbox = GaussianBBox {
            min: [-1.0; 3],
            max: [1.0; 3],
        };
        assert!(bbox.contains([0.0, 0.0, 0.0]));
        assert!(bbox.contains([1.0, 1.0, 1.0]));
        assert!(!bbox.contains([1.5, 0.0, 0.0]));
    }
    #[test]
    fn test_bbox_intersects() {
        let a = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let b = GaussianBBox {
            min: [0.5; 3],
            max: [1.5; 3],
        };
        let c = GaussianBBox {
            min: [2.0; 3],
            max: [3.0; 3],
        };
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }
    #[test]
    fn test_bbox_union() {
        let a = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let b = GaussianBBox {
            min: [-1.0; 3],
            max: [0.5; 3],
        };
        let u = a.union(&b);
        assert_eq!(u.min, [-1.0; 3]);
        assert_eq!(u.max, [1.0; 3]);
    }
    #[test]
    fn test_bbox_iou_identical() {
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        assert!(approx_eq(bbox.iou(&bbox), 1.0, 1e-5));
    }
    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let b = GaussianBBox {
            min: [2.0; 3],
            max: [3.0; 3],
        };
        assert!(approx_eq(bbox_iou(&a, &b), 0.0, 1e-5));
    }
    fn bbox_iou(a: &GaussianBBox, b: &GaussianBBox) -> f32 {
        a.iou(b)
    }
    #[test]
    fn test_bbox_intersection_none() {
        let a = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let b = GaussianBBox {
            min: [2.0; 3],
            max: [3.0; 3],
        };
        assert!(a.intersection(&b).is_none());
    }
    #[test]
    fn test_bbox_intersection_some() {
        let a = GaussianBBox {
            min: [0.0; 3],
            max: [2.0; 3],
        };
        let b = GaussianBBox {
            min: [1.0; 3],
            max: [3.0; 3],
        };
        let i = a.intersection(&b).expect("should intersect");
        assert_eq!(i.min, [1.0; 3]);
        assert_eq!(i.max, [2.0; 3]);
    }
    #[test]
    fn test_bbox_expand() {
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let exp = bbox.expand(0.5);
        assert_eq!(exp.min, [-0.5; 3]);
        assert_eq!(exp.max, [1.5; 3]);
    }
    #[test]
    fn test_sphere_contains() {
        let s = BoundingSphere {
            center: [0.0; 3],
            radius: 1.0,
        };
        assert!(s.contains([0.0, 0.0, 0.5]));
        assert!(!s.contains([1.0, 1.0, 0.0]));
    }
    #[test]
    fn test_sphere_intersects_overlap() {
        let a = BoundingSphere {
            center: [0.0; 3],
            radius: 1.0,
        };
        let b = BoundingSphere {
            center: [1.5, 0.0, 0.0],
            radius: 1.0,
        };
        assert!(a.intersects(&b));
    }
    #[test]
    fn test_sphere_intersects_no_overlap() {
        let a = BoundingSphere {
            center: [0.0; 3],
            radius: 1.0,
        };
        let b = BoundingSphere {
            center: [3.0, 0.0, 0.0],
            radius: 1.0,
        };
        assert!(!a.intersects(&b));
    }
    #[test]
    fn test_rigid_identity_apply() {
        let t = RigidTransform::identity();
        let p = [1.0, 2.0, 3.0];
        let out = t.apply_to_point(p);
        assert!(approx_eq3(out, p, 1e-5));
    }
    #[test]
    fn test_rigid_translation() {
        let t = RigidTransform::translation_only([1.0, 0.0, 0.0]);
        let out = t.apply_to_point([0.0, 0.0, 0.0]);
        assert!(approx_eq3(out, [1.0, 0.0, 0.0], 1e-5));
    }
    #[test]
    fn test_rigid_scale() {
        let t = RigidTransform::from_scale(2.0);
        let out = t.apply_to_point([1.0, 1.0, 1.0]);
        assert!(approx_eq3(out, [2.0, 2.0, 2.0], 1e-5));
    }
    #[test]
    fn test_rigid_rotation_90_deg_y() {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let t = RigidTransform {
            rotation: [0.0, s, 0.0, s],
            translation: [0.0; 3],
            scale: 1.0,
        };
        let out = t.apply_to_point([1.0, 0.0, 0.0]);
        assert!(approx_eq3(out, [0.0, 0.0, -1.0], 1e-5));
    }
    #[test]
    fn test_rigid_direction() {
        let t = RigidTransform::identity();
        let d = [0.0, 1.0, 0.0];
        assert!(approx_eq3(t.apply_to_direction(d), d, 1e-5));
    }
    #[test]
    fn test_rigid_compose_translations() {
        let a = RigidTransform::translation_only([1.0, 0.0, 0.0]);
        let b = RigidTransform::translation_only([0.0, 1.0, 0.0]);
        let c = a.compose(&b);
        let out = c.apply_to_point([0.0; 3]);
        assert!(approx_eq3(out, [1.0, 1.0, 0.0], 1e-5));
    }
    #[test]
    fn test_rigid_inverse_translation() {
        let t = RigidTransform::translation_only([3.0, -2.0, 1.0]);
        let inv = t.inverse();
        let p = [5.0, 5.0, 5.0];
        let round_trip = inv.apply_to_point(t.apply_to_point(p));
        assert!(approx_eq3(round_trip, p, 1e-4));
    }
    #[test]
    fn test_compute_bbox_known() {
        let positions = vec![0.0f32, 0.0, 0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let bbox = compute_gaussian_bbox(&positions).expect("bbox");
        assert_eq!(bbox.min, [-1.0, -2.0, -3.0]);
        assert_eq!(bbox.max, [1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_compute_bbox_single_point() {
        let positions = vec![5.0f32, -3.0, 1.0];
        let bbox = compute_gaussian_bbox(&positions).expect("bbox");
        assert_eq!(bbox.min, [5.0, -3.0, 1.0]);
        assert_eq!(bbox.max, [5.0, -3.0, 1.0]);
    }
    #[test]
    fn test_compute_bbox_empty_error() {
        let result = compute_gaussian_bbox(&[]);
        assert!(matches!(result, Err(GeometryError::EmptyCloud)));
    }
    #[test]
    fn test_compute_sphere_contains_all() {
        let positions = vec![
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let sphere = compute_bounding_sphere(&positions).expect("sphere");
        let n = positions.len() / 3;
        for i in 0..n {
            let base = i * 3;
            let p = [positions[base], positions[base + 1], positions[base + 2]];
            assert!(
                sphere.contains(p)
                    || approx_eq(
                        (positions[base] - sphere.center[0]).powi(2)
                            + (positions[base + 1] - sphere.center[1]).powi(2)
                            + (positions[base + 2] - sphere.center[2]).powi(2),
                        sphere.radius * sphere.radius,
                        1e-3,
                    ),
                "Point {i} {:?} not in sphere {:?}",
                p,
                sphere
            );
        }
    }
    #[test]
    fn test_compute_sphere_single_point() {
        let positions = vec![1.0f32, 2.0, 3.0];
        let sphere = compute_bounding_sphere(&positions).expect("sphere");
        assert!(approx_eq3(sphere.center, [1.0, 2.0, 3.0], 1e-5));
        assert!(sphere.radius >= 0.0);
    }
    #[test]
    fn test_centroid_known() {
        let positions = vec![0.0f32, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0];
        let c = compute_centroid(&positions).expect("centroid");
        assert!(approx_eq3(c, [2.0 / 3.0, 4.0 / 3.0, 0.0], 1e-5));
    }
    #[test]
    fn test_centroid_symmetric() {
        let positions = vec![-1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0];
        let c = compute_centroid(&positions).expect("centroid");
        assert!(approx_eq3(c, [0.0; 3], 1e-5));
    }
    #[test]
    fn test_centroid_empty_error() {
        let result = compute_centroid(&[]);
        assert!(matches!(result, Err(GeometryError::EmptyCloud)));
    }
    #[test]
    fn test_geometry_stats_basic() {
        let positions = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let scales = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let stats = compute_geometry_stats(&positions, &scales).expect("stats");
        assert_eq!(stats.n_gaussians, 2);
        assert!(approx_eq(stats.mean_scale, 1.0, 1e-5));
    }
    #[test]
    fn test_geometry_stats_count_mismatch() {
        let positions = vec![0.0f32; 6];
        let scales = vec![0.0f32; 9];
        let result = compute_geometry_stats(&positions, &scales);
        assert!(matches!(result, Err(GeometryError::CountMismatch { .. })));
    }
    #[test]
    fn test_transform_positions_translation() {
        let mut positions = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let t = RigidTransform::translation_only([1.0, 2.0, 3.0]);
        transform_positions(&mut positions, &t).expect("transform");
        assert!(approx_eq3(
            [positions[0], positions[1], positions[2]],
            [1.0, 2.0, 3.0],
            1e-5
        ));
        assert!(approx_eq3(
            [positions[3], positions[4], positions[5]],
            [2.0, 2.0, 3.0],
            1e-5
        ));
    }
    #[test]
    fn test_transform_positions_scale() {
        let mut positions = vec![1.0f32, 0.0, 0.0, 0.0, 2.0, 0.0];
        let t = RigidTransform::from_scale(3.0);
        transform_positions(&mut positions, &t).expect("transform");
        assert!(approx_eq3(
            [positions[0], positions[1], positions[2]],
            [3.0, 0.0, 0.0],
            1e-5
        ));
        assert!(approx_eq3(
            [positions[3], positions[4], positions[5]],
            [0.0, 6.0, 0.0],
            1e-5
        ));
    }
    #[test]
    fn test_transform_positions_invalid_length() {
        let mut positions = vec![0.0f32, 1.0];
        let t = RigidTransform::identity();
        let result = transform_positions(&mut positions, &t);
        assert!(matches!(
            result,
            Err(GeometryError::InvalidPositionLength { .. })
        ));
    }
    #[test]
    fn test_transform_rotations_identity() {
        let mut rotations = vec![0.0f32, 0.0, 0.0, 1.0];
        let t = RigidTransform::identity();
        transform_rotations(&mut rotations, &t).expect("transform");
        assert!(approx_eq3(
            [rotations[0], rotations[1], rotations[2]],
            [0.0, 0.0, 0.0],
            1e-5,
        ));
        assert!(approx_eq(rotations[3], 1.0, 1e-5));
    }
    #[test]
    fn test_transform_rotations_invalid_length() {
        let mut rotations = vec![0.0f32; 5];
        let t = RigidTransform::identity();
        let result = transform_rotations(&mut rotations, &t);
        assert!(matches!(
            result,
            Err(GeometryError::InvalidRotationLength { .. })
        ));
    }
    #[test]
    fn test_center_at_origin_centroid_zero() {
        let mut positions = vec![-1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let centroid = center_at_origin(&mut positions).expect("center");
        let new_centroid = compute_centroid(&positions).expect("centroid");
        assert!(approx_eq3(new_centroid, [0.0; 3], 1e-5));
        assert!(approx_eq(centroid[1], 2.0 / 3.0, 1e-5));
    }
    #[test]
    fn test_center_at_origin_empty_error() {
        let mut positions: Vec<f32> = vec![];
        let result = center_at_origin(&mut positions);
        assert!(matches!(result, Err(GeometryError::EmptyCloud)));
    }
    #[test]
    fn test_normalize_to_unit_cube_fits() {
        let mut positions = vec![
            -1.0f32, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
        ];
        normalize_to_unit_cube(&mut positions).expect("normalize");
        for v in &positions {
            assert!(
                *v >= -0.5 - 1e-5 && *v <= 0.5 + 1e-5,
                "Value {v} out of [-0.5, 0.5]"
            );
        }
    }
    #[test]
    fn test_normalize_to_unit_cube_returns_scale() {
        let mut positions = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0];
        let scale = normalize_to_unit_cube(&mut positions).expect("normalize");
        assert!(approx_eq(scale, 10.0, 1e-3));
    }
    #[test]
    fn test_filter_by_bbox_inside() {
        let positions = vec![0.5f32, 0.5, 0.5, 2.0, 2.0, 2.0];
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let kept = filter_by_bbox(&positions, &bbox).expect("filter");
        assert_eq!(kept, vec![0]);
    }
    #[test]
    fn test_filter_by_bbox_all_outside() {
        let positions = vec![5.0f32, 5.0, 5.0, -5.0, -5.0, -5.0];
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let kept = filter_by_bbox(&positions, &bbox).expect("filter");
        assert!(kept.is_empty());
    }
    #[test]
    fn test_filter_by_bbox_boundary() {
        let positions = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let kept = filter_by_bbox(&positions, &bbox).expect("filter");
        assert_eq!(kept.len(), 2);
    }
    #[test]
    fn test_filter_by_sphere_inside() {
        let positions = vec![0.0f32, 0.0, 0.0, 5.0, 0.0, 0.0];
        let kept = filter_by_sphere(&positions, [0.0; 3], 1.0).expect("filter");
        assert_eq!(kept, vec![0]);
    }
    #[test]
    fn test_filter_by_sphere_all_inside() {
        let positions = vec![0.0f32, 0.0, 0.0, 0.5, 0.0, 0.0];
        let kept = filter_by_sphere(&positions, [0.0; 3], 1.0).expect("filter");
        assert_eq!(kept.len(), 2);
    }
    #[test]
    fn test_filter_by_sphere_empty_error() {
        let result = filter_by_sphere(&[], [0.0; 3], 1.0);
        assert!(matches!(result, Err(GeometryError::EmptyCloud)));
    }
    #[test]
    fn test_cloud_distance_known() {
        let a = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![3.0f32, 0.0, 0.0, 3.0, 0.0, 0.0];
        let dist = cloud_distance(&a, &b).expect("distance");
        assert!(approx_eq(dist, 3.0, 1e-5));
    }
    #[test]
    fn test_cloud_distance_zero() {
        let a = vec![1.0f32, 1.0, 1.0];
        let b = vec![1.0f32, 1.0, 1.0];
        let dist = cloud_distance(&a, &b).expect("distance");
        assert!(approx_eq(dist, 0.0, 1e-5));
    }
    #[test]
    fn test_mean_scale_zero_log() {
        let scales = vec![0.0f32; 9];
        let mean = mean_gaussian_scale(&scales).expect("mean");
        assert!(approx_eq(mean, 1.0, 1e-5));
    }
    #[test]
    fn test_mean_scale_known() {
        let v = 2.0f32.ln();
        let scales = vec![v, v, v];
        let mean = mean_gaussian_scale(&scales).expect("mean");
        assert!(approx_eq(mean, 2.0, 1e-5));
    }
    #[test]
    fn test_rescale_achieves_target() {
        let mut scales = vec![0.0f32; 9];
        let delta = rescale_gaussians(&mut scales, 2.0).expect("rescale");
        let new_mean = mean_gaussian_scale(&scales).expect("mean");
        assert!(approx_eq(new_mean, 2.0, 1e-4));
        assert!(approx_eq(delta, 2.0f32.ln(), 1e-5));
    }
    #[test]
    fn test_rescale_nonpositive_target_error() {
        let mut scales = vec![0.0f32; 3];
        let result = rescale_gaussians(&mut scales, -1.0);
        assert!(matches!(
            result,
            Err(GeometryError::InvalidTransform { .. })
        ));
    }
    #[test]
    fn test_nn_distances_k1() {
        let positions = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let dists = nearest_neighbor_distances(&positions, 1).expect("nn");
        assert_eq!(dists.len(), 3);
        assert!(approx_eq(dists[0], 1.0, 1e-5));
        assert!(approx_eq(dists[1], 1.0, 1e-5));
        assert!(approx_eq(dists[2], 2.0, 1e-5));
    }
    #[test]
    fn test_nn_distances_k_too_large_error() {
        let positions = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let result = nearest_neighbor_distances(&positions, 2);
        assert!(matches!(
            result,
            Err(GeometryError::InvalidTransform { .. })
        ));
    }
    #[test]
    fn test_spatial_coverage_full() {
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [2.0; 3],
        };
        let positions: Vec<f32> = vec![
            0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5,
            1.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5,
        ];
        let coverage = spatial_coverage(&positions, &bbox, 2).expect("coverage");
        assert!(
            approx_eq(coverage, 1.0, 1e-5),
            "Expected full coverage, got {coverage}"
        );
    }
    #[test]
    fn test_spatial_coverage_half() {
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [2.0; 3],
        };
        let positions: Vec<f32> = vec![0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2];
        let coverage = spatial_coverage(&positions, &bbox, 2).expect("coverage");
        assert!(
            approx_eq(coverage, 1.0 / 8.0, 1e-5),
            "Expected 1/8, got {coverage}"
        );
    }
    #[test]
    fn test_spatial_coverage_zero_resolution_error() {
        let bbox = GaussianBBox {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let positions = vec![0.5f32, 0.5, 0.5];
        let result = spatial_coverage(&positions, &bbox, 0);
        assert!(matches!(
            result,
            Err(GeometryError::InvalidTransform { .. })
        ));
    }
    #[test]
    fn test_quat_rotate_identity_preserves_vector() {
        let q = [0.0f32, 0.0, 0.0, 1.0];
        let v = [1.0, 2.0, 3.0];
        let out = quat_rotate(q, v);
        assert!(approx_eq3(out, v, 1e-5));
    }
    #[test]
    fn test_quat_inverse_of_identity() {
        let q = [0.0f32, 0.0, 0.0, 1.0];
        let qi = quat_inverse(q);
        assert_eq!(qi, [0.0, 0.0, 0.0, 1.0]);
    }
    #[test]
    fn test_normalize3_zero_vector() {
        let out = normalize3([0.0; 3]);
        assert_eq!(out, [0.0; 3]);
    }
    #[test]
    fn test_dot3() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!(approx_eq(dot3(a, b), 32.0, 1e-5));
    }
    #[test]
    fn test_compute_obb_returns_centroid_center() {
        let positions = vec![-1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (center, half_extents, q) = compute_obb(&positions).expect("obb");
        assert!(approx_eq3(center, [0.0; 3], 1e-5));
        assert!(approx_eq(half_extents[0], 1.0, 1e-5));
        assert!(approx_eq(q[3], 1.0, 1e-5));
    }
    #[test]
    fn test_obb_axis_aligned_box() {
        let mut positions = Vec::new();
        for &x in &[-1.0f32, 1.0] {
            for &y in &[-2.0f32, 2.0] {
                for &z in &[-3.0f32, 3.0] {
                    positions.extend_from_slice(&[x, y, z]);
                }
            }
        }
        let (center, extents, q) = compute_obb(&positions).unwrap();
        assert!(center.iter().all(|&c| c.abs() < 0.1));
        let mut e = extents;
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((e[0] - 1.0).abs() < 0.2, "smallest half-extent: {}", e[0]);
        assert!((e[1] - 2.0).abs() < 0.2, "middle half-extent: {}", e[1]);
        assert!((e[2] - 3.0).abs() < 0.2, "largest half-extent: {}", e[2]);
        assert!(
            q[3].abs() > 0.7,
            "quaternion should be near-identity for axis-aligned box, w={}",
            q[3]
        );
    }
    /// Verify that an axis-aligned rectangular box yields near-identity rotation
    /// and correct half-extents.  The box spans x∈[-1,1], y∈[-2,2], z∈[-3,3].
    #[test]
    fn test_obb_axis_aligned_identity_rotation() {
        let mut positions = Vec::new();
        for &x in &[-1.0f32, 1.0] {
            for &y in &[-2.0f32, 2.0] {
                for &z in &[-3.0f32, 3.0] {
                    positions.extend_from_slice(&[x, y, z]);
                }
            }
        }
        let (center, extents, q) = compute_obb(&positions).expect("obb for axis-aligned box");
        assert!(
            center.iter().all(|&c| c.abs() < 1e-4),
            "OBB center should be near origin, got {center:?}"
        );
        let mut e_sorted = extents;
        e_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            (e_sorted[0] - 1.0).abs() < 0.15,
            "smallest half-extent expected ~1, got {}",
            e_sorted[0]
        );
        assert!(
            (e_sorted[1] - 2.0).abs() < 0.15,
            "middle half-extent expected ~2, got {}",
            e_sorted[1]
        );
        assert!(
            (e_sorted[2] - 3.0).abs() < 0.15,
            "largest half-extent expected ~3, got {}",
            e_sorted[2]
        );
        let is_axis_aligned = |v: [f32; 3]| -> bool {
            let large = v.iter().filter(|&&c| c.abs() > 0.9).count();
            let small = v.iter().filter(|&&c| c.abs() < 0.1).count();
            large == 1 && small == 2
        };
        let rx = quat_rotate(q, [1.0, 0.0, 0.0]);
        let ry = quat_rotate(q, [0.0, 1.0, 0.0]);
        let rz = quat_rotate(q, [0.0, 0.0, 1.0]);
        assert!(
            is_axis_aligned(rx),
            "rotation of [1,0,0] should be axis-aligned for axis-aligned box, got {rx:?}"
        );
        assert!(
            is_axis_aligned(ry),
            "rotation of [0,1,0] should be axis-aligned for axis-aligned box, got {ry:?}"
        );
        assert!(
            is_axis_aligned(rz),
            "rotation of [0,0,1] should be axis-aligned for axis-aligned box, got {rz:?}"
        );
    }
    /// A 45-degree rotated cloud: five collinear points along y = x in the XY
    /// plane.  PCA must recover the primary axis as [1/√2, 1/√2, 0] (up to sign).
    #[test]
    fn test_obb_45_degree_rotated_cloud() {
        let positions: Vec<f32> = vec![
            -2.0, -2.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0,
        ];
        let (center, extents, q) = compute_obb(&positions).expect("obb for 45-degree cloud");
        assert!(
            center.iter().all(|&c| c.abs() < 1e-4),
            "OBB center should be near origin, got {center:?}"
        );
        let expected_largest = 2.0f32 * std::f32::consts::SQRT_2;
        assert!(
            (extents[0] - expected_largest).abs() < 0.05,
            "largest half-extent expected {expected_largest:.4}, got {}",
            extents[0]
        );
        let rotated_x = quat_rotate(q, [1.0, 0.0, 0.0]);
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let xy_equal = (rotated_x[0].abs() - inv_sqrt2).abs() < 0.05
            && (rotated_x[1].abs() - inv_sqrt2).abs() < 0.05;
        let z_near_zero = rotated_x[2].abs() < 0.05;
        let same_sign = (rotated_x[0] - rotated_x[1]).abs() < 0.05;
        assert!(
            xy_equal && z_near_zero && same_sign,
            "primary axis expected [±1/√2, ±1/√2, 0], got rotated_x={rotated_x:?}"
        );
    }
    /// A degenerate single-point cloud.  After centering, the covariance is the
    /// zero matrix; eigendecomposition succeeds with all-zero eigenvalues, so
    /// all half-extents should be zero (or near-zero).
    #[test]
    fn test_obb_single_point_zero_extents() {
        let positions = vec![3.0f32, -1.5, 7.0];
        let (center, extents, _q) = compute_obb(&positions).expect("obb for single point");
        assert!(
            approx_eq3(center, [3.0, -1.5, 7.0], 1e-4),
            "OBB center for single point should equal the point, got {center:?}"
        );
        for (i, &he) in extents.iter().enumerate() {
            assert!(
                he.abs() < 1e-4,
                "half-extent[{i}] should be ~0 for a single point, got {he}"
            );
        }
    }

    /// An empty positions slice must not panic and must return an error (not a
    /// default OBB value that could silently mask a caller bug).
    ///
    /// The underlying `compute_centroid` returns `GeometryError::EmptyCloud` for
    /// zero-length input, which `compute_obb` propagates unchanged.
    #[test]
    fn test_obb_empty_input_no_panic() {
        let result = compute_obb(&[]);
        assert!(
            matches!(result, Err(GeometryError::EmptyCloud)),
            "compute_obb with empty input must return Err(EmptyCloud), got {result:?}"
        );
    }
}
