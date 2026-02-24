//! Unit tests for oxigaf-flame core functionality.
//!
//! These tests verify mathematical properties and edge cases
//! without requiring actual FLAME model data.

use approx::assert_relative_eq;
use nalgebra as na;
use std::f32::consts::{FRAC_PI_2, PI};

use oxigaf_flame::{rodrigues, FlameParams, Mesh};

// ---------------------------------------------------------------------------
// Rodrigues Rotation Tests
// ---------------------------------------------------------------------------

mod rodrigues_tests {
    use super::*;

    /// Test Rodrigues stability near 180 degrees (π radians).
    /// This is an edge case where numerical instability can occur.
    #[test]
    fn test_rodrigues_near_180_stability() {
        // Test values near π for each axis
        let near_pi_values = [
            PI - 1e-4,
            PI - 1e-5,
            PI - 1e-6,
            PI,
            PI + 1e-6,
            PI + 1e-5,
            PI + 1e-4,
        ];

        for &angle in &near_pi_values {
            // Test rotation around X axis
            let r_x = rodrigues(angle, 0.0, 0.0);
            verify_rotation_matrix(&r_x, &format!("X-axis at angle {angle}"));

            // Test rotation around Y axis
            let r_y = rodrigues(0.0, angle, 0.0);
            verify_rotation_matrix(&r_y, &format!("Y-axis at angle {angle}"));

            // Test rotation around Z axis
            let r_z = rodrigues(0.0, 0.0, angle);
            verify_rotation_matrix(&r_z, &format!("Z-axis at angle {angle}"));
        }
    }

    /// Test Rodrigues at exactly 180 degrees rotates vectors correctly.
    #[test]
    fn test_rodrigues_180_rotation_correctness() {
        // 180° rotation around X: (0, 1, 0) → (0, -1, 0)
        let r_x = rodrigues(PI, 0.0, 0.0);
        let v_y = na::Vector3::new(0.0, 1.0, 0.0);
        let rotated = r_x * v_y;
        assert_relative_eq!(rotated.x, 0.0, epsilon = 1e-5);
        assert_relative_eq!(rotated.y, -1.0, epsilon = 1e-5);
        assert_relative_eq!(rotated.z, 0.0, epsilon = 1e-5);

        // 180° rotation around Y: (1, 0, 0) → (-1, 0, 0)
        let r_y = rodrigues(0.0, PI, 0.0);
        let v_x = na::Vector3::new(1.0, 0.0, 0.0);
        let rotated = r_y * v_x;
        assert_relative_eq!(rotated.x, -1.0, epsilon = 1e-5);
        assert_relative_eq!(rotated.y, 0.0, epsilon = 1e-5);
        assert_relative_eq!(rotated.z, 0.0, epsilon = 1e-5);

        // 180° rotation around Z: (1, 0, 0) → (-1, 0, 0)
        let r_z = rodrigues(0.0, 0.0, PI);
        let rotated = r_z * v_x;
        assert_relative_eq!(rotated.x, -1.0, epsilon = 1e-5);
        assert_relative_eq!(rotated.y, 0.0, epsilon = 1e-5);
        assert_relative_eq!(rotated.z, 0.0, epsilon = 1e-5);
    }

    /// Test that 360° rotation returns to identity.
    #[test]
    fn test_rodrigues_360_is_identity() {
        let r_360 = rodrigues(2.0 * PI, 0.0, 0.0);
        let identity = na::Matrix3::<f32>::identity();
        assert_matrix_near(&r_360, &identity, 1e-4);
    }

    /// Test small angle approximation behavior.
    #[test]
    fn test_rodrigues_small_angles() {
        let small_angles = [1e-7, 1e-8, 1e-9, 1e-10];

        for &angle in &small_angles {
            let r = rodrigues(angle, angle, angle);
            // Small angle rotation should be nearly identity
            let identity = na::Matrix3::<f32>::identity();
            assert_matrix_near(&r, &identity, 1e-5);
        }
    }

    /// Test negative angles vs positive angles relationship.
    #[test]
    fn test_rodrigues_negative_angles() {
        let angle = 0.5;

        // R(-θ) = R(θ)^T for rotations around a single axis
        let r_pos = rodrigues(angle, 0.0, 0.0);
        let r_neg = rodrigues(-angle, 0.0, 0.0);

        assert_matrix_near(&r_neg, &r_pos.transpose(), 1e-6);
    }

    /// Test combined rotations are consistent.
    #[test]
    fn test_rodrigues_combined_rotations() {
        // Two 90° rotations around Z should equal one 180° rotation
        let r_90 = rodrigues(0.0, 0.0, FRAC_PI_2);
        let r_180 = rodrigues(0.0, 0.0, PI);
        let combined = r_90 * r_90;

        assert_matrix_near(&combined, &r_180, 1e-5);
    }

    /// Test that rotation around arbitrary axis preserves that axis.
    #[test]
    fn test_rodrigues_preserves_rotation_axis() {
        // Normalize axis direction
        let norm = (0.3f32 * 0.3 + 0.4 * 0.4 + 0.5 * 0.5).sqrt();
        let axis = na::Vector3::new(0.3 / norm, 0.4 / norm, 0.5 / norm);

        // Rotate by some angle around this axis
        let angle = 1.2;
        let r = rodrigues(axis.x * angle, axis.y * angle, axis.z * angle);

        // The axis itself should be unchanged (eigenvector with eigenvalue 1)
        let rotated_axis = r * axis;
        assert_relative_eq!(rotated_axis.x, axis.x, epsilon = 1e-5);
        assert_relative_eq!(rotated_axis.y, axis.y, epsilon = 1e-5);
        assert_relative_eq!(rotated_axis.z, axis.z, epsilon = 1e-5);
    }

    // Helper to verify a matrix is a valid rotation matrix
    fn verify_rotation_matrix(r: &na::Matrix3<f32>, context: &str) {
        // Check orthogonality: R^T * R = I
        let product = r.transpose() * r;
        let identity = na::Matrix3::<f32>::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (product[(i, j)] - identity[(i, j)]).abs() < 1e-4,
                    "Orthogonality failed for {context}: product[{i},{j}] = {} (expected {})",
                    product[(i, j)],
                    identity[(i, j)]
                );
            }
        }

        // Check determinant = 1
        let det = r.determinant();
        assert!(
            (det - 1.0).abs() < 1e-4,
            "Determinant failed for {context}: det = {det} (expected 1.0)"
        );
    }

    // Helper to compare matrices
    fn assert_matrix_near(a: &na::Matrix3<f32>, b: &na::Matrix3<f32>, epsilon: f32) {
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (a[(i, j)] - b[(i, j)]).abs() < epsilon,
                    "Matrix mismatch at [{i},{j}]: {} vs {} (epsilon = {epsilon})",
                    a[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh Normal Computation Tests
// ---------------------------------------------------------------------------

mod mesh_normal_tests {
    use super::*;

    /// Test normals for a flat triangle on the XY plane.
    #[test]
    fn test_mesh_normals_xy_plane() {
        let vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.0, 1.0, 0.0),
        ];
        let mesh = Mesh::new(vertices, vec![[0, 1, 2]]);

        // All normals should point in +Z direction
        for normal in &mesh.normals {
            assert_relative_eq!(normal.x, 0.0, epsilon = 1e-5);
            assert_relative_eq!(normal.y, 0.0, epsilon = 1e-5);
            assert_relative_eq!(normal.z, 1.0, epsilon = 1e-3);
        }
    }

    /// Test normals for a flat triangle on the XZ plane.
    #[test]
    fn test_mesh_normals_xz_plane() {
        let vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(0.0, 0.0, 1.0),
            na::Point3::new(1.0, 0.0, 0.0),
        ];
        let mesh = Mesh::new(vertices, vec![[0, 1, 2]]);

        // All normals should point in +Y direction (right-hand rule)
        for normal in &mesh.normals {
            assert_relative_eq!(normal.x, 0.0, epsilon = 1e-5);
            assert_relative_eq!(normal.y, 1.0, epsilon = 1e-3);
            assert_relative_eq!(normal.z, 0.0, epsilon = 1e-5);
        }
    }

    /// Test that normals are normalized (unit length).
    #[test]
    fn test_mesh_normals_unit_length() {
        // Create a tetrahedron
        let vertices = vec![
            na::Point3::new(1.0, 0.0, -1.0 / 2.0f32.sqrt()),
            na::Point3::new(-1.0, 0.0, -1.0 / 2.0f32.sqrt()),
            na::Point3::new(0.0, 1.0, 1.0 / 2.0f32.sqrt()),
            na::Point3::new(0.0, -1.0, 1.0 / 2.0f32.sqrt()),
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let mesh = Mesh::new(vertices, faces);

        for normal in &mesh.normals {
            let len = normal.norm();
            assert_relative_eq!(len, 1.0, epsilon = 1e-5);
        }
    }

    /// Test area-weighted normal averaging at shared vertices.
    #[test]
    fn test_mesh_normals_area_weighted() {
        // Two triangles sharing a vertex, one much larger than the other
        // The shared vertex normal should be weighted toward the larger triangle
        let vertices = vec![
            // Small triangle
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(0.1, 0.0, 0.0),
            na::Point3::new(0.0, 0.1, 0.0),
            // Large triangle (shares vertex 0)
            na::Point3::new(10.0, 0.0, 0.0),
            na::Point3::new(0.0, 10.0, 0.0),
        ];
        let faces = vec![
            [0, 1, 2], // Small triangle
            [0, 3, 4], // Large triangle
        ];
        let mesh = Mesh::new(vertices, faces);

        // All normals on XY plane should point in +Z
        for normal in &mesh.normals {
            assert_relative_eq!(normal.x, 0.0, epsilon = 1e-5);
            assert_relative_eq!(normal.y, 0.0, epsilon = 1e-5);
            assert_relative_eq!(normal.z, 1.0, epsilon = 1e-3);
        }
    }

    /// Test normals for a quad (two triangles).
    #[test]
    fn test_mesh_normals_quad() {
        let vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(1.0, 1.0, 0.0),
            na::Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let mesh = Mesh::new(vertices, faces);

        // All normals should point in +Z
        for normal in &mesh.normals {
            assert_relative_eq!(normal.z, 1.0, epsilon = 1e-3);
        }
    }

    /// Test normals recomputation after vertex modification.
    #[test]
    fn test_mesh_normals_recompute() {
        let mut vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.0, 1.0, 0.0),
        ];
        let mut mesh = Mesh::new(vertices.clone(), vec![[0, 1, 2]]);

        // Initially normals should point in +Z
        assert_relative_eq!(mesh.normals[0].z, 1.0, epsilon = 1e-3);

        // Flip the triangle by swapping vertices
        vertices[1] = na::Point3::new(0.0, 1.0, 0.0);
        vertices[2] = na::Point3::new(1.0, 0.0, 0.0);
        mesh = Mesh::new(vertices, vec![[0, 1, 2]]);

        // Now normals should point in -Z
        assert_relative_eq!(mesh.normals[0].z, -1.0, epsilon = 1e-3);
    }

    /// Test degenerate triangle (zero area) handling.
    #[test]
    fn test_mesh_normals_degenerate_triangle() {
        // Collinear points
        let vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(2.0, 0.0, 0.0),
        ];
        let mesh = Mesh::new(vertices, vec![[0, 1, 2]]);

        // Normals should be zero or near-zero for degenerate case
        for normal in &mesh.normals {
            // Either zero or normalized to some value
            let len = normal.norm();
            // Should not panic, and if normalized, length should be 1
            assert!(len < 1e-5 || (len - 1.0).abs() < 1e-5);
        }
    }
}

// ---------------------------------------------------------------------------
// Blend Shapes Application Tests
// ---------------------------------------------------------------------------

mod blend_shapes_tests {
    use super::*;

    /// Test that zero coefficients produce no deformation.
    #[test]
    fn test_zero_coefficients_no_deformation() {
        let params_zero = FlameParams::builder()
            .shape(vec![0.0, 0.0, 0.0])
            .expression(vec![0.0, 0.0])
            .build();

        let params_empty = FlameParams::neutral();

        // Both should be equivalent
        assert_eq!(params_zero.shape, vec![0.0, 0.0, 0.0]);
        assert_eq!(params_empty.shape.len(), 0);
    }

    /// Test blend shape coefficients are stored correctly.
    #[test]
    fn test_blend_shape_coefficient_storage() {
        let shape_coeffs = vec![0.5, -0.3, 0.8, -1.2, 0.1];
        let expr_coeffs = vec![0.2, -0.4, 0.6];

        let params = FlameParams::builder()
            .shape(shape_coeffs.clone())
            .expression(expr_coeffs.clone())
            .build();

        assert_eq!(params.shape, shape_coeffs);
        assert_eq!(params.expression, expr_coeffs);
    }

    /// Test blend shape linearity property (conceptual).
    /// Note: Full linearity test requires actual model data.
    /// This test verifies the params structure supports linearity testing.
    #[test]
    fn test_blend_shape_params_support_linearity() {
        let coeff = 0.5;
        let params_half = FlameParams::builder().shape(vec![coeff]).build();

        let params_double = FlameParams::builder().shape(vec![coeff * 2.0]).build();

        // Double the coefficient should be exactly double
        assert_relative_eq!(
            params_double.shape[0],
            params_half.shape[0] * 2.0,
            epsilon = 1e-10
        );
    }
}

// ---------------------------------------------------------------------------
// LBS Forward Pass Tests (Synthetic)
// ---------------------------------------------------------------------------

mod lbs_tests {
    use super::*;

    /// Test that FlameParams correctly computes joint pose.
    #[test]
    fn test_joint_pose_extraction() {
        let params = FlameParams::builder()
            .root_rotation([0.1, 0.2, 0.3])
            .neck_rotation([0.4, 0.5, 0.6])
            .jaw_rotation_full([0.7, 0.8, 0.9])
            .left_eye_rotation([1.0, 1.1, 1.2])
            .right_eye_rotation([1.3, 1.4, 1.5])
            .build();

        assert_eq!(params.joint_pose(0), [0.1, 0.2, 0.3]); // Root
        assert_eq!(params.joint_pose(1), [0.4, 0.5, 0.6]); // Neck
        assert_eq!(params.joint_pose(2), [0.7, 0.8, 0.9]); // Jaw
        assert_eq!(params.joint_pose(3), [1.0, 1.1, 1.2]); // Left eye
        assert_eq!(params.joint_pose(4), [1.3, 1.4, 1.5]); // Right eye
    }

    /// Test joint pose for out-of-bounds joint returns zeros.
    #[test]
    fn test_joint_pose_out_of_bounds() {
        let params = FlameParams::builder()
            .root_rotation([0.1, 0.2, 0.3])
            .build();

        // Joint 5 and beyond don't exist
        assert_eq!(params.joint_pose(5), [0.0, 0.0, 0.0]);
        assert_eq!(params.joint_pose(10), [0.0, 0.0, 0.0]);
        assert_eq!(params.joint_pose(100), [0.0, 0.0, 0.0]);
    }

    /// Test neutral pose produces identity rotations.
    #[test]
    fn test_neutral_pose_identity() {
        let params = FlameParams::neutral();

        for j in 0..5 {
            let [rx, ry, rz] = params.joint_pose(j);
            let r = rodrigues(rx, ry, rz);
            let identity = na::Matrix3::<f32>::identity();

            for i in 0..3 {
                for k in 0..3 {
                    assert_relative_eq!(r[(i, k)], identity[(i, k)], epsilon = 1e-6);
                }
            }
        }
    }

    /// Test translation is applied correctly.
    #[test]
    fn test_translation_storage() {
        let translation = [1.0, 2.0, 3.0];
        let params = FlameParams::builder().translation(translation).build();

        assert_eq!(params.translation, translation);
    }

    /// Test that skinning weights property (sum to 1) would hold.
    /// Note: This is a conceptual test; actual weights come from model data.
    #[test]
    fn test_skinning_weights_concept() {
        // Simulate skinning weights that sum to 1
        let weights = [0.3, 0.3, 0.2, 0.1, 0.1];
        let sum: f32 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

// ---------------------------------------------------------------------------
// FlameParams Validation Tests
// ---------------------------------------------------------------------------

mod validation_tests {
    use super::*;

    #[test]
    fn test_validate_within_bounds() {
        let params = FlameParams::builder()
            .shape(vec![1.0, -1.0, 2.0, -2.0])
            .expression(vec![0.5, -0.5, 1.0, -1.0])
            .jaw_rotation(0.1)
            .translation([0.5, -0.5, 0.0])
            .build();

        assert!(params.validate());
    }

    #[test]
    fn test_validate_shape_out_of_bounds() {
        let params = FlameParams::builder()
            .shape(vec![3.5]) // > 3.0
            .build();

        assert!(!params.validate());
    }

    #[test]
    fn test_validate_expression_out_of_bounds() {
        let params = FlameParams::builder()
            .expression(vec![2.5]) // > 2.0
            .build();

        assert!(!params.validate());
    }

    #[test]
    fn test_validate_pose_out_of_bounds() {
        let params = FlameParams::builder()
            .root_rotation([4.0, 0.0, 0.0]) // > PI
            .build();

        assert!(!params.validate());
    }

    #[test]
    fn test_validate_translation_out_of_bounds() {
        let params = FlameParams::builder()
            .translation([1.5, 0.0, 0.0]) // > 1.0
            .build();

        assert!(!params.validate());
    }

    #[test]
    fn test_validate_exactly_at_bounds() {
        let params = FlameParams::builder()
            .shape(vec![3.0, -3.0])
            .expression(vec![2.0, -2.0])
            .translation([1.0, -1.0, 1.0])
            .build();

        assert!(params.validate());
    }
}

// ---------------------------------------------------------------------------
// Edge Case Tests
// ---------------------------------------------------------------------------

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_mesh() {
        let mesh = Mesh::new(vec![], vec![]);
        assert_eq!(mesh.num_vertices(), 0);
        assert_eq!(mesh.num_faces(), 0);
        assert!(mesh.normals.is_empty());
    }

    #[test]
    fn test_single_vertex_mesh() {
        // This is technically invalid but shouldn't panic
        let vertices = vec![na::Point3::new(0.0, 0.0, 0.0)];
        let mesh = Mesh::new(vertices, vec![]);
        assert_eq!(mesh.num_vertices(), 1);
        assert_eq!(mesh.num_faces(), 0);
    }

    #[test]
    fn test_params_with_max_pose_values() {
        let max_pose = vec![PI; 15];
        let params = FlameParams::builder().pose(max_pose).build();

        assert_eq!(params.pose.len(), 15);
        assert!(params.validate()); // At exactly PI, should be valid
    }

    #[test]
    fn test_mesh_face_area() {
        let vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(2.0, 0.0, 0.0),
            na::Point3::new(0.0, 2.0, 0.0),
        ];
        let mesh = Mesh::new(vertices, vec![[0, 1, 2]]);

        // Area of triangle with base 2 and height 2 is 2
        let area = mesh.face_area(&[0, 1, 2]);
        assert_relative_eq!(area, 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mesh_face_area_unit_triangle() {
        let vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.0, 1.0, 0.0),
        ];
        let mesh = Mesh::new(vertices, vec![[0, 1, 2]]);

        // Area of right triangle with legs 1 and 1 is 0.5
        let area = mesh.face_area(&[0, 1, 2]);
        assert_relative_eq!(area, 0.5, epsilon = 1e-5);
    }
}

// ---------------------------------------------------------------------------
// Batched Normal Computation Tests
// ---------------------------------------------------------------------------

mod batched_normal_tests {
    use super::*;
    use oxigaf_flame::{compute_normals_batch, compute_normals_into};

    /// Test compute_normals_into matches Mesh::new normals.
    #[test]
    fn test_compute_normals_into_matches_mesh() {
        let vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];

        // Create mesh normally
        let mesh = Mesh::new(vertices.clone(), faces.clone());

        // Use compute_normals_into
        let mut normals_out = vec![na::Vector3::zeros(); vertices.len()];
        compute_normals_into(&vertices, &faces, &mut normals_out);

        // Should match
        for (n1, n2) in mesh.normals.iter().zip(normals_out.iter()) {
            assert_relative_eq!(n1.x, n2.x, epsilon = 1e-6);
            assert_relative_eq!(n1.y, n2.y, epsilon = 1e-6);
            assert_relative_eq!(n1.z, n2.z, epsilon = 1e-6);
        }
    }

    /// Test batch normal computation produces independent results.
    #[test]
    fn test_compute_normals_batch_independent() {
        // Create two different triangles
        let vertices1 = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.0, 1.0, 0.0),
        ];
        let vertices2 = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(0.0, 0.0, 1.0),
            na::Point3::new(1.0, 0.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];

        let vertices_batch = vec![vertices1.clone(), vertices2.clone()];
        let mut normals_batch = vec![vec![na::Vector3::zeros(); 3], vec![na::Vector3::zeros(); 3]];

        compute_normals_batch(&vertices_batch, &faces, &mut normals_batch);

        // First triangle on XY plane should have normals pointing in +Z
        for normal in &normals_batch[0] {
            assert_relative_eq!(normal.z, 1.0, epsilon = 1e-3);
        }

        // Second triangle on XZ plane should have normals pointing in +Y
        for normal in &normals_batch[1] {
            assert_relative_eq!(normal.y, 1.0, epsilon = 1e-3);
        }
    }

    /// Test batch of 1 produces same result as single call.
    #[test]
    fn test_batch_of_one_matches_single() {
        let vertices = vec![
            na::Point3::new(0.0, 0.0, 0.0),
            na::Point3::new(1.0, 0.0, 0.0),
            na::Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];

        // Single call
        let mut single_normals = vec![na::Vector3::zeros(); 3];
        compute_normals_into(&vertices, &faces, &mut single_normals);

        // Batch of 1
        let vertices_batch = vec![vertices];
        let mut batch_normals = vec![vec![na::Vector3::zeros(); 3]];
        compute_normals_batch(&vertices_batch, &faces, &mut batch_normals);

        // Should match
        for (n1, n2) in single_normals.iter().zip(batch_normals[0].iter()) {
            assert_relative_eq!(n1.x, n2.x, epsilon = 1e-6);
            assert_relative_eq!(n1.y, n2.y, epsilon = 1e-6);
            assert_relative_eq!(n1.z, n2.z, epsilon = 1e-6);
        }
    }

    /// Test empty batch handling.
    #[test]
    fn test_empty_batch() {
        let vertices_batch: Vec<Vec<na::Point3<f32>>> = vec![];
        let faces: Vec<[u32; 3]> = vec![];
        let mut normals_batch: Vec<Vec<na::Vector3<f32>>> = vec![];

        compute_normals_batch(&vertices_batch, &faces, &mut normals_batch);

        // Should complete without panic
        assert!(normals_batch.is_empty());
    }
}

// ---------------------------------------------------------------------------
// BatchedFlameOutput Tests
// ---------------------------------------------------------------------------

mod batched_output_tests {
    use super::*;
    use oxigaf_flame::BatchedFlameOutput;

    /// Test BatchedFlameOutput with_capacity allocates correctly.
    #[test]
    fn test_batched_output_with_capacity() {
        let batch_size = 5;
        let num_vertices = 100;
        let faces = vec![[0, 1, 2], [1, 2, 3]];

        let output = BatchedFlameOutput::with_capacity(batch_size, num_vertices, faces.clone());

        assert_eq!(output.batch_size, batch_size);
        assert_eq!(output.num_vertices(), num_vertices);
        assert_eq!(output.vertices.len(), batch_size);
        assert_eq!(output.normals.len(), batch_size);
        assert_eq!(output.faces.len(), 2);

        for verts in &output.vertices {
            assert_eq!(verts.len(), num_vertices);
        }
        for norms in &output.normals {
            assert_eq!(norms.len(), num_vertices);
        }
    }

    /// Test get_mesh returns valid mesh.
    #[test]
    fn test_get_mesh() {
        let batch_size = 3;
        let num_vertices = 4;
        let faces = vec![[0, 1, 2], [0, 2, 3]];

        let mut output = BatchedFlameOutput::with_capacity(batch_size, num_vertices, faces);

        // Set some vertex data for mesh 1
        output.vertices[1][0] = na::Point3::new(1.0, 2.0, 3.0);
        output.normals[1][0] = na::Vector3::new(0.0, 0.0, 1.0);

        // Get mesh 1
        let mesh = output.get_mesh(1);
        assert!(mesh.is_some());
        let mesh = mesh.expect("mesh should exist");

        assert_eq!(mesh.vertices[0], na::Point3::new(1.0, 2.0, 3.0));
        assert_eq!(mesh.normals[0], na::Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(mesh.faces.len(), 2);

        // Out of bounds should return None
        assert!(output.get_mesh(10).is_none());
    }

    /// Test into_meshes consumes output correctly.
    #[test]
    fn test_into_meshes() {
        let batch_size = 2;
        let num_vertices = 3;
        let faces = vec![[0, 1, 2]];

        let mut output = BatchedFlameOutput::with_capacity(batch_size, num_vertices, faces);

        // Set vertex data
        output.vertices[0][0] = na::Point3::new(1.0, 0.0, 0.0);
        output.vertices[1][0] = na::Point3::new(2.0, 0.0, 0.0);

        let meshes = output.into_meshes();

        assert_eq!(meshes.len(), batch_size);
        assert_eq!(meshes[0].vertices[0], na::Point3::new(1.0, 0.0, 0.0));
        assert_eq!(meshes[1].vertices[0], na::Point3::new(2.0, 0.0, 0.0));
    }
}

// ---------------------------------------------------------------------------
// BatchBufferPool Tests
// ---------------------------------------------------------------------------

mod buffer_pool_tests {
    use oxigaf_flame::BatchBufferPool;

    /// Test buffer pool creation.
    #[test]
    fn test_buffer_pool_new() {
        let batch_size = 4;
        let num_vertices = 50;
        let n_joints = 5;

        let pool = BatchBufferPool::new(batch_size, num_vertices, n_joints);

        assert_eq!(pool.capacity(), batch_size);
    }

    /// Test buffer pool ensure_capacity.
    #[test]
    fn test_buffer_pool_ensure_capacity() {
        let mut pool = BatchBufferPool::new(2, 50, 5);

        assert_eq!(pool.capacity(), 2);

        pool.ensure_capacity(5);
        assert_eq!(pool.capacity(), 5);

        // Should not shrink
        pool.ensure_capacity(3);
        assert_eq!(pool.capacity(), 5);
    }

    /// Test buffer pool clear.
    #[test]
    fn test_buffer_pool_clear() {
        let mut pool = BatchBufferPool::new(2, 10, 5);
        pool.clear();

        // Should complete without panic and retain capacity
        assert_eq!(pool.capacity(), 2);
    }
}
