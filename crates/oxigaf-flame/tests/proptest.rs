//! Property-based tests for oxigaf-flame using proptest.
//!
//! Tests mathematical invariants and properties that should hold
//! for all valid inputs.

use approx::assert_relative_eq;
use nalgebra as na;
use proptest::prelude::*;

use oxigaf_flame::{FlameModel, FlameParams};

// ---------------------------------------------------------------------------
// Rodrigues Rotation Properties
// ---------------------------------------------------------------------------

/// Test that Rodrigues rotation matrices are orthogonal: R^T * R = I
#[test]
fn rodrigues_orthogonality() {
    proptest!(|(
        rx in -std::f32::consts::PI..std::f32::consts::PI,
        ry in -std::f32::consts::PI..std::f32::consts::PI,
        rz in -std::f32::consts::PI..std::f32::consts::PI,
    )| {
        let r = oxigaf_flame::rodrigues(rx, ry, rz);
        let rt = r.transpose();
        let product = rt * r;
        let identity = na::Matrix3::<f32>::identity();

        // Check orthogonality with reasonable epsilon for f32
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(
                    product[(i, j)],
                    identity[(i, j)],
                    epsilon = 1e-5
                );
            }
        }
    });
}

/// Test that Rodrigues rotation matrices have determinant = 1
#[test]
fn rodrigues_determinant() {
    proptest!(|(
        rx in -std::f32::consts::PI..std::f32::consts::PI,
        ry in -std::f32::consts::PI..std::f32::consts::PI,
        rz in -std::f32::consts::PI..std::f32::consts::PI,
    )| {
        let r = oxigaf_flame::rodrigues(rx, ry, rz);
        let det = r.determinant();

        assert_relative_eq!(det, 1.0, epsilon = 1e-5);
    });
}

/// Test that R^T = R^-1 for rotation matrices
#[test]
fn rodrigues_transpose_is_inverse() {
    proptest!(|(
        rx in -std::f32::consts::PI..std::f32::consts::PI,
        ry in -std::f32::consts::PI..std::f32::consts::PI,
        rz in -std::f32::consts::PI..std::f32::consts::PI,
    )| {
        let r = oxigaf_flame::rodrigues(rx, ry, rz);
        let rt = r.transpose();

        // R^-1 should exist for non-singular matrix
        let r_inv = r.try_inverse()
            .expect("Rotation matrix should be invertible");

        // Check R^T = R^-1
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(
                    rt[(i, j)],
                    r_inv[(i, j)],
                    epsilon = 1e-5
                );
            }
        }
    });
}

/// Test that zero rotation gives identity matrix
#[test]
fn rodrigues_zero_is_identity() {
    let r = oxigaf_flame::rodrigues(0.0, 0.0, 0.0);
    let identity = na::Matrix3::<f32>::identity();

    for i in 0..3 {
        for j in 0..3 {
            assert_relative_eq!(r[(i, j)], identity[(i, j)], epsilon = 1e-6);
        }
    }
}

/// Test that rotation preserves vector length
#[test]
fn rodrigues_preserves_length() {
    proptest!(|(
        rx in -std::f32::consts::PI..std::f32::consts::PI,
        ry in -std::f32::consts::PI..std::f32::consts::PI,
        rz in -std::f32::consts::PI..std::f32::consts::PI,
        vx in -10.0f32..10.0f32,
        vy in -10.0f32..10.0f32,
        vz in -10.0f32..10.0f32,
    )| {
        let r = oxigaf_flame::rodrigues(rx, ry, rz);
        let v = na::Vector3::new(vx, vy, vz);
        let rotated = r * v;

        let original_len = v.norm();
        let rotated_len = rotated.norm();

        assert_relative_eq!(original_len, rotated_len, epsilon = 1e-5);
    });
}

// ---------------------------------------------------------------------------
// FlameParams Properties
// ---------------------------------------------------------------------------

/// Test that FlameParams validation correctly identifies out-of-range values
#[test]
fn params_validation_shape_range() {
    proptest!(|(
        val in -5.0f32..5.0f32,
    )| {
        let params = FlameParams::builder()
            .shape(vec![val])
            .build();

        let is_valid = params.validate();
        let should_be_valid = val.abs() <= 3.0;

        assert_eq!(is_valid, should_be_valid,
            "Shape value {} should be {} but validation returned {}",
            val, should_be_valid, is_valid);
    });
}

/// Test that FlameParams validation correctly identifies out-of-range expression values
#[test]
fn params_validation_expression_range() {
    proptest!(|(
        val in -4.0f32..4.0f32,
    )| {
        let params = FlameParams::builder()
            .expression(vec![val])
            .build();

        let is_valid = params.validate();
        let should_be_valid = val.abs() <= 2.0;

        assert_eq!(is_valid, should_be_valid,
            "Expression value {} should be {} but validation returned {}",
            val, should_be_valid, is_valid);
    });
}

/// Test that FlameParams validation correctly identifies out-of-range pose values
#[test]
fn params_validation_pose_range() {
    proptest!(|(
        val in -4.0f32..4.0f32,
    )| {
        let params = FlameParams::builder()
            .jaw_rotation(val)
            .build();

        let is_valid = params.validate();
        let should_be_valid = val.abs() <= std::f32::consts::PI;

        assert_eq!(is_valid, should_be_valid,
            "Pose value {} should be {} but validation returned {}",
            val, should_be_valid, is_valid);
    });
}

/// Test that joint_pose returns zeros for out-of-bounds joint indices
#[test]
fn params_joint_pose_bounds() {
    proptest!(|(
        joint_idx in 0usize..10,
    )| {
        let params = FlameParams::neutral();
        let pose = params.joint_pose(joint_idx);

        // All joints should return valid data or zeros
        // Neutral params have all zeros
        assert_eq!(pose, [0.0, 0.0, 0.0]);
    });
}

// ---------------------------------------------------------------------------
// Builder Pattern Properties
// ---------------------------------------------------------------------------

/// Test that builder methods are composable in any order
#[test]
fn builder_composition_order_independence() {
    proptest!(|(
        shape_val in -2.0f32..2.0f32,
        expr_val in -1.0f32..1.0f32,
        jaw_val in -0.5f32..0.5f32,
        trans_x in -0.5f32..0.5f32,
    )| {
        // Build in one order
        let params1 = FlameParams::builder()
            .shape(vec![shape_val])
            .expression(vec![expr_val])
            .jaw_rotation(jaw_val)
            .translation([trans_x, 0.0, 0.0])
            .build();

        // Build in different order
        let params2 = FlameParams::builder()
            .translation([trans_x, 0.0, 0.0])
            .jaw_rotation(jaw_val)
            .expression(vec![expr_val])
            .shape(vec![shape_val])
            .build();

        // Results should be identical
        assert_eq!(params1.shape, params2.shape);
        assert_eq!(params1.expression, params2.expression);
        assert_eq!(params1.pose[6], params2.pose[6]);
        assert_eq!(params1.translation, params2.translation);
    });
}

/// Test that builder always produces valid pose vector size
#[test]
fn builder_pose_size_invariant() {
    proptest!(|(
        has_root in proptest::bool::ANY,
        has_jaw in proptest::bool::ANY,
        has_eyes in proptest::bool::ANY,
    )| {
        let mut builder = FlameParams::builder();

        if has_root {
            builder = builder.root_rotation([0.1, 0.0, 0.0]);
        }
        if has_jaw {
            builder = builder.jaw_rotation(0.1);
        }
        if has_eyes {
            builder = builder.left_eye_rotation([0.0, 0.1, 0.0]);
        }

        let params = builder.build();

        // Pose should always have exactly 15 values (5 joints * 3)
        assert_eq!(params.pose.len(), 15);
    });
}

// ---------------------------------------------------------------------------
// Mesh Properties (requires actual FLAME model data)
// ---------------------------------------------------------------------------

// Note: These tests would require loading actual FLAME model data,
// which we skip in property tests to keep them fast. Instead, we test
// the mathematical properties above that don't require model data.

#[cfg(test)]
mod mesh_properties {
    use super::*;

    /// Helper to check if FLAME model data exists
    fn has_flame_data() -> bool {
        std::path::Path::new("data/flame_model").exists()
    }

    /// Test that LBS forward pass preserves vertex count
    /// (Only runs if model data is available)
    #[test]
    fn lbs_preserves_vertex_count() {
        if !has_flame_data() {
            println!("Skipping LBS test: FLAME model data not found");
            return;
        }

        proptest!(|(
            shape_val in -2.0f32..2.0f32,
            expr_val in -1.0f32..1.0f32,
        )| {
            let model = FlameModel::load("data/flame_model")
                .expect("Should load FLAME model");

            let n_verts = model.v_template.nrows();

            let params = FlameParams::builder()
                .shape(vec![shape_val])
                .expression(vec![expr_val])
                .build();

            let mesh = model.forward(&params);

            // Vertex count should be preserved
            assert_eq!(mesh.vertices.len(), n_verts);
        });
    }

    /// Test that neutral parameters produce template mesh
    /// (Only runs if model data is available)
    #[test]
    fn neutral_params_produce_template() {
        if !has_flame_data() {
            println!("Skipping neutral params test: FLAME model data not found");
            return;
        }

        let model = FlameModel::load("data/flame_model").expect("Should load FLAME model");
        let params = FlameParams::neutral();
        let mesh = model.forward(&params);

        // With neutral parameters (zero shape, expression, pose),
        // vertices should be close to template
        for i in 0..mesh.vertices.len().min(10) {
            let v_template = [
                model.v_template[[i, 0]],
                model.v_template[[i, 1]],
                model.v_template[[i, 2]],
            ];

            for (j, &template_val) in v_template.iter().enumerate() {
                assert_relative_eq!(mesh.vertices[i][j], template_val, epsilon = 1e-4);
            }
        }
    }
}
