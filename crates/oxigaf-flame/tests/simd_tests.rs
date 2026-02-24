//! SIMD integration tests: numerical equivalence validation.
//!
//! These tests compare SIMD-accelerated operations against scalar implementations
//! to verify correctness and numerical stability.

#![cfg(all(feature = "simd", nightly))]

use approx::assert_relative_eq;
use nalgebra as na;
use ndarray::{Array2, Array3, ArrayView2};
use oxigaf_flame::simd::*;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

// ---------------------------------------------------------------------------
// Test constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-5;
const LARGE_N: usize = 1024;

// ---------------------------------------------------------------------------
// Scalar reference implementations
// ---------------------------------------------------------------------------

/// Scalar Rodrigues' rotation formula.
fn rodrigues_scalar(rx: f32, ry: f32, rz: f32) -> na::Matrix3<f32> {
    let angle_sq = rx * rx + ry * ry + rz * rz;
    if angle_sq < 1e-16 {
        return na::Matrix3::identity();
    }

    let angle = angle_sq.sqrt();
    let inv_angle = 1.0 / angle;
    let ax = rx * inv_angle;
    let ay = ry * inv_angle;
    let az = rz * inv_angle;

    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let t = 1.0 - cos_a;

    na::Matrix3::new(
        t * ax * ax + cos_a,
        t * ax * ay - sin_a * az,
        t * ax * az + sin_a * ay,
        t * ay * ax + sin_a * az,
        t * ay * ay + cos_a,
        t * ay * az - sin_a * ax,
        t * az * ax - sin_a * ay,
        t * az * ay + sin_a * ax,
        t * az * az + cos_a,
    )
}

/// Scalar blend shape application.
fn apply_blend_shapes_scalar(v: &mut Array2<f32>, dirs: &Array3<f32>, coeffs: &[f32]) {
    let n = v.nrows();
    let k = coeffs.len().min(dirs.shape()[2]);

    for (c_idx, &coeff) in coeffs.iter().enumerate().take(k) {
        if coeff.abs() <= 1e-12 {
            continue;
        }
        for i in 0..n {
            for j in 0..3 {
                v[[i, j]] += coeff * dirs[[i, j, c_idx]];
            }
        }
    }
}

/// Scalar LBS application.
fn apply_lbs_scalar(
    v_posed: &Array2<f32>,
    transforms: &[na::Matrix4<f32>],
    lbs_weights: &ArrayView2<f32>,
    translation: [f32; 3],
) -> Vec<na::Point3<f32>> {
    let n = v_posed.nrows();
    let nj = transforms.len();
    let [tx, ty, tz] = translation;

    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let mut blended = na::Matrix4::zeros();

        for j in 0..nj {
            let w = lbs_weights[[i, j]];
            if w.abs() > 1e-12 {
                blended += transforms[j] * w;
            }
        }

        let v = na::Vector4::new(v_posed[[i, 0]], v_posed[[i, 1]], v_posed[[i, 2]], 1.0);
        let r = blended * v;
        out.push(na::Point3::new(r[0] + tx, r[1] + ty, r[2] + tz));
    }

    out
}

// ---------------------------------------------------------------------------
// Rodrigues rotation tests
// ---------------------------------------------------------------------------

#[test]
fn test_rodrigues_simd_vs_scalar_identity() {
    let simd = rodrigues_simd(0.0, 0.0, 0.0);
    let scalar = rodrigues_scalar(0.0, 0.0, 0.0);

    assert_relative_eq!(simd, scalar, epsilon = EPSILON);
}

#[test]
fn test_rodrigues_simd_vs_scalar_axis_aligned() {
    let test_cases = vec![
        (FRAC_PI_2, 0.0, 0.0), // 90° around X
        (0.0, FRAC_PI_2, 0.0), // 90° around Y
        (0.0, 0.0, FRAC_PI_2), // 90° around Z
        (PI, 0.0, 0.0),        // 180° around X
        (0.0, PI, 0.0),        // 180° around Y
        (0.0, 0.0, PI),        // 180° around Z
    ];

    for (rx, ry, rz) in test_cases {
        let simd = rodrigues_simd(rx, ry, rz);
        let scalar = rodrigues_scalar(rx, ry, rz);

        assert_relative_eq!(simd, scalar, epsilon = EPSILON);
    }
}

#[test]
fn test_rodrigues_simd_vs_scalar_arbitrary() {
    let test_cases = vec![
        (0.1, 0.2, 0.3),
        (1.5, -0.5, 0.8),
        (-0.3, 0.7, -1.2),
        (FRAC_PI_4, FRAC_PI_4, FRAC_PI_4),
    ];

    for (rx, ry, rz) in test_cases {
        let simd = rodrigues_simd(rx, ry, rz);
        let scalar = rodrigues_scalar(rx, ry, rz);

        assert_relative_eq!(simd, scalar, epsilon = EPSILON);
    }
}

#[test]
fn test_rodrigues_batch_equivalence() {
    let rotations = vec![
        [0.1, 0.2, 0.3],
        [FRAC_PI_2, 0.0, 0.0],
        [0.0, FRAC_PI_2, 0.0],
        [0.0, 0.0, FRAC_PI_2],
        [-0.5, 0.8, -1.2],
    ];

    let batch_result = rodrigues_batch(&rotations);

    for (i, &[rx, ry, rz]) in rotations.iter().enumerate() {
        let scalar = rodrigues_scalar(rx, ry, rz);
        assert_relative_eq!(batch_result[i], scalar, epsilon = EPSILON);
    }
}

// ---------------------------------------------------------------------------
// Matrix operations tests
// ---------------------------------------------------------------------------

#[test]
fn test_mat4_mul_simd_vs_scalar() {
    let a = na::Matrix4::new(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    );
    let b = na::Matrix4::new(
        16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    );

    let simd = mat4_mul_simd(&a, &b);
    let scalar = a * b;

    assert_relative_eq!(simd, scalar, epsilon = EPSILON);
}

#[test]
fn test_mat4_vec4_mul_simd_vs_scalar() {
    let m = na::Matrix4::new(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 1.0,
    );
    let v = na::Vector4::new(1.0, 2.0, 3.0, 1.0);

    let simd = mat4_vec4_mul_simd(&m, &v);
    let scalar = m * v;

    assert_relative_eq!(simd, scalar, epsilon = EPSILON);
}

#[test]
fn test_weighted_matrix_sum_simd_vs_scalar() {
    let matrices = vec![
        na::Matrix4::identity(),
        na::Matrix4::new(
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0,
        ),
        na::Matrix4::new(
            2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ),
    ];
    let weights = vec![0.5, 0.3, 0.2];

    let simd = weighted_matrix_sum_simd(&matrices, &weights);

    // Scalar computation
    let mut scalar = na::Matrix4::zeros();
    for (m, &w) in matrices.iter().zip(weights.iter()) {
        scalar += m * w;
    }

    assert_relative_eq!(simd, scalar, epsilon = EPSILON);
}

// ---------------------------------------------------------------------------
// Blend shapes tests
// ---------------------------------------------------------------------------

#[test]
fn test_blend_shapes_simd_vs_scalar_small() {
    let n = 16;
    let k = 5;

    let v_original = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f32);
    let dirs = Array3::from_shape_fn((n, 3, k), |(i, j, c)| ((i + j + c) as f32) * 0.1);
    let coeffs = vec![0.5, -0.3, 0.2, 1.0, -0.5];

    let mut v_simd = v_original.clone();
    let mut v_scalar = v_original;

    apply_blend_shapes_simd(&mut v_simd, &dirs, &coeffs);
    apply_blend_shapes_scalar(&mut v_scalar, &dirs, &coeffs);

    for i in 0..n {
        for j in 0..3 {
            assert_relative_eq!(v_simd[[i, j]], v_scalar[[i, j]], epsilon = EPSILON);
        }
    }
}

#[test]
fn test_blend_shapes_simd_vs_scalar_large() {
    let n = LARGE_N;
    let k = 10;

    let v_original = Array2::from_shape_fn((n, 3), |(i, j)| i as f32 * 0.1 + j as f32 * 0.01);
    let dirs = Array3::from_shape_fn((n, 3, k), |(i, j, c)| ((i + j + c) as f32 * 0.01).sin());
    let coeffs: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();

    let mut v_simd = v_original.clone();
    let mut v_scalar = v_original;

    apply_blend_shapes_simd(&mut v_simd, &dirs, &coeffs);
    apply_blend_shapes_scalar(&mut v_scalar, &dirs, &coeffs);

    for i in 0..n {
        for j in 0..3 {
            assert_relative_eq!(
                v_simd[[i, j]],
                v_scalar[[i, j]],
                epsilon = EPSILON,
                max_relative = 1e-4
            );
        }
    }
}

// ---------------------------------------------------------------------------
// LBS skinning tests
// ---------------------------------------------------------------------------

#[test]
fn test_lbs_simd_vs_scalar_simple() {
    let n = 8;
    let nj = 4;

    let v_posed = Array2::from_shape_fn((n, 3), |(i, j)| i as f32 + j as f32 * 0.1);

    let transforms: Vec<na::Matrix4<f32>> = (0..nj)
        .map(|i| {
            na::Matrix4::new(
                1.0,
                0.0,
                0.0,
                i as f32 * 0.1,
                0.0,
                1.0,
                0.0,
                i as f32 * 0.2,
                0.0,
                0.0,
                1.0,
                i as f32 * 0.3,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        })
        .collect();

    let lbs_weights = Array2::from_shape_fn((n, nj), |(i, j)| {
        let w = (i + j) as f32 * 0.1;
        w / ((0..nj).map(|k| (i + k) as f32 * 0.1).sum::<f32>() + 1e-6)
    });

    let translation = [0.1, 0.2, 0.3];

    let lbs_view = lbs_weights.view();
    let result_simd = apply_lbs_simd(&v_posed, &transforms, &lbs_view, translation);
    let result_scalar = apply_lbs_scalar(&v_posed, &transforms, &lbs_view, translation);

    assert_eq!(result_simd.len(), result_scalar.len());

    for (s, sc) in result_simd.iter().zip(result_scalar.iter()) {
        assert_relative_eq!(s.x, sc.x, epsilon = EPSILON, max_relative = 1e-4);
        assert_relative_eq!(s.y, sc.y, epsilon = EPSILON, max_relative = 1e-4);
        assert_relative_eq!(s.z, sc.z, epsilon = EPSILON, max_relative = 1e-4);
    }
}

// ---------------------------------------------------------------------------
// SoA layout tests
// ---------------------------------------------------------------------------

#[test]
fn test_vertices_soa_transform_vs_aos() {
    let vertices = vec![
        na::Point3::new(1.0, 2.0, 3.0),
        na::Point3::new(4.0, 5.0, 6.0),
        na::Point3::new(7.0, 8.0, 9.0),
        na::Point3::new(10.0, 11.0, 12.0),
        na::Point3::new(13.0, 14.0, 15.0),
        na::Point3::new(16.0, 17.0, 18.0),
        na::Point3::new(19.0, 20.0, 21.0),
        na::Point3::new(22.0, 23.0, 24.0),
    ];

    let transform = na::Matrix4::new(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 1.0,
    );

    // SIMD via SoA
    let mut soa = VerticesSoA::from_aos(&vertices);
    soa.transform_simd(&transform);
    let result_simd = soa.to_aos();

    // Scalar via AoS
    let mut result_scalar = Vec::with_capacity(vertices.len());
    for v in &vertices {
        let v4 = na::Vector4::new(v.x, v.y, v.z, 1.0);
        let transformed = transform * v4;
        result_scalar.push(na::Point3::new(
            transformed[0],
            transformed[1],
            transformed[2],
        ));
    }

    assert_eq!(result_simd.len(), result_scalar.len());

    for (s, sc) in result_simd.iter().zip(result_scalar.iter()) {
        assert_relative_eq!(s.x, sc.x, epsilon = EPSILON, max_relative = 1e-4);
        assert_relative_eq!(s.y, sc.y, epsilon = EPSILON, max_relative = 1e-4);
        assert_relative_eq!(s.z, sc.z, epsilon = EPSILON, max_relative = 1e-4);
    }
}

// ---------------------------------------------------------------------------
// Normal computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_cross_product_simd_correctness() {
    let v0 = [0.0, 0.0, 0.0];
    let v1 = [1.0, 0.0, 0.0];
    let v2 = [0.0, 1.0, 0.0];

    let result = cross_product_simd(&v0, &v1, &v2);

    // Expected: (1,0,0) × (0,1,0) = (0,0,1)
    assert_relative_eq!(result[0], 0.0, epsilon = EPSILON);
    assert_relative_eq!(result[1], 0.0, epsilon = EPSILON);
    assert_relative_eq!(result[2], 1.0, epsilon = EPSILON);
}

#[test]
fn test_normalize_vectors_simd_correctness() {
    let mut vectors = vec![
        [3.0, 4.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 7.0],
        [1.0, 1.0, 1.0],
    ];

    normalize_vectors_simd(&mut vectors);

    // Check all are unit vectors
    for v in vectors.iter() {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert_relative_eq!(len, 1.0, epsilon = EPSILON, max_relative = 1e-4);
    }

    // Check specific results
    assert_relative_eq!(vectors[0][0], 0.6, epsilon = EPSILON);
    assert_relative_eq!(vectors[0][1], 0.8, epsilon = EPSILON);
    assert_relative_eq!(vectors[0][2], 0.0, epsilon = EPSILON);

    assert_relative_eq!(vectors[1][0], 1.0, epsilon = EPSILON);
    assert_relative_eq!(vectors[1][1], 0.0, epsilon = EPSILON);
    assert_relative_eq!(vectors[1][2], 0.0, epsilon = EPSILON);
}

// ---------------------------------------------------------------------------
// Full forward pass simulation
// ---------------------------------------------------------------------------

#[test]
fn test_full_forward_pass_simd_vs_scalar() {
    // Simulate a simplified FLAME forward pass

    // 1. Blend shapes
    let n = 128;
    let k = 5;

    let v_template = Array2::from_shape_fn((n, 3), |(i, j)| i as f32 * 0.1 + j as f32 * 0.01);
    let blend_dirs =
        Array3::from_shape_fn((n, 3, k), |(i, j, c)| ((i + j + c) as f32 * 0.01).sin());
    let blend_coeffs = vec![0.3, -0.2, 0.5, 0.1, -0.4];

    let mut v_simd = v_template.clone();
    let mut v_scalar = v_template;

    apply_blend_shapes_simd(&mut v_simd, &blend_dirs, &blend_coeffs);
    apply_blend_shapes_scalar(&mut v_scalar, &blend_dirs, &blend_coeffs);

    // 2. Pose (Rodrigues)
    let rotations = vec![[0.1, 0.2, 0.3], [0.0, FRAC_PI_4, 0.0], [-0.1, 0.0, 0.5]];

    let rots_simd = rodrigues_batch(&rotations);
    let _rots_scalar: Vec<na::Matrix3<f32>> = rotations
        .iter()
        .map(|&[rx, ry, rz]| rodrigues_scalar(rx, ry, rz))
        .collect();

    // 3. LBS
    let nj = 3;
    let transforms: Vec<na::Matrix4<f32>> = (0..nj)
        .map(|i| {
            let rot = rots_simd[i];
            na::Matrix4::new(
                rot[(0, 0)],
                rot[(0, 1)],
                rot[(0, 2)],
                0.1 * i as f32,
                rot[(1, 0)],
                rot[(1, 1)],
                rot[(1, 2)],
                0.2 * i as f32,
                rot[(2, 0)],
                rot[(2, 1)],
                rot[(2, 2)],
                0.3 * i as f32,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        })
        .collect();

    let lbs_weights = Array2::from_shape_fn((n, nj), |(i, j)| {
        let w = (i + j) as f32 * 0.1;
        w / ((0..nj).map(|k| (i + k) as f32 * 0.1).sum::<f32>() + 1e-6)
    });

    let translation = [0.0, 0.0, 0.0];

    let lbs_view = lbs_weights.view();
    let final_simd = apply_lbs_simd(&v_simd, &transforms, &lbs_view, translation);
    let final_scalar = apply_lbs_scalar(&v_scalar, &transforms, &lbs_view, translation);

    // 4. Verify full forward pass equivalence
    assert_eq!(final_simd.len(), final_scalar.len());

    for (s, sc) in final_simd.iter().zip(final_scalar.iter()) {
        // Slightly larger epsilon for accumulated error
        assert_relative_eq!(s.x, sc.x, epsilon = EPSILON * 10.0, max_relative = 1e-3);
        assert_relative_eq!(s.y, sc.y, epsilon = EPSILON * 10.0, max_relative = 1e-3);
        assert_relative_eq!(s.z, sc.z, epsilon = EPSILON * 10.0, max_relative = 1e-3);
    }
}
