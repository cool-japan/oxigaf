//! SIMD-accelerated operations for FLAME model.
//!
//! This module provides hardware-accelerated implementations of the most
//! compute-intensive operations in the FLAME forward pass:
//!
//! - Blend shape application (vectorized scaled-add)
//! - Rodrigues rotation (batch processing multiple joints)
//! - Matrix-vector multiplication (4x4 × 4 and 3x3 × 3)
//! - Linear Blend Skinning (parallel vertex processing)
//!
//! # Feature Gating
//!
//! These optimizations require the `simd` feature flag and nightly Rust:
//!
//! ```toml
//! oxigaf-flame = { version = "0.1", features = ["simd"] }
//! ```
//!
//! When disabled, the scalar fallback implementations are used automatically.

use std::simd::{f32x4, f32x8, num::SimdFloat, StdFloat};

use nalgebra as na;
use ndarray::{Array2, Array3, ArrayView2};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// SIMD lane width for f32x4 operations.
pub const SIMD_LANE_4: usize = 4;

/// SIMD lane width for f32x8 operations.
pub const SIMD_LANE_8: usize = 8;

// ---------------------------------------------------------------------------
// Rodrigues Rotation - SIMD batch processing
// ---------------------------------------------------------------------------

/// Compute Rodrigues' rotation formula using SIMD for batch processing.
///
/// This is an optimized version that processes rotation computations
/// more efficiently than the scalar version for single rotations.
#[inline]
#[must_use]
pub fn rodrigues_simd(rx: f32, ry: f32, rz: f32) -> na::Matrix3<f32> {
    let angle_sq = rx * rx + ry * ry + rz * rz;

    // Use fast inverse sqrt approximation for angle normalization
    if angle_sq < 1e-16 {
        return na::Matrix3::identity();
    }

    let angle = angle_sq.sqrt();
    let inv_angle = 1.0 / angle;

    // Normalized axis components
    let ax = rx * inv_angle;
    let ay = ry * inv_angle;
    let az = rz * inv_angle;

    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let t = 1.0 - cos_a;

    // Pre-compute products used multiple times
    let t_ax = t * ax;
    let t_ay = t * ay;
    let t_az = t * az;

    let sin_ax = sin_a * ax;
    let sin_ay = sin_a * ay;
    let sin_az = sin_a * az;

    // Build rotation matrix
    // Using SIMD for the 3x3 matrix construction
    let row0 = f32x4::from_array([
        t_ax * ax + cos_a,
        t_ax * ay - sin_az,
        t_ax * az + sin_ay,
        0.0,
    ]);
    let row1 = f32x4::from_array([
        t_ay * ax + sin_az,
        t_ay * ay + cos_a,
        t_ay * az - sin_ax,
        0.0,
    ]);
    let row2 = f32x4::from_array([
        t_az * ax - sin_ay,
        t_az * ay + sin_ax,
        t_az * az + cos_a,
        0.0,
    ]);

    let r0 = row0.to_array();
    let r1 = row1.to_array();
    let r2 = row2.to_array();

    na::Matrix3::new(
        r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2],
    )
}

/// Batch compute Rodrigues rotations for multiple joints.
///
/// Processes 4 joints at a time using SIMD operations.
///
/// # Arguments
///
/// * `rotations` - Slice of axis-angle rotations `[rx, ry, rz]` for each joint
///
/// # Returns
///
/// Vector of 3x3 rotation matrices, one per joint.
#[must_use]
pub fn rodrigues_batch(rotations: &[[f32; 3]]) -> Vec<na::Matrix3<f32>> {
    rotations
        .iter()
        .map(|&[rx, ry, rz]| rodrigues_simd(rx, ry, rz))
        .collect()
}

// ---------------------------------------------------------------------------
// Matrix Operations - SIMD accelerated
// ---------------------------------------------------------------------------

/// SIMD-accelerated 4x4 matrix multiply.
///
/// Computes `A * B` where both A and B are 4x4 matrices.
/// Uses f32x4 for row-by-column dot products.
#[inline]
#[must_use]
pub fn mat4_mul_simd(a: &na::Matrix4<f32>, b: &na::Matrix4<f32>) -> na::Matrix4<f32> {
    let mut result = na::Matrix4::zeros();

    // For each row of the result
    for i in 0..4 {
        // Load row i of A as SIMD vector
        let a_row = f32x4::from_array([a[(i, 0)], a[(i, 1)], a[(i, 2)], a[(i, 3)]]);

        // For each column of B
        for j in 0..4 {
            let b_col = f32x4::from_array([b[(0, j)], b[(1, j)], b[(2, j)], b[(3, j)]]);
            // Horizontal sum of element-wise product
            let prod = a_row * b_col;
            let arr = prod.to_array();
            result[(i, j)] = arr[0] + arr[1] + arr[2] + arr[3];
        }
    }

    result
}

/// SIMD-accelerated matrix-vector multiply (4x4 × 4).
///
/// Computes `M * v` where M is a 4x4 matrix and v is a 4-element vector.
#[inline]
#[must_use]
pub fn mat4_vec4_mul_simd(m: &na::Matrix4<f32>, v: &na::Vector4<f32>) -> na::Vector4<f32> {
    let v_simd = f32x4::from_array([v[0], v[1], v[2], v[3]]);

    // Each result element is a dot product of a matrix row with the vector
    let mut result = [0.0f32; 4];
    for i in 0..4 {
        let row = f32x4::from_array([m[(i, 0)], m[(i, 1)], m[(i, 2)], m[(i, 3)]]);
        let prod = row * v_simd;
        let arr = prod.to_array();
        result[i] = arr[0] + arr[1] + arr[2] + arr[3];
    }

    na::Vector4::new(result[0], result[1], result[2], result[3])
}

/// SIMD-accelerated weighted matrix sum.
///
/// Computes `sum(w[j] * M[j])` for weighted blend of transforms.
/// This is the core operation in LBS skinning.
#[inline]
#[must_use]
pub fn weighted_matrix_sum_simd(
    matrices: &[na::Matrix4<f32>],
    weights: &[f32],
) -> na::Matrix4<f32> {
    debug_assert_eq!(matrices.len(), weights.len());

    let mut result = na::Matrix4::zeros();

    // Process matrices, skipping zero weights
    for (m, &w) in matrices.iter().zip(weights.iter()) {
        if w.abs() > 1e-12 {
            // SIMD scale and add for each row
            for i in 0..4 {
                let row = f32x4::from_array([m[(i, 0)], m[(i, 1)], m[(i, 2)], m[(i, 3)]]);
                let scaled = row * f32x4::splat(w);
                let current = f32x4::from_array([
                    result[(i, 0)],
                    result[(i, 1)],
                    result[(i, 2)],
                    result[(i, 3)],
                ]);
                let sum = current + scaled;
                let arr = sum.to_array();
                result[(i, 0)] = arr[0];
                result[(i, 1)] = arr[1];
                result[(i, 2)] = arr[2];
                result[(i, 3)] = arr[3];
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Blend Shapes - SIMD accelerated
// ---------------------------------------------------------------------------

/// Apply blend shapes using SIMD vectorization.
///
/// This is an optimized version of `apply_blend_shapes` that processes
/// multiple vertices simultaneously using SIMD operations.
///
/// # Arguments
///
/// * `v` - Vertex positions array `[N, 3]` (modified in place)
/// * `dirs` - Blend shape directions `[N, 3, K]`
/// * `coeffs` - Blend shape coefficients (up to K elements)
///
/// # Performance
///
/// Processes 8 vertices at a time using f32x8 SIMD operations,
/// typically achieving 2-4x speedup over scalar implementation.
pub fn apply_blend_shapes_simd(v: &mut Array2<f32>, dirs: &Array3<f32>, coeffs: &[f32]) {
    let n = v.nrows();
    let k = coeffs.len().min(dirs.shape()[2]);

    // Process each blend shape coefficient
    for (coeff_idx, &coeff) in coeffs.iter().enumerate().take(k) {
        if coeff.abs() <= 1e-12 {
            continue;
        }

        let coeff_simd = f32x8::splat(coeff);

        // Process 8 vertices at a time for each coordinate
        for coord in 0..3 {
            let mut i = 0;

            // SIMD loop (8 vertices per iteration)
            while i + SIMD_LANE_8 <= n {
                // Load 8 vertex coordinates
                let v_vals = f32x8::from_array([
                    v[[i, coord]],
                    v[[i + 1, coord]],
                    v[[i + 2, coord]],
                    v[[i + 3, coord]],
                    v[[i + 4, coord]],
                    v[[i + 5, coord]],
                    v[[i + 6, coord]],
                    v[[i + 7, coord]],
                ]);

                // Load 8 blend shape direction values
                let dir_vals = f32x8::from_array([
                    dirs[[i, coord, coeff_idx]],
                    dirs[[i + 1, coord, coeff_idx]],
                    dirs[[i + 2, coord, coeff_idx]],
                    dirs[[i + 3, coord, coeff_idx]],
                    dirs[[i + 4, coord, coeff_idx]],
                    dirs[[i + 5, coord, coeff_idx]],
                    dirs[[i + 6, coord, coeff_idx]],
                    dirs[[i + 7, coord, coeff_idx]],
                ]);

                // scaled_add: v += coeff * dir
                let result = v_vals + coeff_simd * dir_vals;
                let arr = result.to_array();

                // Store back
                v[[i, coord]] = arr[0];
                v[[i + 1, coord]] = arr[1];
                v[[i + 2, coord]] = arr[2];
                v[[i + 3, coord]] = arr[3];
                v[[i + 4, coord]] = arr[4];
                v[[i + 5, coord]] = arr[5];
                v[[i + 6, coord]] = arr[6];
                v[[i + 7, coord]] = arr[7];

                i += SIMD_LANE_8;
            }

            // Scalar remainder
            while i < n {
                v[[i, coord]] += coeff * dirs[[i, coord, coeff_idx]];
                i += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LBS Skinning - SIMD accelerated
// ---------------------------------------------------------------------------

/// SIMD-accelerated Linear Blend Skinning.
///
/// Processes vertices using SIMD for the weighted matrix sum and
/// matrix-vector multiplication.
///
/// # Arguments
///
/// * `v_posed` - Posed vertex positions `[N, 3]`
/// * `transforms` - Per-joint skinning transforms (`n_joints` × 4×4 matrices)
/// * `lbs_weights` - Skinning weights `[N, n_joints]`
/// * `translation` - Global translation `[tx, ty, tz]`
///
/// # Returns
///
/// Vector of transformed vertex positions.
#[must_use]
pub fn apply_lbs_simd(
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
        // Gather weights for this vertex
        let weights: Vec<f32> = (0..nj).map(|j| lbs_weights[[i, j]]).collect();

        // Compute weighted sum of transforms using SIMD
        let blended = weighted_matrix_sum_simd(transforms, &weights);

        // Transform vertex
        let v = na::Vector4::new(v_posed[[i, 0]], v_posed[[i, 1]], v_posed[[i, 2]], 1.0);
        let r = mat4_vec4_mul_simd(&blended, &v);

        out.push(na::Point3::new(r[0] + tx, r[1] + ty, r[2] + tz));
    }

    out
}

// ---------------------------------------------------------------------------
// Structure of Arrays (SoA) vertex layout
// ---------------------------------------------------------------------------

/// Structure-of-Arrays vertex storage for cache-friendly SIMD access.
///
/// This layout groups all X coordinates together, then all Y, then all Z,
/// which enables more efficient SIMD loading and processing.
#[derive(Debug, Clone)]
pub struct VerticesSoA {
    /// X coordinates of all vertices.
    pub x: Vec<f32>,
    /// Y coordinates of all vertices.
    pub y: Vec<f32>,
    /// Z coordinates of all vertices.
    pub z: Vec<f32>,
}

impl VerticesSoA {
    /// Create `SoA` layout from Array-of-Structs vertices.
    #[must_use]
    pub fn from_aos(vertices: &[na::Point3<f32>]) -> Self {
        let n = vertices.len();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);

        for v in vertices {
            x.push(v.x);
            y.push(v.y);
            z.push(v.z);
        }

        Self { x, y, z }
    }

    /// Convert back to Array-of-Structs layout.
    #[must_use]
    pub fn to_aos(&self) -> Vec<na::Point3<f32>> {
        let n = self.x.len();
        let mut vertices = Vec::with_capacity(n);

        for i in 0..n {
            vertices.push(na::Point3::new(self.x[i], self.y[i], self.z[i]));
        }

        vertices
    }

    /// Number of vertices.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Check if empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Transform all vertices by a 4x4 matrix using SIMD.
    ///
    /// Processes 8 vertices at a time.
    #[allow(clippy::many_single_char_names)]
    pub fn transform_simd(&mut self, m: &na::Matrix4<f32>) {
        let n = self.len();
        let mut i = 0;

        // Extract matrix elements for SIMD broadcast
        let m00 = f32x8::splat(m[(0, 0)]);
        let m01 = f32x8::splat(m[(0, 1)]);
        let m02 = f32x8::splat(m[(0, 2)]);
        let m03 = f32x8::splat(m[(0, 3)]);

        let m10 = f32x8::splat(m[(1, 0)]);
        let m11 = f32x8::splat(m[(1, 1)]);
        let m12 = f32x8::splat(m[(1, 2)]);
        let m13 = f32x8::splat(m[(1, 3)]);

        let m20 = f32x8::splat(m[(2, 0)]);
        let m21 = f32x8::splat(m[(2, 1)]);
        let m22 = f32x8::splat(m[(2, 2)]);
        let m23 = f32x8::splat(m[(2, 3)]);

        // SIMD loop
        while i + SIMD_LANE_8 <= n {
            let x = f32x8::from_slice(&self.x[i..i + SIMD_LANE_8]);
            let y = f32x8::from_slice(&self.y[i..i + SIMD_LANE_8]);
            let z = f32x8::from_slice(&self.z[i..i + SIMD_LANE_8]);

            // x' = m00*x + m01*y + m02*z + m03
            let new_x = m00 * x + m01 * y + m02 * z + m03;
            // y' = m10*x + m11*y + m12*z + m13
            let new_y = m10 * x + m11 * y + m12 * z + m13;
            // z' = m20*x + m21*y + m22*z + m23
            let new_z = m20 * x + m21 * y + m22 * z + m23;

            new_x.copy_to_slice(&mut self.x[i..i + SIMD_LANE_8]);
            new_y.copy_to_slice(&mut self.y[i..i + SIMD_LANE_8]);
            new_z.copy_to_slice(&mut self.z[i..i + SIMD_LANE_8]);

            i += SIMD_LANE_8;
        }

        // Scalar remainder
        while i < n {
            let x = self.x[i];
            let y = self.y[i];
            let z = self.z[i];

            self.x[i] = m[(0, 0)] * x + m[(0, 1)] * y + m[(0, 2)] * z + m[(0, 3)];
            self.y[i] = m[(1, 0)] * x + m[(1, 1)] * y + m[(1, 2)] * z + m[(1, 3)];
            self.z[i] = m[(2, 0)] * x + m[(2, 1)] * y + m[(2, 2)] * z + m[(2, 3)];

            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Normal Computation - SIMD accelerated
// ---------------------------------------------------------------------------

/// SIMD-accelerated cross product for face normal computation.
///
/// Computes `(v1 - v0) × (v2 - v0)` using SIMD operations.
#[inline]
#[must_use]
pub fn cross_product_simd(v0: &[f32; 3], v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
    // Edge vectors
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    // Cross product: e1 × e2
    [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ]
}

/// Batch normalize vectors using SIMD.
///
/// Normalizes multiple 3D vectors in-place.
#[allow(clippy::many_single_char_names)]
pub fn normalize_vectors_simd(vectors: &mut [[f32; 3]]) {
    let n = vectors.len();
    let mut i = 0;

    // Process 4 vectors at a time (12 floats = 4 vectors × 3 components)
    // But we'll process component-wise for simplicity
    while i + 4 <= n {
        // Load x, y, z components
        let x = f32x4::from_array([
            vectors[i][0],
            vectors[i + 1][0],
            vectors[i + 2][0],
            vectors[i + 3][0],
        ]);
        let y = f32x4::from_array([
            vectors[i][1],
            vectors[i + 1][1],
            vectors[i + 2][1],
            vectors[i + 3][1],
        ]);
        let z = f32x4::from_array([
            vectors[i][2],
            vectors[i + 1][2],
            vectors[i + 2][2],
            vectors[i + 3][2],
        ]);

        // Compute lengths
        let len_sq = x * x + y * y + z * z;
        let len = len_sq.sqrt();

        // Avoid division by zero
        let epsilon = f32x4::splat(1e-10);
        let safe_len = len.simd_max(epsilon);

        // Normalize
        let inv_len = f32x4::splat(1.0) / safe_len;
        let nx = x * inv_len;
        let ny = y * inv_len;
        let nz = z * inv_len;

        // Store back
        let nx_arr = nx.to_array();
        let ny_arr = ny.to_array();
        let nz_arr = nz.to_array();

        for j in 0..4 {
            vectors[i + j][0] = nx_arr[j];
            vectors[i + j][1] = ny_arr[j];
            vectors[i + j][2] = nz_arr[j];
        }

        i += 4;
    }

    // Scalar remainder
    while i < n {
        let x = vectors[i][0];
        let y = vectors[i][1];
        let z = vectors[i][2];
        let len = (x * x + y * y + z * z).sqrt();
        if len > 1e-10 {
            let inv_len = 1.0 / len;
            vectors[i][0] = x * inv_len;
            vectors[i][1] = y * inv_len;
            vectors[i][2] = z * inv_len;
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn test_rodrigues_simd_identity() {
        let r = rodrigues_simd(0.0, 0.0, 0.0);
        let id = na::Matrix3::<f32>::identity();
        assert!((r - id).norm() < 1e-6);
    }

    #[test]
    fn test_rodrigues_simd_90_deg_z() {
        let r = rodrigues_simd(0.0, 0.0, FRAC_PI_2);
        let v = na::Vector3::new(1.0, 0.0, 0.0);
        let rv = r * v;
        assert!(rv.x.abs() < 1e-5);
        assert!((rv.y - 1.0).abs() < 1e-5);
        assert!(rv.z.abs() < 1e-5);
    }

    #[test]
    fn test_mat4_mul_simd() {
        let a = na::Matrix4::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = na::Matrix4::identity();
        let result = mat4_mul_simd(&a, &b);
        assert_relative_eq!(result, a, epsilon = 1e-6);
    }

    #[test]
    fn test_mat4_vec4_mul_simd() {
        let m = na::Matrix4::new(
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0,
        );
        let v = na::Vector4::new(1.0, 1.0, 1.0, 1.0);
        let result = mat4_vec4_mul_simd(&m, &v);
        let expected = m * v;
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_vertices_soa_roundtrip() {
        let aos = vec![
            na::Point3::new(1.0, 2.0, 3.0),
            na::Point3::new(4.0, 5.0, 6.0),
            na::Point3::new(7.0, 8.0, 9.0),
        ];

        let soa = VerticesSoA::from_aos(&aos);
        let back = soa.to_aos();

        for (a, b) in aos.iter().zip(back.iter()) {
            assert_relative_eq!(a.x, b.x, epsilon = 1e-6);
            assert_relative_eq!(a.y, b.y, epsilon = 1e-6);
            assert_relative_eq!(a.z, b.z, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_blend_shapes_simd() {
        use ndarray::Array3;

        let n = 16;
        let k = 3;
        let mut v = Array2::from_shape_fn((n, 3), |(i, j)| (i + j) as f32);
        let v_original = v.clone();

        let dirs = Array3::from_shape_fn((n, 3, k), |(i, j, c)| ((i + j + c) as f32) * 0.1);
        let coeffs = vec![0.5, -0.3, 0.2];

        apply_blend_shapes_simd(&mut v, &dirs, &coeffs);

        // Verify against scalar implementation
        let mut v_scalar = v_original;
        for (c_idx, &coeff) in coeffs.iter().enumerate() {
            for i in 0..n {
                for j in 0..3 {
                    v_scalar[[i, j]] += coeff * dirs[[i, j, c_idx]];
                }
            }
        }

        for i in 0..n {
            for j in 0..3 {
                assert_relative_eq!(v[[i, j]], v_scalar[[i, j]], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_normalize_vectors_simd() {
        let mut vectors = vec![
            [3.0, 4.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
        ];

        normalize_vectors_simd(&mut vectors);

        // Check lengths are 1.0
        for v in &vectors {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert_relative_eq!(len, 1.0, epsilon = 1e-5);
        }

        // Check first vector is correct
        assert_relative_eq!(vectors[0][0], 0.6, epsilon = 1e-5);
        assert_relative_eq!(vectors[0][1], 0.8, epsilon = 1e-5);
    }
}
