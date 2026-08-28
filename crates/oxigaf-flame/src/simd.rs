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

use std::simd::{
    cmp::SimdPartialOrd, f32x4, f32x8, num::SimdFloat, simd_swizzle, Select, StdFloat,
};

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

/// Compute Rodrigues' rotation formula for a single axis-angle rotation.
///
/// This is a plain scalar implementation: a single 3x3 matrix built from
/// nine independent scalar expressions has no shared lane structure to
/// vectorize (packing the nine results into `f32x4` registers just to
/// immediately unpack them again is pure overhead, not real SIMD work).
/// The genuine SIMD speedup lives in [`rodrigues_batch`], which computes
/// four joints' worth of this formula in parallel lanes. This function
/// remains the scalar building block: it's used directly for one-off
/// rotations and as the scalar remainder inside `rodrigues_batch`.
#[inline]
#[must_use]
pub fn rodrigues_simd(rx: f32, ry: f32, rz: f32) -> na::Matrix3<f32> {
    let angle_sq = rx * rx + ry * ry + rz * rz;

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

    na::Matrix3::new(
        t_ax * ax + cos_a,
        t_ax * ay - sin_az,
        t_ax * az + sin_ay,
        t_ay * ax + sin_az,
        t_ay * ay + cos_a,
        t_ay * az - sin_ax,
        t_az * ax - sin_ay,
        t_az * ay + sin_ax,
        t_az * az + cos_a,
    )
}

/// Batch compute Rodrigues rotations for multiple joints.
///
/// Processes 4 joints at a time in genuine SIMD lanes: each joint's
/// `rx`/`ry`/`rz` occupies one lane of three `f32x4` vectors, and `angle`,
/// its normalization, and all nine matrix entries are computed for all 4
/// joints simultaneously via lane-wise arithmetic (including a masked
/// `select` for joints whose rotation angle is ~0, which fall back to the
/// identity matrix). Only `sin`/`cos` themselves are evaluated per-lane via
/// [`f32::sin`]/[`f32::cos`] and repacked -- `std::simd`'s vectorized trig
/// intrinsics are not exercised anywhere else in this crate, so this
/// avoids depending on them, while every surrounding computation (the bulk
/// of the arithmetic) still runs in real 4-wide SIMD.
///
/// # Arguments
///
/// * `rotations` - Slice of axis-angle rotations `[rx, ry, rz]` for each joint
///
/// # Returns
///
/// Vector of 3x3 rotation matrices, one per joint, in input order.
#[must_use]
pub fn rodrigues_batch(rotations: &[[f32; 3]]) -> Vec<na::Matrix3<f32>> {
    let n = rotations.len();
    let mut out = Vec::with_capacity(n);
    let mut i = 0;

    while i + SIMD_LANE_4 <= n {
        let rx = f32x4::from_array([
            rotations[i][0],
            rotations[i + 1][0],
            rotations[i + 2][0],
            rotations[i + 3][0],
        ]);
        let ry = f32x4::from_array([
            rotations[i][1],
            rotations[i + 1][1],
            rotations[i + 2][1],
            rotations[i + 3][1],
        ]);
        let rz = f32x4::from_array([
            rotations[i][2],
            rotations[i + 1][2],
            rotations[i + 2][2],
            rotations[i + 3][2],
        ]);

        let angle_sq = rx * rx + ry * ry + rz * rz;
        let degenerate = angle_sq.simd_lt(f32x4::splat(1e-16));

        // Clamp away from zero before dividing so degenerate lanes never
        // produce NaN/Inf; their result is discarded by `select` below.
        let angle = angle_sq.simd_max(f32x4::splat(1e-16)).sqrt();
        let inv_angle = f32x4::splat(1.0) / angle;

        let ax = rx * inv_angle;
        let ay = ry * inv_angle;
        let az = rz * inv_angle;

        let angle_arr = angle.to_array();
        let cos_a = f32x4::from_array(angle_arr.map(f32::cos));
        let sin_a = f32x4::from_array(angle_arr.map(f32::sin));

        let one = f32x4::splat(1.0);
        let zero = f32x4::splat(0.0);
        let t = one - cos_a;

        let t_ax = t * ax;
        let t_ay = t * ay;
        let t_az = t * az;

        let sin_ax = sin_a * ax;
        let sin_ay = sin_a * ay;
        let sin_az = sin_a * az;

        let r00 = degenerate.select(one, t_ax * ax + cos_a);
        let r01 = degenerate.select(zero, t_ax * ay - sin_az);
        let r02 = degenerate.select(zero, t_ax * az + sin_ay);
        let r10 = degenerate.select(zero, t_ay * ax + sin_az);
        let r11 = degenerate.select(one, t_ay * ay + cos_a);
        let r12 = degenerate.select(zero, t_ay * az - sin_ax);
        let r20 = degenerate.select(zero, t_az * ax - sin_ay);
        let r21 = degenerate.select(zero, t_az * ay + sin_ax);
        let r22 = degenerate.select(one, t_az * az + cos_a);

        let r00a = r00.to_array();
        let r01a = r01.to_array();
        let r02a = r02.to_array();
        let r10a = r10.to_array();
        let r11a = r11.to_array();
        let r12a = r12.to_array();
        let r20a = r20.to_array();
        let r21a = r21.to_array();
        let r22a = r22.to_array();

        for lane in 0..SIMD_LANE_4 {
            out.push(na::Matrix3::new(
                r00a[lane], r01a[lane], r02a[lane], r10a[lane], r11a[lane], r12a[lane], r20a[lane],
                r21a[lane], r22a[lane],
            ));
        }

        i += SIMD_LANE_4;
    }

    // Scalar remainder for the tail (fewer than 4 joints left).
    while i < n {
        let [rx, ry, rz] = rotations[i];
        out.push(rodrigues_simd(rx, ry, rz));
        i += 1;
    }

    out
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
            // Genuine vector reduction instead of extracting to an array
            // and summing scalars by hand.
            result[(i, j)] = (a_row * b_col).reduce_sum();
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
        result[i] = (row * v_simd).reduce_sum();
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

    // Accumulate each of the 4 rows in its own SIMD register across the
    // whole pass, instead of round-tripping every (matrix, row) pair
    // through `na::Matrix4` indexing -- the old code read `result[(i,j)]`
    // back out of the matrix, added, and immediately wrote it back, every
    // single iteration. The final matrix is now assembled once, at the end.
    let mut acc = [f32x4::splat(0.0); 4];

    for (m, &w) in matrices.iter().zip(weights.iter()) {
        if w.abs() > 1e-12 {
            let w_simd = f32x4::splat(w);
            for (row, acc_row) in acc.iter_mut().enumerate() {
                let m_row = f32x4::from_array([m[(row, 0)], m[(row, 1)], m[(row, 2)], m[(row, 3)]]);
                *acc_row += m_row * w_simd;
            }
        }
    }

    let r0 = acc[0].to_array();
    let r1 = acc[1].to_array();
    let r2 = acc[2].to_array();
    let r3 = acc[3].to_array();

    na::Matrix4::new(
        r0[0], r0[1], r0[2], r0[3], r1[0], r1[1], r1[2], r1[3], r2[0], r2[1], r2[2], r2[3], r3[0],
        r3[1], r3[2], r3[3],
    )
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
/// `dirs` is `[N, 3, K]` in row-major order, so consecutive coefficient
/// indices for a fixed `(vertex, coord)` are contiguous in memory while
/// consecutive vertices are `3*K` floats apart. This walks vertices (then
/// coordinates, then coefficients) in that order -- the traversal direction
/// that matches the array's memory layout -- streaming `dirs` once with
/// real `f32x8` vector loads over the (contiguous) coefficient axis,
/// instead of re-streaming the whole ~N*3*K array once per coefficient
/// with per-element gathers.
pub fn apply_blend_shapes_simd(v: &mut Array2<f32>, dirs: &Array3<f32>, coeffs: &[f32]) {
    let n = v.nrows();
    let k = coeffs.len().min(dirs.shape()[2]);

    if k == 0 || n == 0 {
        return;
    }

    // Zero out negligible coefficients up front so they contribute exactly
    // nothing below -- equivalent to the old "skip near-zero coefficient"
    // check, but composes with a single dense pass over the (contiguous)
    // coefficient axis instead of branching per coefficient.
    let coeffs: Vec<f32> = coeffs[..k]
        .iter()
        .map(|&c| if c.abs() > 1e-12 { c } else { 0.0 })
        .collect();
    if coeffs.iter().all(|&c| c == 0.0) {
        return;
    }

    let dirs_k = dirs.shape()[2];

    if let Some(dirs_flat) = dirs.as_slice() {
        // `dirs` is in standard (row-major, C-contiguous) layout -- true
        // for an owned `Array3` unless explicitly permuted -- so
        // `[i, coord, ..]` is the contiguous run
        // `[(i*3 + coord) * dirs_k .. + dirs_k]` of the flat buffer.
        for i in 0..n {
            for coord in 0..3 {
                let base = (i * 3 + coord) * dirs_k;
                let dir_slice = &dirs_flat[base..base + k];

                let mut acc = f32x8::splat(0.0);
                let mut j = 0;
                while j + SIMD_LANE_8 <= k {
                    let c_vals = f32x8::from_slice(&coeffs[j..j + SIMD_LANE_8]);
                    let d_vals = f32x8::from_slice(&dir_slice[j..j + SIMD_LANE_8]);
                    acc += c_vals * d_vals;
                    j += SIMD_LANE_8;
                }
                let mut delta = acc.reduce_sum();

                // Scalar remainder for K not a multiple of 8.
                while j < k {
                    delta += coeffs[j] * dir_slice[j];
                    j += 1;
                }

                v[[i, coord]] += delta;
            }
        }
    } else {
        // Non-contiguous/non-standard-layout `dirs` view (unexpected for
        // an owned `Array3`, but handled defensively): fall back to plain
        // indexed scalar access rather than assuming a memory layout that
        // doesn't hold.
        for i in 0..n {
            for coord in 0..3 {
                let mut delta = 0.0f32;
                for (j, &c) in coeffs.iter().enumerate() {
                    delta += c * dirs[[i, coord, j]];
                }
                v[[i, coord]] += delta;
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

    debug_assert!(
        lbs_weights.ncols() >= nj,
        "lbs_weights must have at least one column per joint transform"
    );
    debug_assert!(
        v_posed.ncols() >= 3,
        "v_posed must have at least 3 columns (x, y, z)"
    );

    let mut out = Vec::with_capacity(n);
    // Scratch buffer reused across vertices, avoiding one heap allocation
    // and deallocation per vertex in this hot loop (previously
    // `(0..nj).map(...).collect()` allocated a fresh `Vec` every
    // iteration).
    let mut weights = vec![0.0f32; nj];

    for i in 0..n {
        // Gather weights for this vertex
        for (j, w) in weights.iter_mut().enumerate() {
            *w = lbs_weights[[i, j]];
        }

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

    /// Transform all vertices by a **single** 4x4 affine matrix using SIMD.
    ///
    /// Processes 8 vertices at a time: `x`/`y`/`z` are separate contiguous
    /// buffers, so each iteration is three straight `f32x8` loads, nine
    /// broadcast multiplies plus six adds, and three straight `f32x8` stores —
    /// no gathers, no horizontal reductions, no per-lane scalar round-trips.
    ///
    /// # Not used by the FLAME forward pass — by design
    ///
    /// This is a standalone bulk-transform utility, not a step of
    /// [`FlameModel::forward_simd`](crate::model::FlameModel), and that is a
    /// property of the algorithm rather than an oversight. The matrix here is
    /// uniform across the whole buffer, whereas the transform Linear Blend
    /// Skinning applies **varies per vertex**: vertex `i` is moved by
    /// `sum_j lbs_weights[i][j] * skinning[j]`, a different 4x4 for every
    /// vertex. [`apply_lbs_simd`] therefore blends that per-vertex matrix and
    /// applies it in one pass over the vertex buffer, fusing the global
    /// translation into the same pass; the forward pass has no step where all
    /// vertices share one matrix.
    ///
    /// Reformulating LBS as one uniform-matrix pass per joint
    /// (`sum_j w_ij * (M_j * v_i)`) would fit this routine, but it is the
    /// wrong trade for FLAME: it needs a fresh copy of all `3N` coordinates
    /// per joint and re-reads and rewrites the whole vertex buffer once per
    /// joint, against [`apply_lbs_simd`]'s single pass with all five 4x4
    /// joint matrices held hot, and it reads `lbs_weights` down a strided
    /// column instead of along its contiguous rows.
    ///
    /// Use this for what it is good at: applying one rigid or affine
    /// transform to an entire vertex buffer — canonicalisation, world/model
    /// placement, or reusing an `SoA` buffer across several such transforms.
    /// It is exercised by `tests/simd_tests.rs`'s
    /// `test_vertices_soa_transform_vs_aos` (checked against the scalar
    /// `Matrix4 * Vector4` path) and benchmarked in `benches/simd_ops.rs`.
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
/// Computes `(v1 - v0) × (v2 - v0)` using the standard SIMD
/// shuffle-multiply-subtract cross-product identity:
/// `cross(a, b) = shuffle(a, [1,2,0]) * shuffle(b, [2,0,1])
///              - shuffle(a, [2,0,1]) * shuffle(b, [1,2,0])`.
/// All three output components are produced in parallel lanes with no
/// horizontal reduction, unlike a scalar cross product.
#[inline]
#[must_use]
pub fn cross_product_simd(v0: &[f32; 3], v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
    // Edge vectors (4th lane padded with 0.0, unused in the result).
    let e1 = f32x4::from_array([v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2], 0.0]);
    let e2 = f32x4::from_array([v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2], 0.0]);

    let a_yzx: f32x4 = simd_swizzle!(e1, [1, 2, 0, 3]);
    let b_zxy: f32x4 = simd_swizzle!(e2, [2, 0, 1, 3]);
    let a_zxy: f32x4 = simd_swizzle!(e1, [2, 0, 1, 3]);
    let b_yzx: f32x4 = simd_swizzle!(e2, [1, 2, 0, 3]);

    let result = (a_yzx * b_zxy - a_zxy * b_yzx).to_array();
    [result[0], result[1], result[2]]
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

    // Scalar remainder. Must match the SIMD path above exactly: divide by
    // a clamped length unconditionally, rather than leaving near-zero
    // vectors unchanged. Otherwise whether a given near-zero vector is
    // normalized-to-near-zero or left untouched would depend on whether it
    // happened to fall in the vectorized chunk or this remainder -- a pure
    // artifact of array length, not anything about the vector itself.
    while i < n {
        let x = vectors[i][0];
        let y = vectors[i][1];
        let z = vectors[i][2];
        let len = (x * x + y * y + z * z).sqrt();
        let inv_len = 1.0 / len.max(1e-10);
        vectors[i][0] = x * inv_len;
        vectors[i][1] = y * inv_len;
        vectors[i][2] = z * inv_len;
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

    #[test]
    fn test_normalize_vectors_simd_tail_matches_simd_semantics() {
        // Regression test for a tail-handling divergence: with an array
        // length that is not a multiple of 4, the first 4 vectors go
        // through the vectorized path while the rest go through the
        // scalar remainder loop. The identical near-zero input vector must
        // normalize identically regardless of which path handles it --
        // previously the SIMD path divided by a clamped length
        // unconditionally while the scalar remainder left near-zero
        // vectors completely unchanged, so the *same* input vector came
        // out different purely depending on its position in the array.
        let tiny = [1e-20_f32, 0.0, 0.0];
        let mut vectors = vec![tiny, tiny, tiny, tiny, tiny, tiny]; // len 6: not a multiple of 4
        normalize_vectors_simd(&mut vectors);

        let simd_result = vectors[0]; // handled by the vectorized path
        for (idx, v) in vectors.iter().enumerate() {
            for c in 0..3 {
                assert!(
                    (v[c] - simd_result[c]).abs() < 1e-15,
                    "vector {idx} component {c} = {} must match the SIMD-handled \
                     result {} for the identical input vector",
                    v[c],
                    simd_result[c]
                );
            }
        }
    }

    #[test]
    fn test_cross_product_simd_axes() {
        let x_axis = [1.0f32, 0.0, 0.0];
        let y_axis = [0.0f32, 1.0, 0.0];
        let origin = [0.0f32, 0.0, 0.0];
        // (x_axis - origin) x (y_axis - origin) = x_axis x y_axis = z_axis
        let result = cross_product_simd(&origin, &x_axis, &y_axis);
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cross_product_simd_matches_scalar_formula() {
        let v0 = [1.0f32, 2.0, 3.0];
        let v1 = [4.0f32, 0.0, -1.0];
        let v2 = [-2.0f32, 5.0, 1.0];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let expected = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        let result = cross_product_simd(&v0, &v1, &v2);
        for c in 0..3 {
            assert_relative_eq!(result[c], expected[c], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_weighted_matrix_sum_simd_basic() {
        let m0 = na::Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let m1 = na::Matrix4::new(
            2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
        );
        let weights = [0.25_f32, 0.75_f32];
        let result = weighted_matrix_sum_simd(&[m0, m1], &weights);

        // 0.25 * I + 0.75 * 2I = (0.25 + 1.5) * I = 1.75 * I
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.75 } else { 0.0 };
                assert_relative_eq!(result[(i, j)], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_apply_lbs_simd_single_joint_identity() {
        let v_posed = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0])
            .expect("test: array shape should be valid");
        let transforms = [na::Matrix4::identity()];
        let weights_data = Array2::from_shape_vec((2, 1), vec![1.0, 1.0])
            .expect("test: array shape should be valid");
        let translation = [0.1, 0.2, 0.3];

        let result = apply_lbs_simd(&v_posed, &transforms, &weights_data.view(), translation);

        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0].x, 1.1, epsilon = 1e-5);
        assert_relative_eq!(result[0].y, 2.2, epsilon = 1e-5);
        assert_relative_eq!(result[0].z, 3.3, epsilon = 1e-5);
        assert_relative_eq!(result[1].x, -0.9, epsilon = 1e-5);
        assert_relative_eq!(result[1].y, 0.7, epsilon = 1e-5);
        assert_relative_eq!(result[1].z, 2.3, epsilon = 1e-5);
    }

    #[test]
    fn test_rodrigues_batch_matches_scalar_for_five_joints() {
        // 5 joints: exercises one full SIMD group of 4 plus a scalar
        // remainder of 1, including a degenerate (near-zero) rotation.
        let rotations: Vec<[f32; 3]> = vec![
            [0.3, 0.1, -0.2],
            [0.0, 0.0, 0.0], // degenerate: identity
            [1.0, -0.5, 0.7],
            [-0.4, 0.9, 0.2],
            [0.05, -0.05, 0.6], // remainder (5th joint)
        ];

        let batch_result = rodrigues_batch(&rotations);
        assert_eq!(batch_result.len(), rotations.len());

        for (i, &[rx, ry, rz]) in rotations.iter().enumerate() {
            let scalar_result = rodrigues_simd(rx, ry, rz);
            assert!(
                (batch_result[i] - scalar_result).norm() < 1e-4,
                "joint {i}: batch result diverges from scalar rodrigues_simd"
            );
        }
    }
}
