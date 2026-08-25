//! Small fixed-size linear-algebra helpers shared by the registration code.
//!
//! Everything here is `pub(super)`: these are implementation details of
//! `cloud_registration`, deliberately not part of the crate's public surface.
//! Matrices are row-major `[f32; 9]`, vectors are `[f32; 3]`, and quaternions
//! are passed as loose `(qx, qy, qz, qw)` components.

/// 3×3 row-major matrix multiply: C = A * B.
#[inline]
pub(super) fn mat3_mul(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let mut c = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            let mut sum = 0.0f32;
            for k in 0..3 {
                sum += a[row * 3 + k] * b[k * 3 + col];
            }
            c[row * 3 + col] = sum;
        }
    }
    c
}

/// Multiply 3×3 row-major matrix by a 3-vector.
#[inline]
pub(super) fn mat3_vec(m: [f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// Element-wise vector subtraction.
#[inline]
pub(super) fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scalar multiplication of a vector.
#[inline]
pub(super) fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Dot product of two 3-vectors.
#[inline]
pub(super) fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// L2 norm of a 3-vector.
#[inline]
pub(super) fn vec3_len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Convert a unit quaternion `(qx, qy, qz, qw)` to a row-major 3×3 rotation matrix.
///
/// Assumes the input is already normalized.
#[inline]
pub(super) fn quat_to_mat3(qx: f32, qy: f32, qz: f32, qw: f32) -> [f32; 9] {
    let x2 = qx * qx;
    let y2 = qy * qy;
    let z2 = qz * qz;
    let xy = qx * qy;
    let xz = qx * qz;
    let yz = qy * qz;
    let wx = qw * qx;
    let wy = qw * qy;
    let wz = qw * qz;

    [
        1.0 - 2.0 * (y2 + z2),
        2.0 * (xy - wz),
        2.0 * (xz + wy),
        2.0 * (xy + wz),
        1.0 - 2.0 * (x2 + z2),
        2.0 * (yz - wx),
        2.0 * (xz - wy),
        2.0 * (yz + wx),
        1.0 - 2.0 * (x2 + y2),
    ]
}

/// Normalize a quaternion to unit length. Returns `(0, 0, 0, 1)` if near-zero.
#[inline]
pub(super) fn quat_normalize(qx: f32, qy: f32, qz: f32, qw: f32) -> (f32, f32, f32, f32) {
    let len = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
    if len < 1e-10 {
        return (0.0, 0.0, 0.0, 1.0);
    }
    (qx / len, qy / len, qz / len, qw / len)
}

/// Eigenvector belonging to the largest eigenvalue of a symmetric 4×4 matrix.
///
/// Cyclic Jacobi rotations: every sweep zeroes the six off-diagonal entries in
/// turn while accumulating the rotations in `v`, whose columns converge to the
/// eigenvectors. Four sweeps normally suffice for a 4×4; a few more are run so
/// that near-degenerate inputs also settle. Entries that are already negligible
/// against their diagonals are skipped, so converged sweeps cost nothing.
pub(super) fn largest_eigenvector_sym4(input: &[f32; 16]) -> [f32; 4] {
    let mut a = *input;
    let mut v = [0.0f32; 16];
    v[0] = 1.0;
    v[5] = 1.0;
    v[10] = 1.0;
    v[15] = 1.0;

    for _sweep in 0..16 {
        for p in 0..3 {
            for q in (p + 1)..4 {
                let apq = a[p * 4 + q];
                let app = a[p * 4 + p];
                let aqq = a[q * 4 + q];
                if apq.abs() <= 1e-12 * (app.abs() + aqq.abs() + 1e-12) {
                    continue;
                }
                // Rotation that annihilates a[p][q]: t = tan(theta) chosen as the
                // smaller root so that the transform stays well conditioned.
                let theta = (aqq - app) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                a[p * 4 + p] = app - t * apq;
                a[q * 4 + q] = aqq + t * apq;
                a[p * 4 + q] = 0.0;
                a[q * 4 + p] = 0.0;
                for k in 0..4 {
                    if k != p && k != q {
                        let akp = a[k * 4 + p];
                        let akq = a[k * 4 + q];
                        a[k * 4 + p] = c * akp - s * akq;
                        a[k * 4 + q] = s * akp + c * akq;
                        a[p * 4 + k] = a[k * 4 + p];
                        a[q * 4 + k] = a[k * 4 + q];
                    }
                    let vkp = v[k * 4 + p];
                    let vkq = v[k * 4 + q];
                    v[k * 4 + p] = c * vkp - s * vkq;
                    v[k * 4 + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut best = 0usize;
    for i in 1..4 {
        if a[i * 5] > a[best * 5] {
            best = i;
        }
    }
    [v[best], v[4 + best], v[8 + best], v[12 + best]]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud_registration::test_support::{approx_eq, close3, mat3_det};

    // -----------------------------------------------------------------------
    // mat3_mul
    // -----------------------------------------------------------------------

    #[test]
    fn test_mat3_mul_identity() {
        let id = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let m = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let r = mat3_mul(id, m);
        for i in 0..9 {
            assert!(approx_eq(r[i], m[i], 1e-6));
        }
    }

    #[test]
    fn test_mat3_mul_known() {
        // [[1,0],[0,1]] × [[1,2],[3,4]] but 3×3 version
        let a = [2.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0];
        let r = mat3_mul(a, b);
        assert!(approx_eq(r[0], 2.0, 1e-6));
        assert!(approx_eq(r[4], 3.0, 1e-6));
        assert!(approx_eq(r[8], 5.0, 1e-6));
    }

    // -----------------------------------------------------------------------
    // quat_to_mat3
    // -----------------------------------------------------------------------

    #[test]
    fn test_quat_to_mat3_identity() {
        let r = quat_to_mat3(0.0, 0.0, 0.0, 1.0);
        let id = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..9 {
            assert!(
                approx_eq(r[i], id[i], 1e-6),
                "idx {}: {} vs {}",
                i,
                r[i],
                id[i]
            );
        }
    }

    #[test]
    fn test_quat_to_mat3_90deg_z() {
        // quat for 90° around z: (0, 0, sin(45°), cos(45°))
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let r = quat_to_mat3(0.0, 0.0, s, s);
        // Should map x → y, y → -x
        let xp = mat3_vec(r, [1.0, 0.0, 0.0]);
        assert!(approx_eq(xp[0], 0.0, 1e-5));
        assert!(approx_eq(xp[1], 1.0, 1e-5));
    }

    #[test]
    fn test_quat_to_mat3_180deg_x() {
        // 180° around x: y→-y, z→-z
        let r = quat_to_mat3(1.0, 0.0, 0.0, 0.0);
        let yp = mat3_vec(r, [0.0, 1.0, 0.0]);
        assert!(approx_eq(yp[1], -1.0, 1e-5));
    }

    // -----------------------------------------------------------------------
    // quat_normalize
    // -----------------------------------------------------------------------

    #[test]
    fn test_quat_normalize_unit() {
        let (x, y, z, w) = quat_normalize(0.0, 0.0, 0.0, 1.0);
        assert!(approx_eq(w, 1.0, 1e-7));
        assert!(approx_eq(x, 0.0, 1e-7));
        assert!(approx_eq(y, 0.0, 1e-7));
        assert!(approx_eq(z, 0.0, 1e-7));
    }

    #[test]
    fn test_quat_normalize_arbitrary() {
        let (x, y, z, w) = quat_normalize(1.0, 1.0, 1.0, 1.0);
        let len = (x * x + y * y + z * z + w * w).sqrt();
        assert!(approx_eq(len, 1.0, 1e-6));
    }

    #[test]
    fn test_quat_normalize_zero() {
        let (x, y, z, w) = quat_normalize(0.0, 0.0, 0.0, 0.0);
        assert!(approx_eq(x, 0.0, 1e-7));
        assert!(approx_eq(y, 0.0, 1e-7));
        assert!(approx_eq(z, 0.0, 1e-7));
        assert!(approx_eq(w, 1.0, 1e-7));
    }

    // -----------------------------------------------------------------------
    // largest_eigenvector_sym4
    // -----------------------------------------------------------------------

    #[test]
    fn test_largest_eigenvector_of_diagonal() {
        // A diagonal matrix: the eigenvector for the largest entry is that axis.
        let mut m = [0.0f32; 16];
        m[0] = 1.0;
        m[5] = 7.0;
        m[10] = -2.0;
        m[15] = 3.0;
        let v = largest_eigenvector_sym4(&m);
        assert!(approx_eq(v[1].abs(), 1.0, 1e-5), "{:?}", v);
        for k in [0usize, 2, 3] {
            assert!(approx_eq(v[k], 0.0, 1e-5), "{:?}", v);
        }
    }

    // -----------------------------------------------------------------------
    // Vector helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_helpers() {
        let ex = [1.0f32, 0.0, 0.0];
        let ey = [0.0f32, 1.0, 0.0];
        let v = [1.0f32, 2.0, 3.0];
        let ones = [1.0f32, 1.0, 1.0];
        assert!(close3(vec3_sub([3.0, 2.0, 1.0], ones), [2.0, 1.0, 0.0]));
        assert!(close3(vec3_scale(v, 2.0), [2.0, 4.0, 6.0]));
        assert!(approx_eq(vec3_dot(ex, ey), 0.0, 1e-6));
        assert!(approx_eq(vec3_dot(v, v), 14.0, 1e-5));
        assert!(approx_eq(vec3_len([3.0, 4.0, 0.0]), 5.0, 1e-5));
        let diag = [2.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        assert!(close3(mat3_vec(diag, [1.0, 2.0, 3.0]), [2.0, 6.0, 12.0]));
        assert!(approx_eq(mat3_det(diag), 24.0, 1e-4));
    }
}
