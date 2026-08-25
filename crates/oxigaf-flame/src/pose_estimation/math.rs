//! Private matrix, quaternion, and PRNG math helpers shared by
//! [`super::types`] (`HeadPose`/`PoseTracker`) and [`super::functions`]
//! (the pose solvers). Split out verbatim from the former monolithic
//! `pose_estimation.rs`.

use crate::rigid_alignment::svd_3x3;

// ---------------------------------------------------------------------------
// Utility matrix operations (private)
// ---------------------------------------------------------------------------

/// Multiply two 3×3 matrices: `A * B`.
pub(super) fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Multiply a 3×3 matrix by a 3-vector: `M * v`.
pub(super) fn mat3_vec3_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Transpose a 3×3 matrix.
pub(super) fn mat3_transpose(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Return the 3×3 identity matrix.
pub(super) fn mat3_identity() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Convert a row-major flat 3×3 matrix into nested-array form.
pub(super) fn mat3_from_flat(m: &[f32; 9]) -> [[f32; 3]; 3] {
    [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]
}

/// Convert a nested-array 3×3 matrix into row-major flat form.
pub(super) fn mat3_to_flat(m: &[[f32; 3]; 3]) -> [f32; 9] {
    [
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    ]
}

/// Project an arbitrary 3×3 matrix onto the closest proper rotation (`SO(3)`).
///
/// Polar factor of the SVD, `R = U·Vᵀ`; `svd_3x3` guarantees `det(U) = det(V) =
/// +1`, so the result is proper.  A fully degenerate input yields the identity.
pub(super) fn orthonormalize_rotation(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let (u, sv, vt) = svd_3x3(&mat3_to_flat(m));
    if sv[0] < 1e-9 {
        return mat3_identity();
    }
    mat3_mul(&mat3_from_flat(&u), &mat3_from_flat(&vt))
}

/// Moore–Penrose pseudo-inverse of a 3×3 matrix.
///
/// Singular values below `1e-5 · σ_max` are treated as zero, so coplanar model
/// points (common for facial landmarks) yield the minimum-norm solution instead
/// of a blow-up along the degenerate direction.
pub(super) fn mat3_pseudo_inverse(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let (u, sv, vt) = svd_3x3(&mat3_to_flat(m));
    let tol = sv[0] * 1e-5;

    let mut s_inv = [[0.0f32; 3]; 3];
    for (i, row) in s_inv.iter_mut().enumerate() {
        if sv[i] > tol && sv[i] > 0.0 {
            row[i] = 1.0 / sv[i];
        }
    }

    let v_mat = mat3_transpose(&mat3_from_flat(&vt));
    let u_t = mat3_transpose(&mat3_from_flat(&u));
    mat3_mul(&mat3_mul(&v_mat, &s_inv), &u_t)
}

/// Euclidean norm of a 3-vector.
pub(super) fn vec3_norm(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ---------------------------------------------------------------------------
// Quaternion helpers (private) — used for rotation-preserving smoothing
// ---------------------------------------------------------------------------

/// Normalize a quaternion `[w, x, y, z]`; returns the identity for a zero input.
pub(super) fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

/// Convert a rotation matrix into a unit quaternion `[w, x, y, z]`.
///
/// Uses Shepperd's method (largest-diagonal branch) for numerical stability.
pub(super) fn mat3_to_quat(r: &[[f32; 3]; 3]) -> [f32; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).max(1e-12).sqrt() * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).max(1e-12).sqrt() * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).max(1e-12).sqrt() * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).max(1e-12).sqrt() * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    };
    quat_normalize(q)
}

/// Convert a unit quaternion `[w, x, y, z]` into a rotation matrix.
pub(super) fn quat_to_mat3(quat: [f32; 4]) -> [[f32; 3]; 3] {
    let [quat_w, quat_x, quat_y, quat_z] = quat_normalize(quat);
    [
        [
            1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z),
            2.0 * (quat_x * quat_y - quat_w * quat_z),
            2.0 * (quat_x * quat_z + quat_w * quat_y),
        ],
        [
            2.0 * (quat_x * quat_y + quat_w * quat_z),
            1.0 - 2.0 * (quat_x * quat_x + quat_z * quat_z),
            2.0 * (quat_y * quat_z - quat_w * quat_x),
        ],
        [
            2.0 * (quat_x * quat_z - quat_w * quat_y),
            2.0 * (quat_y * quat_z + quat_w * quat_x),
            1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y),
        ],
    ]
}

/// Spherical linear interpolation between two quaternions.
///
/// `t = 0` returns `q0`, `t = 1` returns `q1`.  Antipodal inputs are handled by
/// negating `q1` so the interpolation always takes the shorter arc.
pub(super) fn quat_slerp(q0: [f32; 4], q1: [f32; 4], t: f32) -> [f32; 4] {
    let q0 = quat_normalize(q0);
    let mut q1 = quat_normalize(q1);

    let mut dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    if dot < 0.0 {
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
        dot = -dot;
    }

    // Nearly parallel: slerp is numerically unstable, use normalized lerp.
    let (wa, wb) = if dot > 0.9995 {
        (1.0 - t, t)
    } else {
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        if sin_theta.abs() < 1e-9 {
            return q0;
        }
        (
            ((1.0 - t) * theta).sin() / sin_theta,
            (t * theta).sin() / sin_theta,
        )
    };
    quat_normalize([
        wa * q0[0] + wb * q1[0],
        wa * q0[1] + wb * q1[1],
        wa * q0[2] + wb * q1[2],
        wa * q0[3] + wb * q1[3],
    ])
}

// ---------------------------------------------------------------------------
// Deterministic RNG for RANSAC sampling (private)
// ---------------------------------------------------------------------------

/// Seed for the RANSAC sampler: deterministic on purpose, so two runs over the
/// same correspondences always produce the same pose.
pub(super) const RANSAC_SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// Minimal `SplitMix64` generator (deterministic, dependency-free).
pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `[0, n)`.  Returns `0` when `n == 0`.
    pub(super) fn next_index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}
