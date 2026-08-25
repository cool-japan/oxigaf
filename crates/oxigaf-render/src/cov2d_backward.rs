//! CPU reference backward pass for the 2D covariance projection.
//!
//! Computes gradients through the transformation:
//!   Σ2D = J · R · Σ3D · Rᵀ · Jᵀ
//!
//! where J is the 2×3 perspective projection Jacobian, R is the 3×3 camera
//! rotation, and Σ3D is the 3D Gaussian covariance matrix.
//!
//! This module is a pure-CPU reference implementation, requiring no GPU.
//!
//! # Relationship to the shader that actually runs
//!
//! The kernel used in training is `preprocess_backward` in
//! `shaders/preprocess_bwd.wgsl` (sections "2. grad_cov2d from grad_conics" and
//! "3. grad_cov3d from grad_cov2d"); `shaders/cov2d_bwd.wgsl` is a third,
//! entry-point-less transcription of the same derivation. Nothing in the build
//! keeps these in step automatically, so the test module below pins the two
//! things that silently drift:
//!
//! * `test_shader_conic_inverse_gradient_matches_finite_differences` checks the
//!   shader's `∂L/∂conic → ∂L/∂Σ2D` matrix-inverse derivative — a step this
//!   module does not model at all — against finite differences.
//! * `test_shader_cov3d_chain_matches_cpu_reference` checks the shader's
//!   `∂L/∂Σ3D = Tᵀ · G · T` (with `T = J·W`) against [`cov2d_backward`].
//!
//! **Off-diagonal convention.** The two derivations disagree by a factor of two
//! on the off-diagonal *by design*, and reconciling them is the whole point of
//! that second test: the shader's `dL_db_elem` is the gradient w.r.t. the
//! off-diagonal *matrix element* (used unhalved in both slots of a symmetric
//! matrix), whereas [`Cov2dBwdInput::d_cov2d`]`[1]` is the *combined* gradient
//! for both slots and is halved internally. Converting one to the other means
//! doubling/halving the inputs, never changing the math on either side.

use nalgebra as na;

/// Gradients flowing back through the 2D covariance computation.
#[derive(Debug, Clone, PartialEq)]
pub struct Cov2dGrads {
    /// Gradient w.r.t. the upper-triangular elements of 3D covariance Σ3D.
    ///
    /// Stored as `[cov3d_0, cov3d_1, cov3d_2, cov3d_3, cov3d_4, cov3d_5]`
    /// corresponding to the symmetric matrix entries
    /// `[(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]`.
    pub d_cov3d: [f32; 6],
}

/// Inputs to the 2D-covariance backward pass.
#[derive(Debug, Clone)]
pub struct Cov2dBwdInput {
    /// Upper-triangular 3D covariance (6 floats, row-major upper triangle).
    ///
    /// Layout: `[Σ(0,0), Σ(0,1), Σ(0,2), Σ(1,1), Σ(1,2), Σ(2,2)]`.
    pub cov3d: [f32; 6],

    /// Camera rotation matrix stored column-major.
    ///
    /// `view_rotation[col][row]`, so `view_rotation[c][r]` is the element at
    /// row `r`, column `c`.
    pub view_rotation: [[f32; 3]; 3],

    /// Tangent-plane Jacobian for the perspective projection (2×3 matrix).
    ///
    /// Rows are `∂x/∂pos` and `∂y/∂pos`.  `jacobian[row][col]`.
    pub jacobian: [[f32; 3]; 2],

    /// Gradient of the loss w.r.t. the 3 independent elements of Σ2D.
    ///
    /// Layout: `[∂L/∂Σxx, ∂L/∂Σxy, ∂L/∂Σyy]`.
    /// The off-diagonal element `Σxy` is treated as the *combined* gradient
    /// (i.e. both `Σ(0,1)` and `Σ(1,0)` are represented by a single value).
    pub d_cov2d: [f32; 3],
}

/// CPU reference backward pass for the 2D covariance projection.
///
/// Computes `∂L/∂Σ3D` given `∂L/∂Σ2D` via the chain rule through
///
/// ```text
/// Σ2D = J · R · Σ3D · Rᵀ · Jᵀ
/// ```
///
/// The chain rule gives:
///
/// ```text
/// ∂L/∂Σ3D = Rᵀ · Jᵀ · D̃ · J · R
/// ```
///
/// where `D̃ = [[d_xx, d_xy/2], [d_xy/2, d_yy]]` is the symmetrized upstream
/// gradient matrix (off-diagonal halved so that the subsequent accumulation of
/// `m(i,j) + m(j,i)` for off-diagonal output entries yields the correct total
/// gradient consistent with the symmetric parametrization of Σ3D).
///
/// Because Σ3D is symmetric, off-diagonal entries of the resulting matrix are
/// accumulated (summed with their transpose counterpart) before being returned.
pub fn cov2d_backward(input: &Cov2dBwdInput) -> Cov2dGrads {
    // Build R from column-major storage: view_rotation[col][row].
    let vr = &input.view_rotation;
    let r = na::Matrix3::<f32>::new(
        vr[0][0], vr[1][0], vr[2][0], vr[0][1], vr[1][1], vr[2][1], vr[0][2], vr[1][2], vr[2][2],
    );

    // Build J (2×3) from row-major storage: jacobian[row][col].
    let jac = &input.jacobian;
    let j = na::Matrix2x3::<f32>::new(
        jac[0][0], jac[0][1], jac[0][2], jac[1][0], jac[1][1], jac[1][2],
    );

    // Build the 2×2 gradient matrix ∂L/∂Σ2D.
    //
    // Convention: `d_cov2d = [d_xx, d_xy, d_yy]` where `d_xy` is the gradient
    // w.r.t. the single stored off-diagonal element Σ2D(0,1).  Because Σ2D is
    // symmetric and the loss is `L = d_xx·Σxx + d_xy·Σxy + d_yy·Σyy`, the
    // matrix form consistent with chain-rule differentiation through a symmetric
    // matrix is the symmetrized version `[[d_xx, d_xy/2], [d_xy/2, d_yy]]`.
    // This distributes the off-diagonal gradient evenly across both off-diagonal
    // positions, which is required for the accumulation step below to agree with
    // finite differences.
    let d = &input.d_cov2d;
    let d_sigma2d = na::Matrix2::<f32>::new(d[0], d[1] * 0.5, d[1] * 0.5, d[2]);

    // Chain rule: ∂L/∂Σ3D = Rᵀ · Jᵀ · (∂L/∂Σ2D) · J · R
    //
    // Dimensions:
    //   Rᵀ          : 3×3
    //   Jᵀ          : 3×2
    //   ∂L/∂Σ2D    : 2×2
    //   J           : 2×3
    //   R           : 3×3
    //   result      : 3×3
    let m: na::Matrix3<f32> = r.transpose() * j.transpose() * d_sigma2d * j * r;

    // Extract the upper triangle and accumulate off-diagonal entries (symmetric
    // matrix: m[i,j] + m[j,i]) to account for the symmetry of Σ3D.
    let d_cov3d = [
        m[(0, 0)],
        m[(0, 1)] + m[(1, 0)],
        m[(0, 2)] + m[(2, 0)],
        m[(1, 1)],
        m[(1, 2)] + m[(2, 1)],
        m[(2, 2)],
    ];

    Cov2dGrads { d_cov3d }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Forward pass: Σ2D = J · R · Σ3D · Rᵀ · Jᵀ.
    ///
    /// Returns the upper triangle `[Σxx, Σxy, Σyy]`.
    fn cov2d_forward(
        cov3d: [f32; 6],
        view_rotation: [[f32; 3]; 3],
        jacobian: [[f32; 3]; 2],
    ) -> [f32; 3] {
        let s = cov3d;
        let sigma3d =
            nalgebra::Matrix3::<f32>::new(s[0], s[1], s[2], s[1], s[3], s[4], s[2], s[4], s[5]);
        let vr = view_rotation;
        let r = nalgebra::Matrix3::<f32>::new(
            vr[0][0], vr[1][0], vr[2][0], vr[0][1], vr[1][1], vr[2][1], vr[0][2], vr[1][2],
            vr[2][2],
        );
        let jac = jacobian;
        let j = nalgebra::Matrix2x3::<f32>::new(
            jac[0][0], jac[0][1], jac[0][2], jac[1][0], jac[1][1], jac[1][2],
        );
        let sigma2d = j * r * sigma3d * r.transpose() * j.transpose();
        [sigma2d[(0, 0)], sigma2d[(0, 1)], sigma2d[(1, 1)]]
    }

    /// Finite-difference gradient check.
    ///
    /// Computes the scalar loss `L = Σ_i d_cov2d[i] * cov2d_fwd[i]` and
    /// approximates `∂L/∂cov3d[k]` by central differences, then compares
    /// against the analytical gradient from `cov2d_backward`.
    ///
    /// Returns `true` if every element satisfies the relative-error tolerance.
    fn fd_check(
        cov3d: [f32; 6],
        view_rotation: [[f32; 3]; 3],
        jacobian: [[f32; 3]; 2],
        d_cov2d: [f32; 3],
        eps: f32,
        rel_tol: f32,
    ) -> bool {
        // Analytical gradient
        let input = Cov2dBwdInput {
            cov3d,
            view_rotation,
            jacobian,
            d_cov2d,
        };
        let grads = cov2d_backward(&input);

        // Scalar loss: L = sum_i d_cov2d[i] * fwd[i]
        let loss = |c: [f32; 6]| -> f32 {
            let fwd = cov2d_forward(c, view_rotation, jacobian);
            d_cov2d[0] * fwd[0] + d_cov2d[1] * fwd[1] + d_cov2d[2] * fwd[2]
        };

        let mut all_ok = true;
        for k in 0..6 {
            let mut c_plus = cov3d;
            let mut c_minus = cov3d;
            c_plus[k] += eps;
            c_minus[k] -= eps;
            let fd = (loss(c_plus) - loss(c_minus)) / (2.0 * eps);
            let an = grads.d_cov3d[k];
            // Use a floor of 1e-4 so near-zero gradients do not inflate the ratio.
            let scale = fd.abs().max(an.abs()).max(1e-4_f32);
            let rel_err = (fd - an).abs() / scale;
            if rel_err > rel_tol {
                eprintln!("k={k}: fd={fd:.6} an={an:.6} rel_err={rel_err:.6}");
                all_ok = false;
            }
        }
        all_ok
    }

    // -----------------------------------------------------------------------
    // Test fixtures
    // -----------------------------------------------------------------------

    /// Identity rotation (3×3 identity, column-major).
    fn identity_rotation() -> [[f32; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    /// A 30° rotation around the Y-axis (column-major: [col0, col1, col2]).
    fn rotation_30_deg_y() -> [[f32; 3]; 3] {
        let theta: f32 = std::f32::consts::PI / 6.0;
        let c = theta.cos();
        let s = theta.sin();
        // R_y = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        // column-major storage: col0=[c,0,-s], col1=[0,1,0], col2=[s,0,c]
        [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
    }

    /// A simple 2×3 scaled perspective Jacobian.
    fn scaled_jacobian() -> [[f32; 3]; 2] {
        [[2.0, 0.0, 0.5], [0.0, 2.0, 0.3]]
    }

    /// Isotropic 3D covariance (diagonal, equal entries).
    fn isotropic_cov3d() -> [f32; 6] {
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    }

    /// Anisotropic 3D covariance with off-diagonal entries.
    fn anisotropic_cov3d() -> [f32; 6] {
        [4.0, 0.5, 0.2, 2.0, 0.1, 1.0]
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    /// Identity rotation, scaled Jacobian, isotropic Σ3D, d_cov2d = [1, 0, 1].
    #[test]
    fn test_fd_gradient_identity_rotation() {
        let cov3d = isotropic_cov3d();
        let view_rotation = identity_rotation();
        let jacobian = scaled_jacobian();
        let d_cov2d = [1.0_f32, 0.0, 1.0];

        assert!(
            fd_check(cov3d, view_rotation, jacobian, d_cov2d, 1e-3, 0.01),
            "finite-difference gradient check failed (identity rotation)"
        );
    }

    /// 30° Y-axis rotation, anisotropic Σ3D, mixed d_cov2d.
    #[test]
    fn test_fd_gradient_general_rotation() {
        let cov3d = anisotropic_cov3d();
        let view_rotation = rotation_30_deg_y();
        let jacobian = scaled_jacobian();
        let d_cov2d = [1.0_f32, 0.5, 0.8];

        assert!(
            fd_check(cov3d, view_rotation, jacobian, d_cov2d, 1e-3, 0.01),
            "finite-difference gradient check failed (general rotation)"
        );
    }

    /// When all upstream gradients are zero, all output gradients must be zero.
    #[test]
    fn test_zero_gradient_produces_zero_output() {
        let input = Cov2dBwdInput {
            cov3d: isotropic_cov3d(),
            view_rotation: rotation_30_deg_y(),
            jacobian: scaled_jacobian(),
            d_cov2d: [0.0, 0.0, 0.0],
        };
        let grads = cov2d_backward(&input);
        for (k, &g) in grads.d_cov3d.iter().enumerate() {
            assert_eq!(g, 0.0, "d_cov3d[{k}] should be zero for zero upstream grad");
        }
    }

    /// Diagonal (positive-definite) Σ3D should yield all-finite gradients.
    #[test]
    fn test_symmetric_sigma3d_preserved() {
        let input = Cov2dBwdInput {
            cov3d: [3.0, 0.0, 0.0, 2.0, 0.0, 1.0],
            view_rotation: rotation_30_deg_y(),
            jacobian: scaled_jacobian(),
            d_cov2d: [1.0, 0.5, 1.0],
        };
        let grads = cov2d_backward(&input);
        for (k, &g) in grads.d_cov3d.iter().enumerate() {
            assert!(g.is_finite(), "d_cov3d[{k}] is not finite: {g}");
        }
    }

    /// Calling `cov2d_backward` twice with the same input must produce identical results.
    #[test]
    fn test_identity_inputs_deterministic() {
        let input = Cov2dBwdInput {
            cov3d: anisotropic_cov3d(),
            view_rotation: rotation_30_deg_y(),
            jacobian: scaled_jacobian(),
            d_cov2d: [1.0, 0.3, 0.7],
        };
        let g1 = cov2d_backward(&input);
        let g2 = cov2d_backward(&input);
        assert_eq!(g1, g2, "cov2d_backward must be deterministic");
    }

    /// Isotropic Σ3D with an arbitrary J and R should produce all-finite gradients.
    #[test]
    fn test_isotropic_sigma3d_gives_finite_grads() {
        let input = Cov2dBwdInput {
            cov3d: isotropic_cov3d(),
            view_rotation: rotation_30_deg_y(),
            jacobian: [[1.5, 0.0, 0.2], [0.0, 1.5, 0.1]],
            d_cov2d: [0.5, 0.2, 0.5],
        };
        let grads = cov2d_backward(&input);
        for (k, &g) in grads.d_cov3d.iter().enumerate() {
            assert!(g.is_finite(), "d_cov3d[{k}] is not finite: {g}");
        }
    }

    /// Large Σ3D diagonal values should still pass the finite-difference check.
    #[test]
    fn test_large_sigma_fd_gradient() {
        let cov3d = [100.0_f32, 0.0, 0.0, 50.0, 0.0, 25.0];
        let view_rotation = rotation_30_deg_y();
        let jacobian = scaled_jacobian();
        let d_cov2d = [1.0_f32, 0.0, 1.0];

        assert!(
            fd_check(cov3d, view_rotation, jacobian, d_cov2d, 1e-1, 0.01),
            "finite-difference gradient check failed (large Σ3D)"
        );
    }

    // -----------------------------------------------------------------------
    // Parity with `shaders/preprocess_bwd.wgsl`
    //
    // These mirror the shader's arithmetic literally so that a change to either
    // derivation shows up as a failure here instead of as a silently biased
    // training run. See the module docs for the off-diagonal convention.
    // -----------------------------------------------------------------------

    /// Forward conic: `conic = inverse([[a, b], [b, c]])`, returned as the three
    /// stored scalars `[ca, cb, cc]` (matching `preprocess.wgsl`'s
    /// `conics[idx] = (c/det, -b/det, a/det)`).
    fn conic_of(cov2d: [f32; 3]) -> [f32; 3] {
        let (a, b, c) = (cov2d[0], cov2d[1], cov2d[2]);
        let det = a * c - b * b;
        [c / det, -b / det, a / det]
    }

    /// Literal mirror of `preprocess_bwd.wgsl` section 2
    /// (`dL_da` / `dL_db_elem` / `dL_dc_cov`).
    ///
    /// `d_conic` is `[∂L/∂ca, ∂L/∂cb, ∂L/∂cc]`, where `cb` is the single stored
    /// off-diagonal scalar. The result is the **matrix-element** gradient
    /// `[∂L/∂Σ2D(0,0), ∂L/∂Σ2D(0,1), ∂L/∂Σ2D(1,1)]`.
    fn shader_dl_dcov2d(conic: [f32; 3], d_conic: [f32; 3]) -> [f32; 3] {
        let (ca, cb, cc) = (conic[0], conic[1], conic[2]);
        let ga = d_conic[0];
        let gb_half = d_conic[1] * 0.5;
        let gc = d_conic[2];
        [
            -(ca * ga + cb * gb_half) * ca - (ca * gb_half + cb * gc) * cb,
            -(ca * ga + cb * gb_half) * cb - (ca * gb_half + cb * gc) * cc,
            -(cb * ga + cc * gb_half) * cb - (cb * gb_half + cc * gc) * cc,
        ]
    }

    /// The shader's `∂L/∂conic → ∂L/∂Σ2D` step (matrix-inverse derivative
    /// `∂L/∂S = −S⁻¹ · G · S⁻¹`) must agree with finite differences of
    /// `conic = inverse(Σ2D)`.
    ///
    /// This step has no counterpart in [`cov2d_backward`], so nothing else in
    /// the crate verifies it.
    #[test]
    fn test_shader_conic_inverse_gradient_matches_finite_differences() {
        let cov2d = [3.0_f32, 0.7, 2.0];
        let d_conic = [0.9_f32, -0.4, 1.3];

        let analytic = shader_dl_dcov2d(conic_of(cov2d), d_conic);

        let loss = |s: [f32; 3]| -> f32 {
            let k = conic_of(s);
            d_conic[0] * k[0] + d_conic[1] * k[1] + d_conic[2] * k[2]
        };

        // The stored scalar `b` occupies BOTH off-diagonal slots of the
        // symmetric matrix, so perturbing it moves the loss by twice the
        // matrix-element gradient.
        let slot_multiplicity = [1.0_f32, 2.0, 1.0];
        let eps = 1e-3_f32;
        for k in 0..3 {
            let mut plus = cov2d;
            let mut minus = cov2d;
            plus[k] += eps;
            minus[k] -= eps;
            let fd = (loss(plus) - loss(minus)) / (2.0 * eps);
            let an = analytic[k] * slot_multiplicity[k];
            let scale = fd.abs().max(an.abs()).max(1e-4_f32);
            assert!(
                (fd - an).abs() / scale < 0.01,
                "conic-inverse gradient mismatch at k={k}: fd={fd:.6} analytic={an:.6}"
            );
        }
    }

    /// The shader's `∂L/∂Σ3D = Tᵀ · G · T` (with `T = J · W`) must reproduce
    /// [`cov2d_backward`] once the off-diagonal convention is mapped.
    #[test]
    fn test_shader_cov3d_chain_matches_cpu_reference() {
        // Projection fixture built exactly as preprocess_bwd.wgsl builds `J`,
        // whose third row is identically zero.
        let fx = 500.0_f32;
        let fy = 480.0_f32;
        let tz = 4.0_f32;
        let tz2 = tz * tz;
        let vx = 0.6_f32;
        let vy = -0.35_f32;
        let jacobian = [[fx / tz, 0.0, fx * vx / tz2], [0.0, fy / tz, fy * vy / tz2]];
        let view_rotation = rotation_30_deg_y();

        // Matrix-element gradient ∂L/∂Σ2D, straight out of the shader's step 2.
        let d_elem = shader_dl_dcov2d(conic_of([3.0, 0.7, 2.0]), [0.9, -0.4, 1.3]);

        // Shader step 3, in matrix form.
        let vr = view_rotation;
        let r = nalgebra::Matrix3::<f32>::new(
            vr[0][0], vr[1][0], vr[2][0], vr[0][1], vr[1][1], vr[2][1], vr[0][2], vr[1][2],
            vr[2][2],
        );
        let j = nalgebra::Matrix2x3::<f32>::new(
            jacobian[0][0],
            jacobian[0][1],
            jacobian[0][2],
            jacobian[1][0],
            jacobian[1][1],
            jacobian[1][2],
        );
        let t = j * r;
        let g = nalgebra::Matrix2::<f32>::new(d_elem[0], d_elem[1], d_elem[1], d_elem[2]);
        let m = t.transpose() * g * t;
        let shader_grads = [
            m[(0, 0)],
            m[(0, 1)] + m[(1, 0)],
            m[(0, 2)] + m[(2, 0)],
            m[(1, 1)],
            m[(1, 2)] + m[(2, 1)],
            m[(2, 2)],
        ];

        // `cov2d_backward` halves `d_cov2d[1]` internally, so feed it the
        // combined (doubled) off-diagonal.
        let cpu = cov2d_backward(&Cov2dBwdInput {
            // Σ3D itself does not enter this chain; any well-formed value works.
            cov3d: anisotropic_cov3d(),
            view_rotation,
            jacobian,
            d_cov2d: [d_elem[0], 2.0 * d_elem[1], d_elem[2]],
        });

        let scale = shader_grads
            .iter()
            .chain(cpu.d_cov3d.iter())
            .fold(0.0_f32, |acc, v| acc.max(v.abs()))
            .max(1e-6_f32);
        for (k, (&sg, &cg)) in shader_grads.iter().zip(cpu.d_cov3d.iter()).enumerate() {
            let diff = (sg - cg).abs();
            assert!(
                diff <= 1e-4 * scale,
                "shader/CPU cov3D gradient drift at k={k}: shader={sg:.6} cpu={cg:.6}"
            );
        }
    }

    /// Scaling d_cov2d by a factor must scale all d_cov3d by the same factor
    /// (linearity of the chain rule).
    #[test]
    fn test_gradient_linearity() {
        let cov3d = anisotropic_cov3d();
        let view_rotation = rotation_30_deg_y();
        let jacobian = scaled_jacobian();
        let d_cov2d_base = [0.5_f32, 0.3, 0.7];

        let input_base = Cov2dBwdInput {
            cov3d,
            view_rotation,
            jacobian,
            d_cov2d: d_cov2d_base,
        };
        let grads_base = cov2d_backward(&input_base);

        let scale = 2.0_f32;
        let input_scaled = Cov2dBwdInput {
            cov3d,
            view_rotation,
            jacobian,
            d_cov2d: [
                d_cov2d_base[0] * scale,
                d_cov2d_base[1] * scale,
                d_cov2d_base[2] * scale,
            ],
        };
        let grads_scaled = cov2d_backward(&input_scaled);

        for k in 0..6 {
            let expected = grads_base.d_cov3d[k] * scale;
            let actual = grads_scaled.d_cov3d[k];
            let tol = expected.abs().max(1e-6_f32) * 1e-5_f32;
            assert!(
                (actual - expected).abs() <= tol,
                "linearity violated at k={k}: expected {expected:.8}, got {actual:.8}"
            );
        }
    }
}
