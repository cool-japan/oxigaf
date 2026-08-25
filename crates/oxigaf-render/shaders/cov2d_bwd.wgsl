// cov2d_bwd.wgsl — Standalone documentation module: 2D Covariance Backward Pass
//
// Mirrors these sections of `preprocess_bwd.wgsl`'s `preprocess_backward`
// entry point (referenced by section heading, never by line number — the
// headings survive edits above them, line numbers do not):
//
//   `---- 2. grad_cov2d from grad_conics ----`
//   `---- 3. grad_cov3d from grad_cov2d ----`
//   `---- 4. grad_rotation, grad_scale from grad_cov3d ----`
//   `---- 4b. Position gradient through covariance (dJ/dp_view contribution) ----`
//
// all of which live under that file's `--- COV2D BACKWARD ---` banner.
//
// PURPOSE
// ───────
// This file documents and isolates the backward-pass gradient math for the
// 2D covariance / conic computation from 3DGS preprocessing. It has NO
// @compute entry point — it is a reference implementation and tutorial-quality
// documentation intended to be read alongside the active shader.
//
// PARITY
// ──────
// Nothing in the build keeps this file and `preprocess_bwd.wgsl` in step
// automatically. `src/cov2d_backward.rs` carries the tests that pin the two
// steps that silently drift — the conic inverse derivative (block 2) against
// finite differences, and the `∂L/∂Σ3D = Tᵀ·G·T` chain (block 3) against the
// CPU reference — so treat that module's test names as the parity contract.
//
// ONE DELIBERATE DIFFERENCE from `preprocess_bwd.wgsl`: `dL_dpos_cov` below is
// returned in CAMERA space. The active shader immediately rotates it to world
// space (`dL_dpos += transpose(W) * …` at the end of its block 4b); this file
// stops one step earlier so the Jacobian sensitivity can be inspected on its
// own. Callers comparing the two must apply `transpose(W)` themselves.
//
// MATHEMATICAL OVERVIEW
// ─────────────────────
// Forward pass (preprocess.wgsl):
//
//   Given: Gaussian position p_view ∈ ℝ³ (camera space),
//          rotation quaternion q ∈ ℝ⁴ (xyzw),
//          log-scale log_s ∈ ℝ³,
//          world→camera rotation W ∈ ℝ³×³,
//          focal lengths [fx, fy].
//
//   Step A:  R    = quat_to_mat(q)                       3×3 rotation matrix
//            s    = exp(log_s)                            element-wise
//            S    = diag(s)                               3×3 diagonal
//            M    = R · S                                 3×3
//            cov3D = M · Mᵀ                              symmetric 3×3 covariance
//
//   Step B:  tz   = −p_view.z                            positive depth (RH camera)
//            J    = [[fx/tz,  0,  fx·vx/tz²],            2×3 projection Jacobian
//                    [0,  fy/tz,  fy·vy/tz²]]
//            (embedded as 3×3 with zero third row for shader arithmetic)
//            T    = J · W                                3×3 (or 2×3)
//
//   Step C:  cov2D = T · cov3D · Tᵀ + 0.3·I            regularised 2×2
//
//   Step D:  conic = inverse(cov2D)                      2×2 → packed vec3 [a,b,c]
//
// Backward pass (this file, preprocess_bwd.wgsl blocks 2–4b):
//
//   Block 2: ∂L/∂cov2D via conic inverse derivative
//            ∂L/∂S⁻¹ = G (reshaped ∂L/∂conic as symmetric 2×2)
//            ∂L/∂S   = −S⁻¹ · G · S⁻¹    (S = cov2D, S⁻¹ = conic matrix)
//
//   Block 3: ∂L/∂cov3D via projection
//            ∂L/∂cov3D = Tᵀ · (∂L/∂cov2D) · T
//
//   Block 4: ∂L/∂(R, log_s) via M·Mᵀ decomposition
//            ∂L/∂M      = 2 · (∂L/∂cov3D) · M
//            ∂L/∂R      = (∂L/∂M) · Sᵀ
//            ∂L/∂S_diag = diag(Rᵀ · (∂L/∂M))
//            ∂L/∂log_s  = ∂L/∂S_diag · s  (chain through exp)
//            ∂L/∂q      = sum over R-entries of (∂R[i,j]/∂q) · (∂L/∂R[i,j])
//
//   Block 4b: ∂L/∂p_view (cov path) via J sensitivity
//             ∂L/∂T = 2 · (∂L/∂cov2D) · T · cov3D
//             ∂L/∂J = (∂L/∂T) · Wᵀ
//             ∂L/∂vx = (∂L/∂J[0,2]) · fx/tz²
//             ∂L/∂vy = (∂L/∂J[1,2]) · fy/tz²
//             ∂L/∂vz  accumulated from J[0,0], J[1,1], J[0,2], J[1,2] sensitivities
//
// NOTE ON WGSL COLUMN-MAJOR INDEXING
// ────────────────────────────────────
// In WGSL, mat3x3<f32> is column-major: mat[col][row].
// Contrast: nalgebra Matrix3 is row-major indexing [(row, col)].
// This matters for reading the quaternion gradient contraction below.

// ─────────────────────────────────────────────────────────────────────────────
// Helper: quaternion → rotation matrix
// ─────────────────────────────────────────────────────────────────────────────
// Builds a 3×3 rotation matrix from a unit quaternion in xyzw convention.
// WGSL mat3x3 is column-major: columns are the vec3 arguments.
//
// Resulting matrix R satisfies R·Rᵀ = I and det(R) = 1.
// Column 0 = first axis of rotated frame, etc.
fn quat_to_mat(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    // Double the components once for reuse in product terms
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    // Squared double-products: e.g. xx = x * (2x) = 2x²
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    // Cross terms with w
    let wx = w * x2; let wy = w * y2; let wz = w * z2;
    // mat3x3(col0, col1, col2) — columns are written as vec3
    return mat3x3<f32>(
        // col 0  (R[:,0])
        vec3<f32>(1.0 - (yy + zz),  xy + wz,        xz - wy       ),
        // col 1  (R[:,1])
        vec3<f32>(xy - wz,           1.0 - (xx + zz), yz + wx       ),
        // col 2  (R[:,2])
        vec3<f32>(xz + wy,           yz - wx,         1.0 - (xx + yy)),
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Return type
// ─────────────────────────────────────────────────────────────────────────────
struct Cov2dBwdResult {
    /// ∂L/∂log_scale  (shape: [3])  — gradient w.r.t. log of Gaussian scales
    dL_dscale   : vec3<f32>,
    /// ∂L/∂quat in xyzw convention (shape: [4])  — gradient w.r.t. rotation quaternion
    dL_dquat    : vec4<f32>,
    /// ∂L/∂p_view (shape: [3])  — camera-space position gradient from Jacobian sensitivity
    dL_dpos_cov : vec3<f32>,
};

// ─────────────────────────────────────────────────────────────────────────────
// Main backward function
// ─────────────────────────────────────────────────────────────────────────────
//
// Arguments
// ---------
// p_view      : camera-space Gaussian center [vx, vy, vz]  (RH, vz < 0 for visible)
// focal       : [fx, fy] focal lengths in pixels
// W           : world→camera rotation 3×3 (upper-left of view matrix)
// rot         : rotation quaternion xyzw
// scale       : log-scales [sx, sy, sz]  (the raw stored values before exp)
// conic       : packed upper-triangular inverse 2×2 covariance [a, b, c]
//               representing the matrix [[a, b], [b, c]]
// dL_dconic   : ∂L/∂conic = [∂L/∂a, ∂L/∂b, ∂L/∂c]  (incoming loss gradient)
//
// Returns the Cov2dBwdResult struct with gradients for scale, quat, and pos_cov.
fn cov2d_backward(
    p_view    : vec3<f32>,
    focal     : vec2<f32>,
    W         : mat3x3<f32>,
    rot       : vec4<f32>,
    scale     : vec3<f32>,
    conic     : vec3<f32>,
    dL_dconic : vec3<f32>,
) -> Cov2dBwdResult {
    let fx = focal.x;
    let fy = focal.y;
    // Positive depth: camera is right-handed so visible points have p_view.z < 0.
    let tz  = -p_view.z;
    let tz2 = tz * tz;
    let tz3 = tz2 * tz;

    // =========================================================================
    // BLOCK 2 — ∂L/∂cov2D via matrix-inverse chain rule
    // =========================================================================
    //
    // Forward relationship: conic = Σ⁻¹  (2×2 symmetric)
    //
    // The derivative of the matrix inverse is:
    //   d(Σ⁻¹)/dΣ · ΔΣ = −Σ⁻¹ · ΔΣ · Σ⁻¹
    //
    // Transposing to the loss gradient:
    //   ∂L/∂Σ = −Σ⁻¹ · (∂L/∂Σ⁻¹)_sym · Σ⁻¹
    //
    // where (∂L/∂Σ⁻¹)_sym is the symmetric 2×2 matrix built from dL_dconic.
    // Because the conic is stored in packed upper-triangular form [a, b, c],
    // the off-diagonal component dL_dconic.y already represents the combined
    // gradient for BOTH off-diagonal matrix positions of the symmetric conic,
    // so each position in the 2×2 gradient matrix receives half: gb/2.

    let ca = conic.x; let cb = conic.y; let cc = conic.z;
    let ga      = dL_dconic.x;
    let gb_half = dL_dconic.y * 0.5;   // split the off-diagonal gradient symmetrically
    let gc      = dL_dconic.z;

    // A = Σ⁻¹ · G_mat  where G_mat = [[ga, gb/2], [gb/2, gc]],  Σ⁻¹ = [[ca,cb],[cb,cc]]
    //   A[row,col] uses standard row-major indexing below
    //   A[0,0] = ca*ga + cb*gb_half
    //   A[0,1] = ca*gb_half + cb*gc
    //   A[1,0] = cb*ga + cc*gb_half
    //   A[1,1] = cb*gb_half + cc*gc
    //
    // ∂L/∂Σ = −A · Σ⁻¹
    //   ∂L/∂Σ[0,0] = −(A[0,0]*ca + A[0,1]*cb)
    //   ∂L/∂Σ[1,1] = −(A[1,0]*cb + A[1,1]*cc)
    //   ∂L/∂Σ[0,1] = −(A[0,0]*cb + A[0,1]*cc)   ← single off-diagonal element

    let dL_dcov2d_a = -(ca * ga + cb * gb_half) * ca
                     -(ca * gb_half + cb * gc) * cb;
    let dL_dcov2d_c = -(cb * ga + cc * gb_half) * cb
                     -(cb * gb_half + cc * gc) * cc;
    // Off-diagonal matrix element (not halved — gb_half already handled the split above)
    let dL_dcov2d_b = -(ca * ga + cb * gb_half) * cb
                     -(ca * gb_half + cb * gc) * cc;

    // Pack as vec3 [a, b, c] — symmetric 2×2 off-diagonal is stored once
    let dL_dcov2d = vec3<f32>(dL_dcov2d_a, dL_dcov2d_b, dL_dcov2d_c);

    // =========================================================================
    // BLOCK 3 — ∂L/∂cov3D via projection Jacobian
    // =========================================================================
    //
    // Forward: cov2D = T · cov3D · Tᵀ  where T = J · W  (2×3 or embedded 3×3)
    //
    // Chain rule for bilinear form A·X·Aᵀ:
    //   ∂L/∂X = Aᵀ · (∂L/∂(AXAᵀ)) · A
    //
    // So:
    //   ∂L/∂cov3D = Tᵀ · (∂L/∂cov2D_mat) · T
    //
    // We embed the 2×2 gradient into a 3×3 matrix (third row/col zero because
    // T's third row is zero — J has no contribution from the w-component).

    let J = mat3x3<f32>(
        // col 0: first column of J (J[*,0])
        vec3<f32>(fx / tz,               0.0,                      0.0),
        // col 1: second column of J
        vec3<f32>(0.0,                   fy / tz,                  0.0),
        // col 2: third column of J (depth-ratio terms)
        vec3<f32>(fx * p_view.x / tz2,   fy * p_view.y / tz2,     0.0),
    );
    let T = J * W;

    // Embed ∂L/∂cov2D as a 3×3 matrix (symmetric, last row/col = 0)
    let dL_dcov2d_mat = mat3x3<f32>(
        vec3<f32>(dL_dcov2d.x,  dL_dcov2d.y,  0.0),
        vec3<f32>(dL_dcov2d.y,  dL_dcov2d.z,  0.0),
        vec3<f32>(0.0,          0.0,           0.0),
    );

    // ∂L/∂cov3D = Tᵀ · (∂L/∂cov2D_mat) · T
    let dL_dcov3d = transpose(T) * dL_dcov2d_mat * T;

    // =========================================================================
    // BLOCK 4 — ∂L/∂(R, log_scale) via cov3D = M·Mᵀ, M = R·S
    // =========================================================================
    //
    // Forward: S = diag(exp(log_scale))
    //          M = R · S
    //          cov3D = M · Mᵀ
    //
    // Chain rule for f(M) = M·Mᵀ  (symmetric output):
    //   ∂L/∂M = (∂L/∂(MMᵀ) + ∂L/∂(MMᵀ)ᵀ) · M = 2 · (∂L/∂cov3D) · M
    //   (using the fact that ∂L/∂cov3D is itself symmetric so the two terms
    //   are identical and we factor out the 2)
    //
    // Then chain through M = R·S:
    //   ∂L/∂S_mat = Rᵀ · (∂L/∂M)
    //   ∂L/∂R     = (∂L/∂M) · Sᵀ = (∂L/∂M) · S  (diagonal)
    //
    // Diagonal of ∂L/∂S_mat gives ∂L/∂s (the un-log scales).
    // Then chain through s = exp(log_s):  ∂L/∂log_s = ∂L/∂s · s

    let R  = quat_to_mat(rot);
    let sx = exp(scale.x); let sy = exp(scale.y); let sz = exp(scale.z);
    // S as column-major mat3x3: diagonal entries are column vectors [sx,0,0], [0,sy,0], [0,0,sz]
    let S  = mat3x3<f32>(
        vec3<f32>(sx,  0.0, 0.0),
        vec3<f32>(0.0, sy,  0.0),
        vec3<f32>(0.0, 0.0, sz ),
    );
    let M  = R * S;

    // ∂L/∂M = 2 · (∂L/∂cov3D) · M
    let dL_dM = 2.0 * dL_dcov3d * M;

    // ∂L/∂R = (∂L/∂M) · Sᵀ  (Sᵀ = S since S is diagonal)
    let dL_dR = dL_dM * transpose(S);

    // ∂L/∂S_mat = Rᵀ · (∂L/∂M)
    let dL_dS_mat = transpose(R) * dL_dM;

    // ∂L/∂log_scale — extract diagonal, multiply by scale (chain through exp)
    // In WGSL column-major: mat[col][row], so diagonal is mat[0][0], mat[1][1], mat[2][2]
    let dL_dscale = vec3<f32>(
        dL_dS_mat[0][0] * sx,   // ∂L/∂log_sx = (∂L/∂sx) · sx
        dL_dS_mat[1][1] * sy,
        dL_dS_mat[2][2] * sz,
    );

    // ─────────────────────────────────────────────────────────────────────────
    // ∂L/∂quat — contract (∂L/∂R) with the Jacobian of quat_to_mat
    // ─────────────────────────────────────────────────────────────────────────
    //
    // The rotation matrix R(q) has the form given in quat_to_mat above.
    // We need  ∂L/∂q_i = Σ_{j,k} (∂R[j,k]/∂q_i) · (∂L/∂R[j,k])
    //
    // All partial derivatives are computed analytically from the quat_to_mat
    // formulas. In WGSL column-major:  dL_dR[col][row] = ∂L/∂R[row,col].
    //
    // The formulas below are equivalent to the 3DGS / diff-gaussian-rasterization
    // open-source reference (Kerbl et al., 2023).

    let qx = rot.x; let qy = rot.y; let qz = rot.z; let qw = rot.w;

    // ∂L/∂qx — contributions from all R entries that contain qx
    let dL_dqx = 2.0 * (
        // R[0,1] = xy + wz  → ∂R[0,1]/∂qx = y2 (the 2x in xy term is y*x2)
        dL_dR[0][1] * (qy)   +
        // R[0,2] = xz - wy  → ∂R[0,2]/∂qx = z2
        dL_dR[0][2] * (qz)   +
        // R[1,0] = xy - wz  → ∂R[1,0]/∂qx = y2
        dL_dR[1][0] * (qy)   +
        // R[1,1] = 1-(xx+zz) → ∂R[1,1]/∂qx = -2x2 = -2*(2qx) but factor of 2 outside
        dL_dR[1][1] * (-2.0 * qx) +
        // R[1,2] = yz + wx  → ∂R[1,2]/∂qx = w2
        dL_dR[1][2] * (qw)   +
        // R[2,0] = xz + wy  → ∂R[2,0]/∂qx = z2
        dL_dR[2][0] * (qz)   +
        // R[2,1] = yz - wx  → ∂R[2,1]/∂qx = -w2
        dL_dR[2][1] * (-qw)  +
        // R[2,2] = 1-(xx+yy) → ∂R[2,2]/∂qx = -2x2
        dL_dR[2][2] * (-2.0 * qx)
    );

    // ∂L/∂qy — contributions from R entries that contain qy
    let dL_dqy = 2.0 * (
        // R[0,0] = 1-(yy+zz) → ∂/∂qy = -2y2
        dL_dR[0][0] * (-2.0 * qy) +
        // R[0,1] = xy + wz  → ∂/∂qy = x2
        dL_dR[0][1] * (qx)   +
        // R[0,2] = xz - wy  → ∂/∂qy = -w2
        dL_dR[0][2] * (-qw)  +
        // R[1,0] = xy - wz  → ∂/∂qy = x2
        dL_dR[1][0] * (qx)   +
        // R[1,2] = yz + wx  → ∂/∂qy = z2
        dL_dR[1][2] * (qz)   +
        // R[2,0] = xz + wy  → ∂/∂qy = w2
        dL_dR[2][0] * (qw)   +
        // R[2,1] = yz - wx  → ∂/∂qy = z2
        dL_dR[2][1] * (qz)   +
        // R[2,2] = 1-(xx+yy) → ∂/∂qy = -2y2
        dL_dR[2][2] * (-2.0 * qy)
    );

    // ∂L/∂qz — contributions from R entries that contain qz
    let dL_dqz = 2.0 * (
        // R[0,0] = 1-(yy+zz) → ∂/∂qz = -2z2
        dL_dR[0][0] * (-2.0 * qz) +
        // R[0,1] = xy + wz  → ∂/∂qz = w2
        dL_dR[0][1] * (qw)   +
        // R[0,2] = xz - wy  → ∂/∂qz = x2
        dL_dR[0][2] * (qx)   +
        // R[1,0] = xy - wz  → ∂/∂qz = -w2
        dL_dR[1][0] * (-qw)  +
        // R[1,1] = 1-(xx+zz) → ∂/∂qz = -2z2
        dL_dR[1][1] * (-2.0 * qz) +
        // R[1,2] = yz + wx  → ∂/∂qz = y2
        dL_dR[1][2] * (qy)   +
        // R[2,0] = xz + wy  → ∂/∂qz = x2
        dL_dR[2][0] * (qx)   +
        // R[2,1] = yz - wx  → ∂/∂qz = y2
        dL_dR[2][1] * (qy)
    );

    // ∂L/∂qw — contributions from R entries that contain qw
    let dL_dqw = 2.0 * (
        // R[0,1] = xy + wz  → ∂/∂qw = z2
        dL_dR[0][1] * (qz)   +
        // R[0,2] = xz - wy  → ∂/∂qw = -y2
        dL_dR[0][2] * (-qy)  +
        // R[1,0] = xy - wz  → ∂/∂qw = -z2
        dL_dR[1][0] * (-qz)  +
        // R[1,2] = yz + wx  → ∂/∂qw = x2
        dL_dR[1][2] * (qx)   +
        // R[2,0] = xz + wy  → ∂/∂qw = y2
        dL_dR[2][0] * (qy)   +
        // R[2,1] = yz - wx  → ∂/∂qw = -x2
        dL_dR[2][1] * (-qx)
    );

    // =========================================================================
    // BLOCK 4b — ∂L/∂p_view from J sensitivity (covariance path only)
    // =========================================================================
    //
    // The projection Jacobian J depends on p_view = [vx, vy, vz]:
    //   J[0,0] = fx/tz,          J[1,1] = fy/tz            (tz = −vz)
    //   J[0,2] = fx·vx/tz²,      J[1,2] = fy·vy/tz²
    //   all other entries = 0
    //
    // Forward: cov2D = T · cov3D · Tᵀ,  T = J · W
    //
    // Chain rule for bilinear form f(T) = T · X · Tᵀ:
    //   ∂L/∂T = 2 · (∂L/∂cov2D_mat) · T · X
    //
    // Then:
    //   ∂L/∂J = (∂L/∂T) · Wᵀ     (from T = J·W, chain to J)
    //
    // Now map ∂L/∂J entries to ∂L/∂p_view by differentiating the J formulas:
    //
    // vx dependence (only J[0,2] = fx·vx/tz²):
    //   ∂J[0,2]/∂vx = fx/tz²
    //   → ∂L/∂vx_cov = (∂L/∂J[0,2]) · fx/tz²
    //     In WGSL col-major: J_math[row=0, col=2] = J[col=2][row=0] = dL_dJ[2][0]
    //
    // vy dependence (only J[1,2] = fy·vy/tz²):
    //   ∂J[1,2]/∂vy = fy/tz²
    //   → ∂L/∂vy_cov = (∂L/∂J[1,2]) · fy/tz²
    //     In WGSL col-major: J_math[row=1, col=2] = J[col=2][row=1] = dL_dJ[2][1]
    //
    // vz dependence (tz = −vz  →  d(·)/dvz = −d(·)/dtz):
    //   J[0,0] = fx/tz   → ∂J[0,0]/∂vz = −fx/tz²  →  (∂L/∂J[0,0])·(fx/tz²)
    //   J[1,1] = fy/tz   → ∂J[1,1]/∂vz = −fy/tz²  →  (∂L/∂J[1,1])·(fy/tz²)
    //   J[0,2] = fx·vx/tz² → ∂/∂vz = −2fx·vx/tz³  →  (∂L/∂J[0,2])·(2fx·vx/tz³)
    //   J[1,2] = fy·vy/tz² → ∂/∂vz = −2fy·vy/tz³  →  (∂L/∂J[1,2])·(2fy·vy/tz³)
    //   (the sign from d/dvz goes through the −tz relation and cancels to positive here)

    let cov3d    = M * transpose(M);
    let dL_dT    = 2.0 * dL_dcov2d_mat * T * cov3d;
    let dL_dJ    = dL_dT * transpose(W);

    // ∂L/∂vx: only J entry depending on vx is J_math[row=0, col=2] = dL_dJ[col=2][row=0]
    let dL_dvx_cov = dL_dJ[2][0] * fx / tz2;

    // ∂L/∂vy: only J entry depending on vy is J_math[row=1, col=2] = dL_dJ[col=2][row=1]
    let dL_dvy_cov = dL_dJ[2][1] * fy / tz2;

    // ∂L/∂vz: accumulated from all four J entries that depend on tz = −vz
    let dL_dvz_cov = dL_dJ[0][0] * fx / tz2        // from J[0,0] = fx/tz
                   + dL_dJ[1][1] * fy / tz2         // from J[1,1] = fy/tz
                   + dL_dJ[2][0] * 2.0 * fx * p_view.x / tz3  // from J[0,2] depth² term
                   + dL_dJ[2][1] * 2.0 * fy * p_view.y / tz3; // from J[1,2] depth² term

    let dL_dpos_cov = vec3<f32>(dL_dvx_cov, dL_dvy_cov, dL_dvz_cov);

    // ─────────────────────────────────────────────────────────────────────────
    // Return gradients
    // ─────────────────────────────────────────────────────────────────────────
    return Cov2dBwdResult(
        dL_dscale,
        vec4<f32>(dL_dqx, dL_dqy, dL_dqz, dL_dqw),
        dL_dpos_cov,
    );
}
