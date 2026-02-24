// Preprocess backward: chain 2D gradients back through projection to 3D gradients.
//
// Inputs from rasterize_bwd: grad_means2d, grad_conics, grad_colors
// Outputs: grad_positions, grad_rotations, grad_scales, grad_sh_coeffs
//
// This shader reverses the forward preprocess operations:
// 1. grad_position from grad_means2d (through projection Jacobian)
// 2. grad_cov2d from grad_conics (through matrix inverse)
// 3. grad_cov3d from grad_cov2d (through J*W projection)
// 4. grad_rotation, grad_scale from grad_cov3d (through R*S decomposition)
// 5. grad_sh_coeffs from grad_colors (through SH evaluation)

struct Uniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    cam_pos: vec3<f32>,
    _pad0: f32,
    focal: vec2<f32>,
    viewport: vec2<f32>,
    tile_grid: vec2<u32>,
    num_gaussians: u32,
    sh_degree: u32,
    near_plane: f32,
    far_plane: f32,
    _pad_bg: vec2<f32>,
    background: vec3<f32>,
    output_flags: u32,
    transmittance_threshold: f32,
    tile_size: u32,
    _pad1: vec2<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> scales: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> cov2d: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> conics: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> sh_coeffs: array<f32>;
// 2D gradients from rasterize_bwd (flat f32 arrays from atomic accumulators)
@group(0) @binding(7) var<storage, read> grad_means2d: array<f32>;
@group(0) @binding(8) var<storage, read> grad_conics: array<f32>;
@group(0) @binding(9) var<storage, read> grad_colors: array<f32>;
// 3D gradient outputs
@group(0) @binding(10) var<storage, read_write> grad_positions: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> grad_rotations: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> grad_scales: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> grad_sh_coeffs: array<f32>;

// ---- SH constants (for backward) ----
const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.4453057213202769;
const SH_C3_6: f32 = -0.5900435899266435;

fn quat_to_mat(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy)),
    );
}

@compute @workgroup_size(256)
fn preprocess_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.num_gaussians {
        return;
    }

    let pos = positions[idx].xyz;
    let rot = rotations[idx];
    let scale = scales[idx].xyz;

    // Read 2D gradients
    let dL_dmean2d = vec2<f32>(grad_means2d[idx * 2u], grad_means2d[idx * 2u + 1u]);
    let dL_dconic = vec3<f32>(grad_conics[idx * 3u], grad_conics[idx * 3u + 1u], grad_conics[idx * 3u + 2u]);
    let dL_dcolor_raw = vec3<f32>(grad_colors[idx * 3u], grad_colors[idx * 3u + 1u], grad_colors[idx * 3u + 2u]);

    // ---- 1. grad_position from grad_means2d ----
    // Forward: p_view = View * [pos, 1]; mean2d = project(p_view)
    let p_view = uniforms.view * vec4<f32>(pos, 1.0);
    let fx = uniforms.focal.x;
    let fy = uniforms.focal.y;
    let tz = -p_view.z;  // Use positive depth (RH camera: objects in front have negative z)
    let tz2 = tz * tz;

    // d(mean2d)/d(p_view) = viewport * 0.5 * d(p_ndc)/d(p_view)
    // With positive depth convention: d(mean2d.x)/d(vx) = fx/depth, d(mean2d.x)/d(vz) = fx*vx/depth²
    let W = mat3x3<f32>(
        uniforms.view[0].xyz,
        uniforms.view[1].xyz,
        uniforms.view[2].xyz,
    );

    // Gradient of mean2d w.r.t. view-space position
    let dL_dp_view = vec3<f32>(
        dL_dmean2d.x * fx / tz,
        dL_dmean2d.y * fy / tz,
        dL_dmean2d.x * (fx * p_view.x / tz2) + dL_dmean2d.y * (fy * p_view.y / tz2),
    );

    // Chain through view matrix to get world-space gradient
    var dL_dpos = transpose(W) * dL_dp_view;

    // ---- 2. grad_cov2d from grad_conics ----
    // Forward: conic = inverse(cov2d_matrix)
    // Backward: dL/dS = -S^{-1} * dL/dS^{-1} * S^{-1}  (S symmetric, S^{-1} = conic)
    let conic = conics[idx].xyz;
    let ca = conic.x; let cb = conic.y; let cc = conic.z;

    // Construct dL/d(cov2d) using the inverse derivative formula
    // dL_dS = -inv(S) * G_mat * inv(S) where inv(S) = conic matrix
    // G_mat is the gradient reshaped as a symmetric matrix.
    // dL_dconic.y (gb) is the COMBINED gradient for both off-diagonal entries of the
    // symmetric conic matrix, so each off-diagonal position gets gb/2.
    let ga = dL_dconic.x;
    let gb_half = dL_dconic.y * 0.5;  // Half the off-diagonal gradient for matrix form
    let gc = dL_dconic.z;

    // dL/d(S) = -S^{-1} * G_mat * S^{-1} where G_mat = [[ga, gb/2], [gb/2, gc]]
    // A = S^{-1} * G_mat = [[ca,cb],[cb,cc]] * [[ga,gb/2],[gb/2,gc]]
    // A[0][0] = ca*ga + cb*gb_half,  A[0][1] = ca*gb_half + cb*gc
    // A[1][0] = cb*ga + cc*gb_half,  A[1][1] = cb*gb_half + cc*gc
    // dL/d(S) = -A * S^{-1}
    let dL_da = -(ca * ga + cb * gb_half) * ca - (ca * gb_half + cb * gc) * cb;
    let dL_dc_cov = -(cb * ga + cc * gb_half) * cb - (cb * gb_half + cc * gc) * cc;
    // Off-diagonal: dL/d(S)[0,1] = -(A[0,0]*cb + A[0,1]*cc)
    let dL_db_elem = -(ca * ga + cb * gb_half) * cb - (ca * gb_half + cb * gc) * cc;

    let dL_dcov2d = vec3<f32>(dL_da, dL_db_elem, dL_dc_cov);
    // dL_db_elem is the actual off-diagonal matrix element; no further halving needed
    // because gb_half already accounts for the symmetric split

    // ---- 3. grad_cov3d from grad_cov2d ----
    // Forward: cov2d = T * cov3d * T^T where T = J * W
    let J = mat3x3<f32>(
        vec3<f32>(fx / tz, 0.0, 0.0),
        vec3<f32>(0.0, fy / tz, 0.0),
        vec3<f32>(fx * p_view.x / tz2, fy * p_view.y / tz2, 0.0),
    );
    let T = J * W;

    // dL/d(cov3d) = T^T * dL/d(cov2d_mat) * T
    // Construct the 2x2 gradient matrix from vec3 (symmetric: a, b, c)
    // We embed into 3x3 for computation (last row/col zero since T's last row is zero)
    let dL_dcov2d_mat = mat3x3<f32>(
        vec3<f32>(dL_dcov2d.x, dL_dcov2d.y, 0.0),
        vec3<f32>(dL_dcov2d.y, dL_dcov2d.z, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );

    let dL_dcov3d = transpose(T) * dL_dcov2d_mat * T;

    // ---- 4. grad_rotation, grad_scale from grad_cov3d ----
    // Forward: cov3d = M * M^T where M = R * S
    // dL/dM = (dL/d(cov3d) + dL/d(cov3d)^T) * M = 2 * dL/d(cov3d) * M (symmetric grad)
    let R = quat_to_mat(rot);
    let sx = exp(scale.x); let sy = exp(scale.y); let sz = exp(scale.z);
    let S = mat3x3<f32>(
        vec3<f32>(sx, 0.0, 0.0),
        vec3<f32>(0.0, sy, 0.0),
        vec3<f32>(0.0, 0.0, sz),
    );
    let M = R * S;

    // dL/dM = 2 * dL/dcov3d * M  (because cov3d is symmetric)
    let dL_dM = 2.0 * dL_dcov3d * M;

    // dL/dR = dL/dM * S^T = dL/dM * S (diagonal)
    let dL_dR = dL_dM * transpose(S);

    // dL/dS = R^T * dL/dM
    let dL_dS_mat = transpose(R) * dL_dM;

    // grad_scale (log-scale): dL/d(log_s) = dL/d(s) * s = diag(dL_dS) * exp(log_s)
    let dL_dscale = vec3<f32>(
        dL_dS_mat[0][0] * sx,
        dL_dS_mat[1][1] * sy,
        dL_dS_mat[2][2] * sz,
    );

    // grad_rotation (quaternion): use the formula dR/dq
    // For rotation matrix R(q), the gradient dL/dq is computed from dL/dR
    let qx = rot.x; let qy = rot.y; let qz = rot.z; let qw = rot.w;

    // Partial derivatives of R w.r.t. quaternion components, contracted with dL/dR
    let dL_dqx = 2.0 * (
        dL_dR[0][1] * (qy) + dL_dR[0][2] * (qz) +
        dL_dR[1][0] * (qy) + dL_dR[1][1] * (-2.0 * qx) + dL_dR[1][2] * (qw) +
        dL_dR[2][0] * (qz) + dL_dR[2][1] * (-qw) + dL_dR[2][2] * (-2.0 * qx)
    );
    let dL_dqy = 2.0 * (
        dL_dR[0][0] * (-2.0 * qy) + dL_dR[0][1] * (qx) + dL_dR[0][2] * (-qw) +
        dL_dR[1][0] * (qx) + dL_dR[1][2] * (qz) +
        dL_dR[2][0] * (qw) + dL_dR[2][1] * (qz) + dL_dR[2][2] * (-2.0 * qy)
    );
    let dL_dqz = 2.0 * (
        dL_dR[0][0] * (-2.0 * qz) + dL_dR[0][1] * (qw) + dL_dR[0][2] * (qx) +
        dL_dR[1][0] * (-qw) + dL_dR[1][1] * (-2.0 * qz) + dL_dR[1][2] * (qy) +
        dL_dR[2][0] * (qx) + dL_dR[2][1] * (qy)
    );
    let dL_dqw = 2.0 * (
        dL_dR[0][1] * (qz) + dL_dR[0][2] * (-qy) +
        dL_dR[1][0] * (-qz) + dL_dR[1][2] * (qx) +
        dL_dR[2][0] * (qy) + dL_dR[2][1] * (-qx)
    );

    // ---- 4b. Position gradient through covariance (dJ/dp_view contribution) ----
    // Forward: cov2d = T * cov3d * T^T where T = J(p_view) * W
    // J depends on p_view, so dL/dp_view has a contribution through cov2d
    // dL/dT = 2 * dL/dcov2d * T * cov3d (symmetric matrices)
    // dL/dJ = dL/dT * W^T
    let cov3d = M * transpose(M);
    let dL_dT_mat = 2.0 * dL_dcov2d_mat * T * cov3d;
    let dL_dJ = dL_dT_mat * transpose(W);
    let tz3 = tz2 * tz;
    // dJ/d(vx): only J_math[0][2] depends on vx → dJ[0][2]/dvx = fx/tz²
    let dL_dvx_cov = dL_dJ[2][0] * fx / tz2;
    // dJ/d(vy): only J_math[1][2] depends on vy → dJ[1][2]/dvy = fy/tz²
    let dL_dvy_cov = dL_dJ[2][1] * fy / tz2;
    // dJ/d(vz): J[0][0], J[1][1] → fx/tz², fy/tz²; J[0][2], J[1][2] → 2fx*vx/tz³, 2fy*vy/tz³
    let dL_dvz_cov = dL_dJ[0][0] * fx / tz2 + dL_dJ[1][1] * fy / tz2
                   + dL_dJ[2][0] * 2.0 * fx * p_view.x / tz3
                   + dL_dJ[2][1] * 2.0 * fy * p_view.y / tz3;
    dL_dpos += transpose(W) * vec3<f32>(dL_dvx_cov, dL_dvy_cov, dL_dvz_cov);

    // ---- 5. SH backward: grad_colors → grad_sh_coeffs ----
    let dir = normalize(pos - uniforms.cam_pos);
    let sh_per_gaussian = (uniforms.sh_degree + 1u) * (uniforms.sh_degree + 1u) * 3u;
    let sh_offset = idx * sh_per_gaussian;

    // 5a. Recompute unclamped SH color for clamp derivative
    // Forward applies: color = max(SH_eval + 0.5, 0.0)
    // Derivative of max(x, 0) = 1 if x > 0, 0 otherwise
    let dc = vec3<f32>(sh_coeffs[sh_offset], sh_coeffs[sh_offset + 1u], sh_coeffs[sh_offset + 2u]);
    var unclamped = SH_C0 * dc + 0.5;

    if uniforms.sh_degree >= 1u {
        let x = dir.x; let y = dir.y; let z = dir.z;
        unclamped += SH_C1 * (-y * vec3<f32>(sh_coeffs[sh_offset + 3u], sh_coeffs[sh_offset + 4u], sh_coeffs[sh_offset + 5u])
                            + z * vec3<f32>(sh_coeffs[sh_offset + 6u], sh_coeffs[sh_offset + 7u], sh_coeffs[sh_offset + 8u])
                            - x * vec3<f32>(sh_coeffs[sh_offset + 9u], sh_coeffs[sh_offset + 10u], sh_coeffs[sh_offset + 11u]));

        if uniforms.sh_degree >= 2u {
            let xx = x * x; let yy = y * y; let zz = z * z;
            let xy = x * y; let xz = x * z; let yz = y * z;
            let o2 = sh_offset + 12u;
            unclamped += SH_C2_0 * xy * vec3<f32>(sh_coeffs[o2], sh_coeffs[o2 + 1u], sh_coeffs[o2 + 2u])
                       + SH_C2_1 * yz * vec3<f32>(sh_coeffs[o2 + 3u], sh_coeffs[o2 + 4u], sh_coeffs[o2 + 5u])
                       + SH_C2_2 * (2.0 * zz - xx - yy) * vec3<f32>(sh_coeffs[o2 + 6u], sh_coeffs[o2 + 7u], sh_coeffs[o2 + 8u])
                       + SH_C2_3 * xz * vec3<f32>(sh_coeffs[o2 + 9u], sh_coeffs[o2 + 10u], sh_coeffs[o2 + 11u])
                       + SH_C2_4 * (xx - yy) * vec3<f32>(sh_coeffs[o2 + 12u], sh_coeffs[o2 + 13u], sh_coeffs[o2 + 14u]);

            if uniforms.sh_degree >= 3u {
                let o3 = sh_offset + 27u;
                unclamped += SH_C3_0 * y * (3.0 * xx - yy) * vec3<f32>(sh_coeffs[o3], sh_coeffs[o3 + 1u], sh_coeffs[o3 + 2u])
                           + SH_C3_1 * xy * z * vec3<f32>(sh_coeffs[o3 + 3u], sh_coeffs[o3 + 4u], sh_coeffs[o3 + 5u])
                           + SH_C3_2 * y * (4.0 * zz - xx - yy) * vec3<f32>(sh_coeffs[o3 + 6u], sh_coeffs[o3 + 7u], sh_coeffs[o3 + 8u])
                           + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * vec3<f32>(sh_coeffs[o3 + 9u], sh_coeffs[o3 + 10u], sh_coeffs[o3 + 11u])
                           + SH_C3_4 * x * (4.0 * zz - xx - yy) * vec3<f32>(sh_coeffs[o3 + 12u], sh_coeffs[o3 + 13u], sh_coeffs[o3 + 14u])
                           + SH_C3_5 * z * (xx - yy) * vec3<f32>(sh_coeffs[o3 + 15u], sh_coeffs[o3 + 16u], sh_coeffs[o3 + 17u])
                           + SH_C3_6 * x * (xx - 3.0 * yy) * vec3<f32>(sh_coeffs[o3 + 18u], sh_coeffs[o3 + 19u], sh_coeffs[o3 + 20u]);
            }
        }
    }

    // 5b. Clamp mask: forward does max(unclamped, 0.0), derivative is 1 where unclamped > 0
    let clamp_mask = vec3<f32>(
        select(0.0, 1.0, unclamped.x > 0.0),
        select(0.0, 1.0, unclamped.y > 0.0),
        select(0.0, 1.0, unclamped.z > 0.0),
    );
    let dL_dcolor = dL_dcolor_raw * clamp_mask;

    // 5c. SH coefficient gradients using masked dL_dcolor
    // Degree 0: color += SH_C0 * sh[0:3]
    grad_sh_coeffs[sh_offset + 0u] = dL_dcolor.x * SH_C0;
    grad_sh_coeffs[sh_offset + 1u] = dL_dcolor.y * SH_C0;
    grad_sh_coeffs[sh_offset + 2u] = dL_dcolor.z * SH_C0;

    if uniforms.sh_degree >= 1u {
        let x = dir.x; let y = dir.y; let z = dir.z;
        // Degree 1 coefficients: sh[3..12]
        grad_sh_coeffs[sh_offset + 3u] = dL_dcolor.x * SH_C1 * (-y);
        grad_sh_coeffs[sh_offset + 4u] = dL_dcolor.y * SH_C1 * (-y);
        grad_sh_coeffs[sh_offset + 5u] = dL_dcolor.z * SH_C1 * (-y);
        grad_sh_coeffs[sh_offset + 6u] = dL_dcolor.x * SH_C1 * z;
        grad_sh_coeffs[sh_offset + 7u] = dL_dcolor.y * SH_C1 * z;
        grad_sh_coeffs[sh_offset + 8u] = dL_dcolor.z * SH_C1 * z;
        grad_sh_coeffs[sh_offset + 9u] = dL_dcolor.x * SH_C1 * (-x);
        grad_sh_coeffs[sh_offset + 10u] = dL_dcolor.y * SH_C1 * (-x);
        grad_sh_coeffs[sh_offset + 11u] = dL_dcolor.z * SH_C1 * (-x);

        if uniforms.sh_degree >= 2u {
            let xx = x * x; let yy = y * y; let zz = z * z;
            let xy = x * y; let xz = x * z; let yz = y * z;
            let o2 = sh_offset + 12u;
            // 5 degree-2 basis functions, each contributing to R,G,B
            let b2_0 = SH_C2_0 * xy;
            let b2_1 = SH_C2_1 * yz;
            let b2_2 = SH_C2_2 * (2.0 * zz - xx - yy);
            let b2_3 = SH_C2_3 * xz;
            let b2_4 = SH_C2_4 * (xx - yy);

            for (var c = 0u; c < 3u; c++) {
                let dL_dc_i = select(select(dL_dcolor.z, dL_dcolor.y, c == 1u), dL_dcolor.x, c == 0u);
                grad_sh_coeffs[o2 + 0u * 3u + c] = dL_dc_i * b2_0;
                grad_sh_coeffs[o2 + 1u * 3u + c] = dL_dc_i * b2_1;
                grad_sh_coeffs[o2 + 2u * 3u + c] = dL_dc_i * b2_2;
                grad_sh_coeffs[o2 + 3u * 3u + c] = dL_dc_i * b2_3;
                grad_sh_coeffs[o2 + 4u * 3u + c] = dL_dc_i * b2_4;
            }

            if uniforms.sh_degree >= 3u {
                let o3 = sh_offset + 27u;
                let b3_0 = SH_C3_0 * y * (3.0 * xx - yy);
                let b3_1 = SH_C3_1 * xy * z;
                let b3_2 = SH_C3_2 * y * (4.0 * zz - xx - yy);
                let b3_3 = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
                let b3_4 = SH_C3_4 * x * (4.0 * zz - xx - yy);
                let b3_5 = SH_C3_5 * z * (xx - yy);
                let b3_6 = SH_C3_6 * x * (xx - 3.0 * yy);

                for (var c = 0u; c < 3u; c++) {
                    let dL_dc_i = select(select(dL_dcolor.z, dL_dcolor.y, c == 1u), dL_dcolor.x, c == 0u);
                    grad_sh_coeffs[o3 + 0u * 3u + c] = dL_dc_i * b3_0;
                    grad_sh_coeffs[o3 + 1u * 3u + c] = dL_dc_i * b3_1;
                    grad_sh_coeffs[o3 + 2u * 3u + c] = dL_dc_i * b3_2;
                    grad_sh_coeffs[o3 + 3u * 3u + c] = dL_dc_i * b3_3;
                    grad_sh_coeffs[o3 + 4u * 3u + c] = dL_dc_i * b3_4;
                    grad_sh_coeffs[o3 + 5u * 3u + c] = dL_dc_i * b3_5;
                    grad_sh_coeffs[o3 + 6u * 3u + c] = dL_dc_i * b3_6;
                }
            }
        }
    }

    // ---- Write outputs ----
    // Position gradient includes mean2d and cov2d contributions (both additive)
    grad_positions[idx] = vec4<f32>(dL_dpos, 0.0);
    grad_rotations[idx] = vec4<f32>(dL_dqx, dL_dqy, dL_dqz, dL_dqw);
    grad_scales[idx] = vec4<f32>(dL_dscale, 0.0);
}
