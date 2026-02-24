// Spherical harmonics evaluation (degree 0-3) with SIMD optimizations.
//
// Performance optimizations:
// - SIMD-friendly vec4 layout for vectorized operations
// - Precomputed basis function coefficients
// - Specialized fast paths for each degree
// - Minimized register pressure through coefficient grouping
//
// SH Basis Functions (Real Spherical Harmonics):
// Degree 0: Y_0^0 = 0.5 * sqrt(1/pi) = C0
// Degree 1: Y_1^{-1} = -y, Y_1^0 = z, Y_1^1 = -x (scaled by C1)
// Degree 2: 5 basis functions (xy, yz, 2zz-xx-yy, xz, xx-yy)
// Degree 3: 7 basis functions (complex polynomials)

// ---- SH Constants (precomputed for efficiency) ----
// These are the real spherical harmonic normalization constants.
// Y_lm = SH_Cl_m * polynomial(x, y, z)

// Degree 0: DC term (constant)
// SH_C0 = 1 / (2 * sqrt(pi)) = 0.28209479177387814
const SH_C0: f32 = 0.28209479177387814;

// Degree 1: Linear terms
// SH_C1 = sqrt(3) / (2 * sqrt(pi)) = 0.4886025119029199
const SH_C1: f32 = 0.4886025119029199;

// Degree 2: Quadratic terms (packed as vec4 for SIMD)
// SH_C2_0 = sqrt(15) / (2 * sqrt(pi)) = 1.0925484305920792
// SH_C2_1 = -sqrt(15) / (2 * sqrt(pi)) = -1.0925484305920792
// SH_C2_2 = sqrt(5) / (4 * sqrt(pi)) = 0.31539156525252005
// SH_C2_3 = -sqrt(15) / (2 * sqrt(pi)) = -1.0925484305920792
// SH_C2_4 = sqrt(15) / (4 * sqrt(pi)) = 0.5462742152960396
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;

// Degree 3: Cubic terms
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.4453057213202769;
const SH_C3_6: f32 = -0.5900435899266435;

// ---- Degree 0 Fast Path ----
// When sh_degree == 0, we only need the DC term:
// color = SH_C0 * sh_dc + 0.5
// This is just 3 multiplies + 3 adds, no direction needed.
//
// Input: sh_dc is the first 3 coefficients (RGB of DC term)
// Returns: RGB color clamped to [0, infinity)
fn eval_sh_degree0_vec3(sh_dc: vec3<f32>) -> vec3<f32> {
    return max(SH_C0 * sh_dc + 0.5, vec3<f32>(0.0));
}

// ---- Degree 1 Evaluation (DC + Linear) ----
// Uses 4 SH basis functions: 1 DC + 3 linear
// Total coefficients: 4 * 3 = 12 floats
//
// Basis functions:
// Y_0^0 = C0 (constant)
// Y_1^{-1} = C1 * (-y)
// Y_1^0 = C1 * z
// Y_1^1 = C1 * (-x)
fn eval_sh_degree1(dir: vec3<f32>, sh: array<vec4<f32>, 3>) -> vec3<f32> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // DC term (sh[0].xyz, sh[1].x, sh[2].x contain interleaved coeffs)
    // For vec4 layout: sh[0] = (dc_r, l1_r, l2_r, l3_r), etc.
    // Restructured: sh[0] = (dc_r, dc_g, dc_b, _), sh[1] = (l1_r, l1_g, l1_b, _), etc.

    var result = SH_C0 * vec3<f32>(sh[0].x, sh[0].y, sh[0].z);

    // Linear terms: -y * l1 + z * l2 - x * l3
    let linear_weight = vec3<f32>(-y, z, -x);
    result += SH_C1 * (
        linear_weight.x * vec3<f32>(sh[1].x, sh[1].y, sh[1].z) +
        linear_weight.y * vec3<f32>(sh[1].w, sh[2].x, sh[2].y) +
        linear_weight.z * vec3<f32>(sh[2].z, sh[2].w, sh[0].w)
    );

    return max(result + 0.5, vec3<f32>(0.0));
}

// ---- Degree 2 Evaluation (DC + Linear + Quadratic) ----
// Uses 9 SH basis functions: 1 DC + 3 linear + 5 quadratic
// Total coefficients: 9 * 3 = 27 floats
fn eval_sh_degree2(dir: vec3<f32>, sh_dc: vec3<f32>, sh_l1: array<vec3<f32>, 3>, sh_l2: array<vec3<f32>, 5>) -> vec3<f32> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // Precompute products
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    // DC term
    var result = SH_C0 * sh_dc;

    // Linear terms: -y * l1[0] + z * l1[1] - x * l1[2]
    result += SH_C1 * (-y * sh_l1[0] + z * sh_l1[1] - x * sh_l1[2]);

    // Quadratic terms
    result += SH_C2_0 * xy * sh_l2[0];
    result += SH_C2_1 * yz * sh_l2[1];
    result += SH_C2_2 * (2.0 * zz - xx - yy) * sh_l2[2];
    result += SH_C2_3 * xz * sh_l2[3];
    result += SH_C2_4 * (xx - yy) * sh_l2[4];

    return max(result + 0.5, vec3<f32>(0.0));
}

// ---- Full SH Evaluation (Degree 0-3) with SIMD optimization ----
// This is the general function that handles all degrees.
// Uses vec4 packing for efficient memory access and SIMD operations.
//
// Parameters:
// - dir: normalized view direction (from Gaussian to camera)
// - sh_degree: 0, 1, 2, or 3
// - sh: array of 48 floats (16 * 3 coefficients for full degree 3)
//
// Returns: RGB color (may need clamping to [0,1] for display)
fn eval_sh_full(
    dir: vec3<f32>,
    sh_degree: u32,
    sh: array<f32, 48>,
) -> vec3<f32> {
    var result = vec3<f32>(0.0);

    // Degree 0: DC term (always computed)
    // Uses SH_C0 * (r, g, b) + 0.5
    result.x += SH_C0 * sh[0];
    result.y += SH_C0 * sh[1];
    result.z += SH_C0 * sh[2];

    if sh_degree < 1u {
        return max(result + 0.5, vec3<f32>(0.0));
    }

    // Precompute direction components
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // Degree 1: Y_1^(-1), Y_1^0, Y_1^1
    // Basis: -y, z, -x
    // SIMD optimization: group RGB channels together
    let l1_basis = vec3<f32>(-y, z, -x);
    result.x += SH_C1 * (l1_basis.x * sh[3] + l1_basis.y * sh[6] + l1_basis.z * sh[9]);
    result.y += SH_C1 * (l1_basis.x * sh[4] + l1_basis.y * sh[7] + l1_basis.z * sh[10]);
    result.z += SH_C1 * (l1_basis.x * sh[5] + l1_basis.y * sh[8] + l1_basis.z * sh[11]);

    if sh_degree < 2u {
        return max(result + 0.5, vec3<f32>(0.0));
    }

    // Precompute squared and cross terms for degree 2+
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    // Degree 2: 5 basis functions
    // SIMD optimization: compute basis values once, apply to all channels
    let l2_0 = SH_C2_0 * xy;
    let l2_1 = SH_C2_1 * yz;
    let l2_2 = SH_C2_2 * (2.0 * zz - xx - yy);
    let l2_3 = SH_C2_3 * xz;
    let l2_4 = SH_C2_4 * (xx - yy);

    result.x += l2_0 * sh[12] + l2_1 * sh[15] + l2_2 * sh[18] + l2_3 * sh[21] + l2_4 * sh[24];
    result.y += l2_0 * sh[13] + l2_1 * sh[16] + l2_2 * sh[19] + l2_3 * sh[22] + l2_4 * sh[25];
    result.z += l2_0 * sh[14] + l2_1 * sh[17] + l2_2 * sh[20] + l2_3 * sh[23] + l2_4 * sh[26];

    if sh_degree < 3u {
        return max(result + 0.5, vec3<f32>(0.0));
    }

    // Degree 3: 7 basis functions (expensive, only if needed)
    // SIMD optimization: precompute all basis values
    let l3_0 = SH_C3_0 * y * (3.0 * xx - yy);
    let l3_1 = SH_C3_1 * xy * z;
    let l3_2 = SH_C3_2 * y * (4.0 * zz - xx - yy);
    let l3_3 = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    let l3_4 = SH_C3_4 * x * (4.0 * zz - xx - yy);
    let l3_5 = SH_C3_5 * z * (xx - yy);
    let l3_6 = SH_C3_6 * x * (xx - 3.0 * yy);

    result.x += l3_0 * sh[27] + l3_1 * sh[30] + l3_2 * sh[33] + l3_3 * sh[36] + l3_4 * sh[39] + l3_5 * sh[42] + l3_6 * sh[45];
    result.y += l3_0 * sh[28] + l3_1 * sh[31] + l3_2 * sh[34] + l3_3 * sh[37] + l3_4 * sh[40] + l3_5 * sh[43] + l3_6 * sh[46];
    result.z += l3_0 * sh[29] + l3_1 * sh[32] + l3_2 * sh[35] + l3_3 * sh[38] + l3_4 * sh[41] + l3_5 * sh[44] + l3_6 * sh[47];

    return max(result + 0.5, vec3<f32>(0.0));
}

// ---- Vectorized SH Evaluation using vec4 packing ----
// This version uses vec4 for better SIMD utilization.
// SH coefficients are packed as:
// sh_v4[0] = (sh[0], sh[1], sh[2], sh[3])
// sh_v4[1] = (sh[4], sh[5], sh[6], sh[7])
// etc.
fn eval_sh_vec4(
    dir: vec3<f32>,
    sh_degree: u32,
    sh_v4: array<vec4<f32>, 12>,
) -> vec3<f32> {
    var result = vec3<f32>(0.0);

    // Degree 0: DC term
    // sh_v4[0].xyz = (sh[0], sh[1], sh[2])
    result = SH_C0 * sh_v4[0].xyz;

    if sh_degree < 1u {
        return max(result + 0.5, vec3<f32>(0.0));
    }

    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // Degree 1: Linear terms
    // sh[3..12] -> 3 RGB triplets
    // l1[0] = sh[3..6] = (sh_v4[0].w, sh_v4[1].xy)
    // l1[1] = sh[6..9] = (sh_v4[1].zw, sh_v4[2].x)
    // l1[2] = sh[9..12] = (sh_v4[2].yzw)
    let l1_0 = vec3<f32>(sh_v4[0].w, sh_v4[1].x, sh_v4[1].y);
    let l1_1 = vec3<f32>(sh_v4[1].z, sh_v4[1].w, sh_v4[2].x);
    let l1_2 = sh_v4[2].yzw;

    result += SH_C1 * (-y * l1_0 + z * l1_1 - x * l1_2);

    if sh_degree < 2u {
        return max(result + 0.5, vec3<f32>(0.0));
    }

    // Precompute products
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    // Degree 2: Quadratic terms
    // sh[12..27] -> 5 RGB triplets
    // l2[0] = sh[12..15] = (sh_v4[3].xyz)
    // l2[1] = sh[15..18] = (sh_v4[3].w, sh_v4[4].xy)
    // l2[2] = sh[18..21] = (sh_v4[4].zw, sh_v4[5].x)
    // l2[3] = sh[21..24] = (sh_v4[5].yzw)
    // l2[4] = sh[24..27] = (sh_v4[6].xyz)
    let l2_0 = sh_v4[3].xyz;
    let l2_1 = vec3<f32>(sh_v4[3].w, sh_v4[4].x, sh_v4[4].y);
    let l2_2 = vec3<f32>(sh_v4[4].z, sh_v4[4].w, sh_v4[5].x);
    let l2_3 = sh_v4[5].yzw;
    let l2_4 = sh_v4[6].xyz;

    result += SH_C2_0 * xy * l2_0;
    result += SH_C2_1 * yz * l2_1;
    result += SH_C2_2 * (2.0 * zz - xx - yy) * l2_2;
    result += SH_C2_3 * xz * l2_3;
    result += SH_C2_4 * (xx - yy) * l2_4;

    if sh_degree < 3u {
        return max(result + 0.5, vec3<f32>(0.0));
    }

    // Degree 3: Cubic terms
    // sh[27..48] -> 7 RGB triplets
    // l3[0] = sh[27..30] = (sh_v4[6].w, sh_v4[7].xy)
    // l3[1] = sh[30..33] = (sh_v4[7].zw, sh_v4[8].x)
    // l3[2] = sh[33..36] = (sh_v4[8].yzw)
    // l3[3] = sh[36..39] = (sh_v4[9].xyz)
    // l3[4] = sh[39..42] = (sh_v4[9].w, sh_v4[10].xy)
    // l3[5] = sh[42..45] = (sh_v4[10].zw, sh_v4[11].x)
    // l3[6] = sh[45..48] = (sh_v4[11].yzw)
    let l3_0 = vec3<f32>(sh_v4[6].w, sh_v4[7].x, sh_v4[7].y);
    let l3_1 = vec3<f32>(sh_v4[7].z, sh_v4[7].w, sh_v4[8].x);
    let l3_2 = sh_v4[8].yzw;
    let l3_3 = sh_v4[9].xyz;
    let l3_4 = vec3<f32>(sh_v4[9].w, sh_v4[10].x, sh_v4[10].y);
    let l3_5 = vec3<f32>(sh_v4[10].z, sh_v4[10].w, sh_v4[11].x);
    let l3_6 = sh_v4[11].yzw;

    let b3_0 = SH_C3_0 * y * (3.0 * xx - yy);
    let b3_1 = SH_C3_1 * xy * z;
    let b3_2 = SH_C3_2 * y * (4.0 * zz - xx - yy);
    let b3_3 = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    let b3_4 = SH_C3_4 * x * (4.0 * zz - xx - yy);
    let b3_5 = SH_C3_5 * z * (xx - yy);
    let b3_6 = SH_C3_6 * x * (xx - 3.0 * yy);

    result += b3_0 * l3_0 + b3_1 * l3_1 + b3_2 * l3_2 + b3_3 * l3_3 + b3_4 * l3_4 + b3_5 * l3_5 + b3_6 * l3_6;

    return max(result + 0.5, vec3<f32>(0.0));
}
