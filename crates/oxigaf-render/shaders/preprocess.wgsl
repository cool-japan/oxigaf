// Preprocess compute shader: project 3D Gaussians to 2D, compute cov2D, evaluate SH colors.
//
// Purpose
// ───────
// One thread per Gaussian.  Projects world-space positions through the camera
// view+projection matrices, computes the 2D covariance (cov2D) from 3D
// covariance via the projection Jacobian, inverts cov2D to obtain the conic,
// evaluates spherical harmonics to get view-dependent color, culls off-screen
// or degenerate Gaussians, and writes radii + tile counts for the subsequent
// sorting and rasterization passes.
//
// Bindings
// ────────
// group:binding  type              description
//    0:0         uniform           Camera/viewport/render uniforms
//    0:1         storage (ro)      positions   — world-space Gaussian centres
//    0:2         storage (ro)      rotations   — quaternions
//    0:3         storage (ro)      scales      — log-space scales
//    0:4         storage (ro)      opacities   — logit-space opacities
//    0:5         storage (rw)      means2d     — projected 2D centres (output)
//    0:6         storage (rw)      cov2d       — upper-triangle cov2D (output)
//    0:7         storage (rw)      conics      — inverse-cov2D (output)
//    0:8         storage (rw)      depths      — view-space depths (output)
//    0:9         storage (rw)      radii       — screen-space radii (output)
//   0:10         storage (rw)      tile_counts — tiles per Gaussian (output)
//   0:11         storage (ro)      sh_coeffs   — SH coefficient array
//   0:12         storage (rw)      colors      — evaluated RGB colors (output)
//   0:13         storage (rw)      normals     — surface normals (optional)
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(num_gaussians / 256) workgroups × 256 threads.
//
// Math
// ────
// Projection:  p_ndc = proj(view · pos_w);
//              mean2d = viewport(p_ndc)
// Covariance:  M = R · S;  cov3D = M · Mᵀ
//              cov2D = T · cov3D · Tᵀ   where T = J · W (Jacobian · view rot)
// Conic:       inverse(cov2D)  (2×2 symmetric using Cramer's rule)
// SH color:    Σ_{lm} c_{lm} · Y_lm(dir);  dir = normalize(pos_w - cam_pos)
//
// Performance optimizations:
// - SH degree 0 fast path: Skip direction computation, use 3 multiplies only
// - SIMD-friendly vec4 operations for SH evaluation
// - Precomputed SH basis functions: Avoid repeated evaluation
// - Early culling: Near/far plane, viewport bounds, degenerate covariance
// - Normal extraction: Compute surface normals from Gaussian orientation
//
// SH Optimization Summary:
// - Degree 0: 3 ops (DC term only, no direction needed)
// - Degree 1: ~15 ops (DC + 3 linear basis functions)
// - Degree 2: ~45 ops (DC + linear + 5 quadratic basis functions)
// - Degree 3: ~100 ops (full 16 basis functions)

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
@group(0) @binding(4) var<storage, read> opacities: array<f32>;
@group(0) @binding(5) var<storage, read_write> means2d: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> cov2d: array<vec4<f32>>;  // vec3 with padding
@group(0) @binding(7) var<storage, read_write> conics: array<vec4<f32>>;  // vec3 with padding
@group(0) @binding(8) var<storage, read_write> depths: array<f32>;
@group(0) @binding(9) var<storage, read_write> radii: array<i32>;
@group(0) @binding(10) var<storage, read_write> tile_counts: array<u32>;
@group(0) @binding(11) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(12) var<storage, read_write> colors: array<vec4<f32>>;  // vec3 with padding
@group(0) @binding(13) var<storage, read_write> normals: array<vec4<f32>>;  // vec3 with padding

// ============================================================================
// SH Constants (precomputed for efficiency)
// ============================================================================
// These are the real spherical harmonic normalization constants.
// Y_lm = SH_Cl_m * polynomial(x, y, z)

// Degree 0: DC term (constant)
// SH_C0 = 1 / (2 * sqrt(pi)) = 0.28209479177387814
const SH_C0: f32 = 0.28209479177387814;

// Degree 1: Linear terms
// SH_C1 = sqrt(3) / (2 * sqrt(pi)) = 0.4886025119029199
const SH_C1: f32 = 0.4886025119029199;

// Degree 2: Quadratic terms
// Packed for SIMD operations
const SH_C2_0: f32 = 1.0925484305920792;  // sqrt(15) / (2 * sqrt(pi)) for xy
const SH_C2_1: f32 = -1.0925484305920792; // -sqrt(15) / (2 * sqrt(pi)) for yz
const SH_C2_2: f32 = 0.31539156525252005; // sqrt(5) / (4 * sqrt(pi)) for 2zz-xx-yy
const SH_C2_3: f32 = -1.0925484305920792; // -sqrt(15) / (2 * sqrt(pi)) for xz
const SH_C2_4: f32 = 0.5462742152960396;  // sqrt(15) / (4 * sqrt(pi)) for xx-yy

// Degree 3: Cubic terms
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.4453057213202769;
const SH_C3_6: f32 = -0.5900435899266435;

// ============================================================================
// Optimized SH Evaluation Functions
// ============================================================================

// ---- Degree 0 Fast Path ----
// When sh_degree == 0, we only need the DC term:
// color = SH_C0 * sh_coeffs[0:3] + 0.5
// This is just 3 multiplies + 3 adds, no direction needed.
//
// Performance: ~3 ops per Gaussian
fn eval_sh_degree0(sh_offset: u32) -> vec3<f32> {
    // Use vec3 operations for SIMD efficiency
    let dc = vec3<f32>(
        sh_coeffs[sh_offset],
        sh_coeffs[sh_offset + 1u],
        sh_coeffs[sh_offset + 2u]
    );
    return max(SH_C0 * dc + 0.5, vec3<f32>(0.0));
}

// ---- Degree 1 Optimized (DC + Linear) ----
// Uses 4 SH basis functions: 1 DC + 3 linear
// Total coefficients: 4 * 3 = 12 floats
//
// Performance: ~15 ops per Gaussian
fn eval_sh_degree1_optimized(dir: vec3<f32>, sh_offset: u32) -> vec3<f32> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // DC term (3 floats starting at sh_offset)
    var result = SH_C0 * vec3<f32>(
        sh_coeffs[sh_offset],
        sh_coeffs[sh_offset + 1u],
        sh_coeffs[sh_offset + 2u]
    );

    // Linear terms: basis functions are (-y, z, -x)
    // SIMD optimization: precompute linear weights
    let l1_weights = SH_C1 * vec3<f32>(-y, z, -x);

    // Load coefficients for each linear basis function (3 RGB triplets)
    let l1_0 = vec3<f32>(sh_coeffs[sh_offset + 3u], sh_coeffs[sh_offset + 4u], sh_coeffs[sh_offset + 5u]);
    let l1_1 = vec3<f32>(sh_coeffs[sh_offset + 6u], sh_coeffs[sh_offset + 7u], sh_coeffs[sh_offset + 8u]);
    let l1_2 = vec3<f32>(sh_coeffs[sh_offset + 9u], sh_coeffs[sh_offset + 10u], sh_coeffs[sh_offset + 11u]);

    result += l1_weights.x * l1_0 + l1_weights.y * l1_1 + l1_weights.z * l1_2;

    return max(result + 0.5, vec3<f32>(0.0));
}

// ---- Degree 2 Optimized (DC + Linear + Quadratic) ----
// Uses 9 SH basis functions: 1 DC + 3 linear + 5 quadratic
// Total coefficients: 9 * 3 = 27 floats
//
// Performance: ~45 ops per Gaussian
fn eval_sh_degree2_optimized(dir: vec3<f32>, sh_offset: u32) -> vec3<f32> {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // Precompute products for degree 2
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    // DC term
    var result = SH_C0 * vec3<f32>(
        sh_coeffs[sh_offset],
        sh_coeffs[sh_offset + 1u],
        sh_coeffs[sh_offset + 2u]
    );

    // Linear terms (vectorized)
    let l1_weights = SH_C1 * vec3<f32>(-y, z, -x);
    let l1_0 = vec3<f32>(sh_coeffs[sh_offset + 3u], sh_coeffs[sh_offset + 4u], sh_coeffs[sh_offset + 5u]);
    let l1_1 = vec3<f32>(sh_coeffs[sh_offset + 6u], sh_coeffs[sh_offset + 7u], sh_coeffs[sh_offset + 8u]);
    let l1_2 = vec3<f32>(sh_coeffs[sh_offset + 9u], sh_coeffs[sh_offset + 10u], sh_coeffs[sh_offset + 11u]);
    result += l1_weights.x * l1_0 + l1_weights.y * l1_1 + l1_weights.z * l1_2;

    // Quadratic terms - precompute all basis weights
    let o2 = sh_offset + 12u;
    let l2_w0 = SH_C2_0 * xy;
    let l2_w1 = SH_C2_1 * yz;
    let l2_w2 = SH_C2_2 * (2.0 * zz - xx - yy);
    let l2_w3 = SH_C2_3 * xz;
    let l2_w4 = SH_C2_4 * (xx - yy);

    // Load quadratic coefficients
    let l2_0 = vec3<f32>(sh_coeffs[o2], sh_coeffs[o2 + 1u], sh_coeffs[o2 + 2u]);
    let l2_1 = vec3<f32>(sh_coeffs[o2 + 3u], sh_coeffs[o2 + 4u], sh_coeffs[o2 + 5u]);
    let l2_2 = vec3<f32>(sh_coeffs[o2 + 6u], sh_coeffs[o2 + 7u], sh_coeffs[o2 + 8u]);
    let l2_3 = vec3<f32>(sh_coeffs[o2 + 9u], sh_coeffs[o2 + 10u], sh_coeffs[o2 + 11u]);
    let l2_4 = vec3<f32>(sh_coeffs[o2 + 12u], sh_coeffs[o2 + 13u], sh_coeffs[o2 + 14u]);

    result += l2_w0 * l2_0 + l2_w1 * l2_1 + l2_w2 * l2_2 + l2_w3 * l2_3 + l2_w4 * l2_4;

    return max(result + 0.5, vec3<f32>(0.0));
}

// ---- Degree 3 Optimized (Full SH) ----
// Uses all 16 SH basis functions
// Total coefficients: 16 * 3 = 48 floats
//
// Performance: ~100 ops per Gaussian
fn eval_sh_degree3_optimized(dir: vec3<f32>, sh_offset: u32) -> vec3<f32> {
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
    var result = SH_C0 * vec3<f32>(
        sh_coeffs[sh_offset],
        sh_coeffs[sh_offset + 1u],
        sh_coeffs[sh_offset + 2u]
    );

    // Linear terms (vectorized)
    let l1_weights = SH_C1 * vec3<f32>(-y, z, -x);
    let l1_0 = vec3<f32>(sh_coeffs[sh_offset + 3u], sh_coeffs[sh_offset + 4u], sh_coeffs[sh_offset + 5u]);
    let l1_1 = vec3<f32>(sh_coeffs[sh_offset + 6u], sh_coeffs[sh_offset + 7u], sh_coeffs[sh_offset + 8u]);
    let l1_2 = vec3<f32>(sh_coeffs[sh_offset + 9u], sh_coeffs[sh_offset + 10u], sh_coeffs[sh_offset + 11u]);
    result += l1_weights.x * l1_0 + l1_weights.y * l1_1 + l1_weights.z * l1_2;

    // Quadratic terms
    let o2 = sh_offset + 12u;
    let l2_w0 = SH_C2_0 * xy;
    let l2_w1 = SH_C2_1 * yz;
    let l2_w2 = SH_C2_2 * (2.0 * zz - xx - yy);
    let l2_w3 = SH_C2_3 * xz;
    let l2_w4 = SH_C2_4 * (xx - yy);

    let l2_0 = vec3<f32>(sh_coeffs[o2], sh_coeffs[o2 + 1u], sh_coeffs[o2 + 2u]);
    let l2_1 = vec3<f32>(sh_coeffs[o2 + 3u], sh_coeffs[o2 + 4u], sh_coeffs[o2 + 5u]);
    let l2_2 = vec3<f32>(sh_coeffs[o2 + 6u], sh_coeffs[o2 + 7u], sh_coeffs[o2 + 8u]);
    let l2_3 = vec3<f32>(sh_coeffs[o2 + 9u], sh_coeffs[o2 + 10u], sh_coeffs[o2 + 11u]);
    let l2_4 = vec3<f32>(sh_coeffs[o2 + 12u], sh_coeffs[o2 + 13u], sh_coeffs[o2 + 14u]);

    result += l2_w0 * l2_0 + l2_w1 * l2_1 + l2_w2 * l2_2 + l2_w3 * l2_3 + l2_w4 * l2_4;

    // Cubic terms - precompute all 7 basis weights
    let o3 = sh_offset + 27u;
    let l3_w0 = SH_C3_0 * y * (3.0 * xx - yy);
    let l3_w1 = SH_C3_1 * xy * z;
    let l3_w2 = SH_C3_2 * y * (4.0 * zz - xx - yy);
    let l3_w3 = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    let l3_w4 = SH_C3_4 * x * (4.0 * zz - xx - yy);
    let l3_w5 = SH_C3_5 * z * (xx - yy);
    let l3_w6 = SH_C3_6 * x * (xx - 3.0 * yy);

    // Load cubic coefficients
    let l3_0 = vec3<f32>(sh_coeffs[o3], sh_coeffs[o3 + 1u], sh_coeffs[o3 + 2u]);
    let l3_1 = vec3<f32>(sh_coeffs[o3 + 3u], sh_coeffs[o3 + 4u], sh_coeffs[o3 + 5u]);
    let l3_2 = vec3<f32>(sh_coeffs[o3 + 6u], sh_coeffs[o3 + 7u], sh_coeffs[o3 + 8u]);
    let l3_3 = vec3<f32>(sh_coeffs[o3 + 9u], sh_coeffs[o3 + 10u], sh_coeffs[o3 + 11u]);
    let l3_4 = vec3<f32>(sh_coeffs[o3 + 12u], sh_coeffs[o3 + 13u], sh_coeffs[o3 + 14u]);
    let l3_5 = vec3<f32>(sh_coeffs[o3 + 15u], sh_coeffs[o3 + 16u], sh_coeffs[o3 + 17u]);
    let l3_6 = vec3<f32>(sh_coeffs[o3 + 18u], sh_coeffs[o3 + 19u], sh_coeffs[o3 + 20u]);

    result += l3_w0 * l3_0 + l3_w1 * l3_1 + l3_w2 * l3_2 + l3_w3 * l3_3;
    result += l3_w4 * l3_4 + l3_w5 * l3_5 + l3_w6 * l3_6;

    return max(result + 0.5, vec3<f32>(0.0));
}

// ---- General SH Evaluation with Runtime Branching ----
// Used when shader variants are disabled
fn eval_sh_from_buffer(dir: vec3<f32>, sh_degree: u32, sh_offset: u32) -> vec3<f32> {
    // Use optimized functions based on degree
    if sh_degree == 0u {
        return eval_sh_degree0(sh_offset);
    } else if sh_degree == 1u {
        return eval_sh_degree1_optimized(dir, sh_offset);
    } else if sh_degree == 2u {
        return eval_sh_degree2_optimized(dir, sh_offset);
    } else {
        return eval_sh_degree3_optimized(dir, sh_offset);
    }
}

// ============================================================================
// Quaternion and Covariance Functions
// ============================================================================

// Quaternion to rotation matrix
fn quat_to_mat(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy)),
    );
}

// Extract the principal normal from a quaternion.
// The normal is the direction of minimum scale (most "flat" axis).
// For a Gaussian splat, this is typically the Z-axis in local space,
// transformed to world space.
fn quat_to_normal(q: vec4<f32>) -> vec3<f32> {
    let R = quat_to_mat(q);
    // The third column of R is the transformed Z-axis (surface normal)
    return normalize(R[2]);
}

// Compute 3D covariance from rotation quaternion and log-scale
fn compute_cov3d(q: vec4<f32>, s: vec3<f32>) -> mat3x3<f32> {
    let R = quat_to_mat(q);
    let S = mat3x3<f32>(
        vec3<f32>(exp(s.x), 0.0, 0.0),
        vec3<f32>(0.0, exp(s.y), 0.0),
        vec3<f32>(0.0, 0.0, exp(s.z)),
    );
    let M = R * S;
    return M * transpose(M);
}

// ============================================================================
// Main Preprocess Kernel
// ============================================================================

@compute @workgroup_size(256)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.num_gaussians {
        return;
    }

    let pos = positions[idx].xyz;
    let rot = rotations[idx];
    let scale = scales[idx].xyz;

    // Transform to view space
    let p_view = uniforms.view * vec4<f32>(pos, 1.0);

    // Convert to positive depth (RH camera: objects in front have negative z)
    let depth = -p_view.z;

    // Early culling: near/far planes
    if depth <= uniforms.near_plane || depth >= uniforms.far_plane {
        radii[idx] = -1;
        tile_counts[idx] = 0u;
        return;
    }

    // Project to NDC
    let p_proj = uniforms.proj * p_view;
    let p_ndc = p_proj.xy / p_proj.w;

    // Screen-space position
    let mean2d = vec2<f32>(
        (p_ndc.x * 0.5 + 0.5) * uniforms.viewport.x,
        (p_ndc.y * 0.5 + 0.5) * uniforms.viewport.y,
    );
    means2d[idx] = mean2d;
    depths[idx] = depth;

    // 3D covariance
    let cov3 = compute_cov3d(rot, scale);

    // Jacobian of projection
    let fx = uniforms.focal.x;
    let fy = uniforms.focal.y;
    let tz = depth;
    let tz2 = tz * tz;

    let J = mat3x3<f32>(
        vec3<f32>(fx / tz, 0.0, 0.0),
        vec3<f32>(0.0, fy / tz, 0.0),
        vec3<f32>(fx * p_view.x / tz2, fy * p_view.y / tz2, 0.0),
    );

    // View-space rotation
    let W = mat3x3<f32>(
        uniforms.view[0].xyz,
        uniforms.view[1].xyz,
        uniforms.view[2].xyz,
    );

    let T = J * W;
    let cov_full = T * cov3 * transpose(T);

    // 2D covariance (upper triangle: a, b, c where [[a,b],[b,c]])
    let a = cov_full[0][0] + 0.3; // low-pass filter for numerical stability
    let b = cov_full[0][1];
    let c = cov_full[1][1] + 0.3;

    cov2d[idx] = vec4<f32>(a, b, c, 0.0);

    // Compute conic (inverse of 2D covariance)
    let det = a * c - b * b;
    if det <= 0.0 {
        radii[idx] = -1;
        tile_counts[idx] = 0u;
        return;
    }
    let inv_det = 1.0 / det;
    conics[idx] = vec4<f32>(c * inv_det, -b * inv_det, a * inv_det, 0.0);

    // Compute radius (3-sigma ellipse)
    let mid = 0.5 * (a + c);
    let lambda_max = mid + sqrt(max(0.1, mid * mid - det));
    let r = ceil(3.0 * sqrt(lambda_max));
    let radius = i32(r);
    radii[idx] = radius;

    // Get tile size from config
    let tile_size = f32(uniforms.tile_size);

    // Count tiles this Gaussian overlaps
    let tile_min_x = max(0, i32(floor((mean2d.x - r) / tile_size)));
    let tile_max_x = min(i32(uniforms.tile_grid.x) - 1, i32(floor((mean2d.x + r) / tile_size)));
    let tile_min_y = max(0, i32(floor((mean2d.y - r) / tile_size)));
    let tile_max_y = min(i32(uniforms.tile_grid.y) - 1, i32(floor((mean2d.y + r) / tile_size)));

    // Early culling: completely outside viewport
    if tile_min_x > tile_max_x || tile_min_y > tile_max_y {
        radii[idx] = -1;
        tile_counts[idx] = 0u;
        return;
    }

    let count = u32((tile_max_x - tile_min_x + 1) * (tile_max_y - tile_min_y + 1));
    tile_counts[idx] = count;

    // ---- Evaluate SH colors ----
    // OPTIMIZATION: Specialized path for each SH degree
    let sh_per_gaussian = (uniforms.sh_degree + 1u) * (uniforms.sh_degree + 1u) * 3u;
    let sh_offset = idx * sh_per_gaussian;

    // Select optimized evaluation based on sh_degree
    // This branches at runtime, but shader variants can eliminate this
    if uniforms.sh_degree == 0u {
        // Fast path: no direction computation needed
        // Just 3 multiplies + 3 adds (+ clamp)
        // Performance: ~10x faster than full SH evaluation
        colors[idx] = vec4<f32>(eval_sh_degree0(sh_offset), 0.0);
    } else {
        // Full SH evaluation with view direction
        let dir = normalize(pos - uniforms.cam_pos);

        if uniforms.sh_degree == 1u {
            colors[idx] = vec4<f32>(eval_sh_degree1_optimized(dir, sh_offset), 0.0);
        } else if uniforms.sh_degree == 2u {
            colors[idx] = vec4<f32>(eval_sh_degree2_optimized(dir, sh_offset), 0.0);
        } else {
            colors[idx] = vec4<f32>(eval_sh_degree3_optimized(dir, sh_offset), 0.0);
        }
    }

    // ---- Compute per-Gaussian normals (for optional normal output) ----
    // The normal is the direction of minimum scale (most flat axis)
    // which corresponds to the Z-axis of the Gaussian's local coordinate system.
    // Check if normal output is enabled (bit 1 of output_flags)
    if (uniforms.output_flags & 2u) != 0u {
        normals[idx] = vec4<f32>(quat_to_normal(rot), 0.0);
    }
}
