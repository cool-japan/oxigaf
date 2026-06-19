// Preprocess compute shader — SH Degree 0 Optimized Variant
//
// Purpose
// ───────
// Specialised preprocess shader for sh_degree=0 (DC color term only).
// Eliminates all direction computation and degree-1/2/3 SH evaluation,
// reducing the per-Gaussian shader to the minimal ops needed for a static
// Gaussian color (approximately 3 multiplies instead of ~100).
//
// Bindings / Dispatch / Math: identical to preprocess.wgsl.
// See preprocess.wgsl for the authoritative documentation.
//
// This shader variant is specialized for sh_degree=0 (DC term only).
// All branching has been eliminated for maximum performance.
//
// Performance: ~3 ops for SH evaluation (vs ~100 for full degree 3)
// - No direction computation needed
// - Just 3 multiplies + 3 adds for color

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
@group(0) @binding(6) var<storage, read_write> cov2d: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> conics: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> depths: array<f32>;
@group(0) @binding(9) var<storage, read_write> radii: array<i32>;
@group(0) @binding(10) var<storage, read_write> tile_counts: array<u32>;
@group(0) @binding(11) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(12) var<storage, read_write> colors: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> normals: array<vec4<f32>>;

// SH Degree 0: DC term only
// SH_C0 = 1 / (2 * sqrt(pi))
const SH_C0: f32 = 0.28209479177387814;

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

fn quat_to_normal(q: vec4<f32>) -> vec3<f32> {
    let R = quat_to_mat(q);
    return normalize(R[2]);
}

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

    let W = mat3x3<f32>(
        uniforms.view[0].xyz,
        uniforms.view[1].xyz,
        uniforms.view[2].xyz,
    );

    let T = J * W;
    let cov_full = T * cov3 * transpose(T);

    let a = cov_full[0][0] + 0.3;
    let b = cov_full[0][1];
    let c = cov_full[1][1] + 0.3;

    cov2d[idx] = vec4<f32>(a, b, c, 0.0);

    let det = a * c - b * b;
    if det <= 0.0 {
        radii[idx] = -1;
        tile_counts[idx] = 0u;
        return;
    }
    let inv_det = 1.0 / det;
    conics[idx] = vec4<f32>(c * inv_det, -b * inv_det, a * inv_det, 0.0);

    let mid = 0.5 * (a + c);
    let lambda_max = mid + sqrt(max(0.1, mid * mid - det));
    let r = ceil(3.0 * sqrt(lambda_max));
    let radius = i32(r);
    radii[idx] = radius;

    let tile_size = f32(uniforms.tile_size);
    let tile_min_x = max(0, i32(floor((mean2d.x - r) / tile_size)));
    let tile_max_x = min(i32(uniforms.tile_grid.x) - 1, i32(floor((mean2d.x + r) / tile_size)));
    let tile_min_y = max(0, i32(floor((mean2d.y - r) / tile_size)));
    let tile_max_y = min(i32(uniforms.tile_grid.y) - 1, i32(floor((mean2d.y + r) / tile_size)));

    if tile_min_x > tile_max_x || tile_min_y > tile_max_y {
        radii[idx] = -1;
        tile_counts[idx] = 0u;
        return;
    }

    let count = u32((tile_max_x - tile_min_x + 1) * (tile_max_y - tile_min_y + 1));
    tile_counts[idx] = count;

    // SH Degree 0: DC term only - no direction needed!
    // Just 3 coefficients per Gaussian
    let sh_offset = idx * 3u;
    let dc = vec3<f32>(
        sh_coeffs[sh_offset],
        sh_coeffs[sh_offset + 1u],
        sh_coeffs[sh_offset + 2u]
    );
    colors[idx] = vec4<f32>(max(SH_C0 * dc + 0.5, vec3<f32>(0.0)), 0.0);

    if (uniforms.output_flags & 2u) != 0u {
        normals[idx] = vec4<f32>(quat_to_normal(rot), 0.0);
    }
}
