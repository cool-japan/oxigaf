// Deform Gaussians compute shader.
//
// Purpose
// ───────
// Transforms N Gaussians from local FLAME mesh coordinates to world space.
//
// Bindings
// ────────
// See the declarations below for the complete binding list.
// Inputs  (group 0, read-only): gaussian_positions, gaussian_rotations,
//         gaussian_face_indices, gaussian_barycentric, gaussian_local_offsets,
//         gaussian_is_rigid, mesh_vertices, mesh_normals, mesh_faces;
//         group 1 holds the DeformUniforms (num_gaussians, num_faces).
// Outputs (group 0, read-write): out_positions, out_rotations.
// There are NO scale bindings and no scale output: scales are pose-invariant
// under this deformation and are left untouched by the kernel.
// `gaussian_is_rigid` IS honoured by this kernel: a rigid Gaussian is
// unaffected by the mesh binding and is written back with its own authored
// position/rotation (see the early-out in `deform_gaussians`). Because the
// kernel enforces that itself, `DeformPipeline::deform` (src/deform.rs)
// does NOT post-process the readback, and `deform_cpu` — which skips rigid
// Gaussians up front, in the same order — stays an exact oracle for this
// kernel.
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(num_gaussians / 256) workgroups × 256 threads.
//
// Math
// ────
// Each Gaussian is bound to a single mesh face via barycentric coordinates and
// a learnable local offset in that face's TBN (Tangent-Bitangent-Normal) frame.
// There are no blend weights and no per-vertex skinning: the frame comes from
// one face, interpolated at the Gaussian's barycentric point.  Rotation
// composition uses quaternion multiplication.
//
// Algorithm per Gaussian:
//   0. Rigid Gaussians are written straight through (no mesh binding at all).
//   1. Look up the face and barycentric coords for this Gaussian.
//   2. Interpolate position p and normal n at the barycentric point.
//   3. Build the TBN orthonormal frame from the face geometry.
//   4. Apply the local offset in TBN space to get the world position.
//   5. Compose the Gaussian's rotation with the TBN quaternion.
//
// NOTE: All vec3 storage buffers use vec4 (padded to 16 bytes) to avoid
// alignment issues, as WGSL mandates 16-byte stride for vec3 arrays.

// ---------------------------------------------------------------------------
// Uniforms (bind group 1)
// ---------------------------------------------------------------------------

struct DeformUniforms {
    num_gaussians: u32,
    num_faces: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(1) @binding(0) var<uniform> uniforms: DeformUniforms;

// ---------------------------------------------------------------------------
// Input buffers (bind group 0, read-only)
// ---------------------------------------------------------------------------

// Gaussian base positions (xyz + pad) in local/world space — [N]
@group(0) @binding(0) var<storage, read> gaussian_positions: array<vec4<f32>>;
// Gaussian rotations (x,y,z,w quaternion) — [N]
@group(0) @binding(1) var<storage, read> gaussian_rotations: array<vec4<f32>>;
// Face index on the mesh for each Gaussian — [N]
@group(0) @binding(2) var<storage, read> gaussian_face_indices: array<u32>;
// Barycentric coordinates (u,v,w + pad) — [N]
@group(0) @binding(3) var<storage, read> gaussian_barycentric: array<vec4<f32>>;
// Local offset in TBN space (x,y,z + pad) — [N]
@group(0) @binding(4) var<storage, read> gaussian_local_offsets: array<vec4<f32>>;
// Rigid flag per Gaussian (0=non-rigid, 1=rigid) — [N]
@group(0) @binding(5) var<storage, read> gaussian_is_rigid: array<u32>;

// Mesh vertex positions (xyz + pad) — [V]
@group(0) @binding(6) var<storage, read> mesh_vertices: array<vec4<f32>>;
// Mesh vertex normals (xyz + pad) — [V]
@group(0) @binding(7) var<storage, read> mesh_normals: array<vec4<f32>>;
// Mesh face vertex indices (v0, v1, v2, pad) — [F]
@group(0) @binding(8) var<storage, read> mesh_faces: array<vec4<u32>>;

// ---------------------------------------------------------------------------
// Output buffers (bind group 0, read-write)
// ---------------------------------------------------------------------------

// Deformed world-space positions (xyz + pad) — [N]
@group(0) @binding(9)  var<storage, read_write> out_positions: array<vec4<f32>>;
// Deformed world-space rotations (x,y,z,w quaternion) — [N]
@group(0) @binding(10) var<storage, read_write> out_rotations: array<vec4<f32>>;

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Multiply two quaternions q_a * q_b (Hamilton product).
/// Both quaternions are stored as (x, y, z, w).
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    // a = (ax, ay, az, aw),  b = (bx, by, bz, bw)
    let ax = a.x; let ay = a.y; let az = a.z; let aw = a.w;
    let bx = b.x; let by = b.y; let bz = b.z; let bw = b.w;
    return vec4<f32>(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    );
}

/// Convert an orthonormal 3×3 rotation matrix (column vectors t, bt, n) to a
/// quaternion using Shepperd's method.
///
/// The columns are:
///   col 0 = tangent    (t)
///   col 1 = bitangent  (bt)
///   col 2 = normal     (n)
///
/// Stored as three separate vec3 arguments for clarity.
/// Returns quaternion as (x, y, z, w) with positive w.
fn mat3_to_quat(t: vec3<f32>, bt: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    // Build the 3×3 matrix in component notation.
    // m[row][col]: m00 = t.x, m10 = t.y, m20 = t.z
    //              m01 = bt.x, m11 = bt.y, m21 = bt.z
    //              m02 = n.x,  m12 = n.y,  m22 = n.z
    let m00 = t.x;  let m01 = bt.x; let m02 = n.x;
    let m10 = t.y;  let m11 = bt.y; let m12 = n.y;
    let m20 = t.z;  let m21 = bt.z; let m22 = n.z;

    let trace = m00 + m11 + m22;

    var qx: f32;
    var qy: f32;
    var qz: f32;
    var qw: f32;

    if trace > 0.0 {
        let s = sqrt(trace + 1.0) * 2.0; // s = 4 * w
        qw = 0.25 * s;
        qx = (m21 - m12) / s;
        qy = (m02 - m20) / s;
        qz = (m10 - m01) / s;
    } else if (m00 > m11) && (m00 > m22) {
        let s = sqrt(1.0 + m00 - m11 - m22) * 2.0; // s = 4 * x
        qw = (m21 - m12) / s;
        qx = 0.25 * s;
        qy = (m01 + m10) / s;
        qz = (m02 + m20) / s;
    } else if m11 > m22 {
        let s = sqrt(1.0 + m11 - m00 - m22) * 2.0; // s = 4 * y
        qw = (m02 - m20) / s;
        qx = (m01 + m10) / s;
        qy = 0.25 * s;
        qz = (m12 + m21) / s;
    } else {
        let s = sqrt(1.0 + m22 - m00 - m11) * 2.0; // s = 4 * z
        qw = (m10 - m01) / s;
        qx = (m02 + m20) / s;
        qy = (m12 + m21) / s;
        qz = 0.25 * s;
    }

    // Normalise to guard against numerical drift.
    let q = vec4<f32>(qx, qy, qz, qw);
    return normalize(q);
}

// ---------------------------------------------------------------------------
// Main kernel — one thread per Gaussian
// ---------------------------------------------------------------------------

@compute @workgroup_size(256)
fn deform_gaussians(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let gaussian_id = gid.x;
    if gaussian_id >= uniforms.num_gaussians {
        return;
    }

    // ---- Rigid Gaussians: not bound to the mesh at all ----
    // A rigid Gaussian keeps its own authored position/rotation: the
    // barycentric interpolation, the TBN local offset and the TBN rotation
    // composition below must all be skipped for it. Checked before the face
    // index so a rigid Gaussian with a stale/garbage `face_indices` entry is
    // still passed through, matching `deform_cpu`'s ordering exactly.
    if gaussian_is_rigid[gaussian_id] != 0u {
        out_positions[gaussian_id] = vec4<f32>(gaussian_positions[gaussian_id].xyz, 1.0);
        out_rotations[gaussian_id] = gaussian_rotations[gaussian_id];
        return;
    }

    // ---- Fetch Gaussian binding data ----
    let fi = gaussian_face_indices[gaussian_id];
    if fi >= uniforms.num_faces {
        // Invalid face index: pass through unchanged. The w component is
        // rewritten to 1.0 rather than copied: `gaussian_positions` is
        // uploaded via `Vec4f32::from_xyz`, which pads w with 0.0, whereas
        // every other write here (and `deform_cpu`'s pass-through) emits a
        // homogeneous point with w = 1.0.
        out_positions[gaussian_id] = vec4<f32>(gaussian_positions[gaussian_id].xyz, 1.0);
        out_rotations[gaussian_id] = gaussian_rotations[gaussian_id];
        return;
    }

    let bary  = gaussian_barycentric[gaussian_id].xyz;
    let local = gaussian_local_offsets[gaussian_id].xyz;
    let g_rot = gaussian_rotations[gaussian_id];

    // ---- Fetch face vertex indices ----
    let face   = mesh_faces[fi];
    let vi0    = face.x;
    let vi1    = face.y;
    let vi2    = face.z;

    // ---- Interpolate position ----
    let p0 = mesh_vertices[vi0].xyz;
    let p1 = mesh_vertices[vi1].xyz;
    let p2 = mesh_vertices[vi2].xyz;
    let interp_pos = bary.x * p0 + bary.y * p1 + bary.z * p2;

    // ---- Interpolate normal ----
    let n0 = mesh_normals[vi0].xyz;
    let n1 = mesh_normals[vi1].xyz;
    let n2 = mesh_normals[vi2].xyz;
    let interp_normal_raw = bary.x * n0 + bary.y * n1 + bary.z * n2;

    // ---- Build TBN frame ----
    let e1 = p1 - p0;   // edge 0→1
    let e2 = p2 - p0;   // edge 0→2

    // Cross product of edges gives the unnormalised geometric normal.
    let geom_normal_raw = cross(e1, e2);
    let geom_normal_len = length(geom_normal_raw);

    var tbn_t:  vec3<f32>;
    var tbn_bt: vec3<f32>;
    var tbn_n:  vec3<f32>;

    if geom_normal_len < 0.0001 {
        // Degenerate triangle: fall back to canonical axes.
        tbn_t  = vec3<f32>(1.0, 0.0, 0.0);
        tbn_bt = vec3<f32>(0.0, 1.0, 0.0);
        tbn_n  = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        // Use interpolated vertex normal when available and non-zero;
        // otherwise fall back to the geometric normal.
        let interp_len = length(interp_normal_raw);
        if interp_len > 0.0001 {
            tbn_n = interp_normal_raw / interp_len;
        } else {
            tbn_n = geom_normal_raw / geom_normal_len;
        }

        // Tangent from first edge, re-orthogonalised against n.
        let e1_len = length(e1);
        var raw_t: vec3<f32>;
        if e1_len > 0.0001 {
            raw_t = e1 / e1_len;
        } else {
            // Degenerate edge: choose an arbitrary perpendicular.
            // Find axis most orthogonal to n to avoid degeneracy.
            let abs_n = abs(tbn_n);
            if abs_n.x <= abs_n.y && abs_n.x <= abs_n.z {
                raw_t = vec3<f32>(1.0, 0.0, 0.0);
            } else if abs_n.y <= abs_n.z {
                raw_t = vec3<f32>(0.0, 1.0, 0.0);
            } else {
                raw_t = vec3<f32>(0.0, 0.0, 1.0);
            }
        }

        // Gram-Schmidt: project raw_t onto the plane perpendicular to n.
        let t_proj = raw_t - dot(raw_t, tbn_n) * tbn_n;
        let t_proj_len = length(t_proj);
        if t_proj_len > 0.0001 {
            tbn_t = t_proj / t_proj_len;
        } else {
            // raw_t was (anti-)parallel to n: pick a different axis.
            let abs_n = abs(tbn_n);
            var fallback_t: vec3<f32>;
            if abs_n.x <= abs_n.y && abs_n.x <= abs_n.z {
                fallback_t = vec3<f32>(1.0, 0.0, 0.0);
            } else if abs_n.y <= abs_n.z {
                fallback_t = vec3<f32>(0.0, 1.0, 0.0);
            } else {
                fallback_t = vec3<f32>(0.0, 0.0, 1.0);
            }
            let tp2 = fallback_t - dot(fallback_t, tbn_n) * tbn_n;
            tbn_t = normalize(tp2);
        }

        // Bitangent = n × t (right-hand rule, ensures orthonormality).
        tbn_bt = normalize(cross(tbn_n, tbn_t));

        // Re-orthogonalise t from (bt, n) to ensure a strict right-hand frame.
        tbn_t = cross(tbn_bt, tbn_n);
    }

    // ---- Apply local offset in TBN space ----
    let world_pos = interp_pos
        + tbn_t  * local.x
        + tbn_bt * local.y
        + tbn_n  * local.z;

    // ---- Rotate Gaussian rotation by TBN quaternion ----
    let q_tbn    = mat3_to_quat(tbn_t, tbn_bt, tbn_n);
    let world_rot = quat_mul(q_tbn, g_rot);

    // ---- Write outputs ----
    out_positions[gaussian_id] = vec4<f32>(world_pos, 1.0);
    out_rotations[gaussian_id] = world_rot;
}
