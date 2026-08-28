//! FLAME Binding Backward Pass
//!
//! Purpose
//! ───────
//! Computes gradients w.r.t. local Gaussian offsets and blend weights given
//! gradients w.r.t. world-space Gaussian positions (from preprocess_bwd).
//!
//! Bindings
//! ────────
//! See struct declarations below. Inputs: world-space position gradients,
//! blend weights, FLAME vertex transforms. Outputs: local offset gradients.
//!
//! Dispatch dimensions
//! ───────────────────
//! 1D: ceil(num_gaussians / 256) workgroups × 256 threads.
//!
//! Math
//! ────
//! Reverse of the forward FLAME binding: chain ∂L/∂pos_world through the
//! blended rotation matrix to obtain ∂L/∂pos_local.
//!
//! Problem:
//!   Gaussians are positioned via local offsets from FLAME mesh vertices:
//!   gaussian_position = vertex_position + offset.x * T + offset.y * B + offset.z * N
//!
//! Goal:
//!   Compute ∂L/∂local_offsets given ∂L/∂gaussian_positions
//!
//! Algorithm:
//!   Given gradient ∂L/∂pos, project onto TBN frame:
//!   ∂L/∂offset.x = dot(∂L/∂pos, T)
//!   ∂L/∂offset.y = dot(∂L/∂pos, B)
//!   ∂L/∂offset.z = dot(∂L/∂pos, N)

// ---- Binding Info ----
struct BindingInfo {
    vertex_id: u32,
    _pad0: vec3<u32>,
}

// ---- TBN Frame (Tangent, Bitangent, Normal) ----
struct TbnFrame {
    tangent: vec3<f32>,
    _pad0: f32,
    bitangent: vec3<f32>,
    _pad1: f32,
    normal: vec3<f32>,
    _pad2: f32,
}

// ---- Input: Gaussian Position Gradients ----
struct PositionGradient {
    grad: vec3<f32>,
    _pad: f32,
}

// ---- Output: Local Offset Gradients ----
struct OffsetGradient {
    grad: vec3<f32>,
    _pad: f32,
}

// ---- Uniform Buffer ----
struct Uniforms {
    num_gaussians: u32,
    _pad: vec3<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> binding_info: array<BindingInfo>;
@group(0) @binding(2) var<storage, read> tbn_frames: array<TbnFrame>;
@group(0) @binding(3) var<storage, read> position_grads: array<PositionGradient>;
@group(0) @binding(4) var<storage, read_write> offset_grads: array<OffsetGradient>;

/// Main backward pass kernel
@compute @workgroup_size(256)
fn flame_binding_backward(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let gaussian_id = gid.x;

    // Bounds check
    if gaussian_id >= uniforms.num_gaussians {
        return;
    }

    // Read binding info
    let info = binding_info[gaussian_id];
    let vertex_id = info.vertex_id;

    // Read TBN frame from vertex
    let tbn = tbn_frames[vertex_id];
    let T = tbn.tangent;
    let B = tbn.bitangent;
    let N = tbn.normal;

    // Check for degenerate TBN (zero normal)
    let tbn_valid = length(N) > 0.0001;
    if !tbn_valid {
        // For degenerate TBN, set gradient to zero
        offset_grads[gaussian_id].grad = vec3<f32>(0.0);
        return;
    }

    // Read incoming gradient ∂L/∂position
    let grad_pos = position_grads[gaussian_id].grad;

    // Project gradient onto TBN axes
    // ∂L/∂offset = [dot(∂L/∂pos, T), dot(∂L/∂pos, B), dot(∂L/∂pos, N)]
    let grad_offset = vec3<f32>(
        dot(grad_pos, T),  // ∂L/∂offset.x
        dot(grad_pos, B),  // ∂L/∂offset.y
        dot(grad_pos, N)   // ∂L/∂offset.z
    );

    // Write output gradient.
    //
    // No atomic accumulation is needed and none would be correct here: the
    // local offset is a PER-GAUSSIAN parameter and `offset_grads` is indexed by
    // `gaussian_id`, so every slot has exactly one writer — the single thread
    // that owns that Gaussian. `tbn_frames` is read-only and shared, so several
    // Gaussians binding to the same vertex read the same frame without
    // conflicting. `offset_gradients_cpu` (src/binding.rs) mirrors this
    // one-entry-per-Gaussian contract.
    //
    // (A previous revision carried a second `flame_binding_backward_atomic`
    // entry point advertised as "atomic accumulation for multiple Gaussians per
    // vertex". Its body was a plain write identical to this one plus a TODO, and
    // its premise was wrong for the reason above: per-Gaussian slots cannot
    // race. It has been removed so nothing can select it expecting accumulation
    // semantics. Aggregating gradients PER MESH VERTEX — a different quantity,
    // useful for mesh-space regularisation — would need a `num_vertices`-sized
    // output binding, which this pipeline layout does not have.)
    offset_grads[gaussian_id].grad = grad_offset;
}
