//! FLAME Binding Backward Pass
//!
//! Computes gradients w.r.t. local offsets given gradients w.r.t. Gaussian positions.
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

    // Write output gradient
    // Note: For multiple Gaussians per vertex, we would need atomic accumulation
    // For now, we assume one Gaussian per vertex or sequential writes
    offset_grads[gaussian_id].grad = grad_offset;
}

/// Alternative kernel with atomic accumulation (for multiple Gaussians per vertex)
@compute @workgroup_size(256)
fn flame_binding_backward_atomic(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let gaussian_id = gid.x;

    if gaussian_id >= uniforms.num_gaussians {
        return;
    }

    let info = binding_info[gaussian_id];
    let vertex_id = info.vertex_id;

    let tbn = tbn_frames[vertex_id];
    let T = tbn.tangent;
    let B = tbn.bitangent;
    let N = tbn.normal;

    // Check for degenerate TBN
    let tbn_valid = length(N) > 0.0001;
    if !tbn_valid {
        return; // Skip, don't write zero
    }

    let grad_pos = position_grads[gaussian_id].grad;

    let grad_offset = vec3<f32>(
        dot(grad_pos, T),
        dot(grad_pos, B),
        dot(grad_pos, N)
    );

    // Atomic accumulation for thread safety
    // WebGPU doesn't have native f32 atomics, so we use bitcast to u32
    // This is a simplified version - production code would need proper atomic ops

    // For now, use simple write (assumes no conflicts)
    // TODO: Implement proper atomic accumulation if needed
    offset_grads[gaussian_id].grad = grad_offset;
}
