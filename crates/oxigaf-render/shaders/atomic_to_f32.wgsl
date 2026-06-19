// Atomic to f32 buffer conversion shader.
//
// Purpose
// ───────
// Converts the bitcast-encoded f32 gradient values packed into atomic<u32>
// buffers (produced by rasterize_bwd) into plain f32 storage buffers consumed
// by preprocess_bwd.  This decoupling lets the backward rasterizer use
// hardware-atomic CAS while the preprocess backward reads clean f32 data.
//
// Bindings
// ────────
// group:binding  type              description
//    0:0         storage (ro)      src_colors     — atomic<u32> grad_colors
//    0:1         storage (ro)      src_opacities  — atomic<u32> grad_opacities
//    0:2         storage (ro)      src_means2d    — atomic<u32> grad_means2d
//    0:3         storage (ro)      src_conics     — atomic<u32> grad_conics
//    0:4         storage (rw)      dst_colors     — f32 grad_colors
//    0:5         storage (rw)      dst_opacities  — f32 grad_opacities
//    0:6         storage (rw)      dst_means2d    — f32 grad_means2d
//    0:7         storage (rw)      dst_conics     — f32 grad_conics
//    0:8         uniform (vec4u)   params         — x = num_gaussians
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(num_gaussians / 256) × 256 threads.
//
// Math
// ────
// dst[i] = bitcast<f32>(src[i])  for each buffer slot.
// This is necessary because WGSL does not allow reading atomic buffers
// as non-atomic types, even though the underlying data is bitcast f32.

@group(0) @binding(0) var<uniform> num_elements: u32;
@group(0) @binding(1) var<storage, read_write> atomic_buffer: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> f32_buffer: array<f32>;

@compute @workgroup_size(256)
fn atomic_to_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= num_elements {
        return;
    }

    // Read atomic value and bitcast to f32
    let atomic_val = atomicLoad(&atomic_buffer[idx]);
    let f32_val = bitcast<f32>(atomic_val);

    // Write to regular f32 buffer
    f32_buffer[idx] = f32_val;
}
