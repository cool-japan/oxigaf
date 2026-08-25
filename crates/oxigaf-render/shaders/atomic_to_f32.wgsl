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
// This is a GENERIC one-buffer-at-a-time converter: it holds exactly three
// bindings and the caller re-binds and re-dispatches it once per gradient
// buffer (rasterizer.rs does so for grad_means2d, grad_conics and grad_colors;
// grad_opacities needs no conversion because it is read back as raw bytes).
//
// group:binding  type              description
//    0:0         uniform (u32)     num_elements — number of f32 SLOTS to copy
//    0:1         storage (rw)      atomic_buffer — atomic<u32> source
//    0:2         storage (rw)      f32_buffer    — f32 destination
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(num_elements / 256) × 256 threads.  The unit is ELEMENTS, not
// Gaussians: callers pass `n * 2` for means2d and `n * 3` for conics/colors.
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
