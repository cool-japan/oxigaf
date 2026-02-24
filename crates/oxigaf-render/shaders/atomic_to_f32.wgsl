// Atomic to f32 buffer conversion shader.
//
// Reads from atomic<u32> buffers (written by rasterize_bwd) and writes
// to regular f32 buffers (read by preprocess_bwd).
//
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
