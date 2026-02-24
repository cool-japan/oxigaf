// GPU radix sort (4-bit per pass, 16 passes for 64-bit keys).
// Simplified single-workgroup counting sort per pass.

struct SortParams {
    count: u32,
    pass_idx: u32,
    radix_bits: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> keys_in: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> values_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: SortParams;

// Extract 4-bit digit from a 64-bit key (stored as vec2<u32>: x=high, y=low)
fn extract_digit(key: vec2<u32>, pass_idx: u32) -> u32 {
    let bit_offset = pass_idx * 4u;
    var word: u32;
    if bit_offset < 32u {
        word = key.y; // low word
    } else {
        word = key.x; // high word
    }
    let shift = bit_offset % 32u;
    return (word >> shift) & 0xFu;
}

var<workgroup> local_histogram: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn radix_sort_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = gid.x;

    // Clear local histogram
    if lid.x < 16u {
        atomicStore(&local_histogram[lid.x], 0u);
    }
    workgroupBarrier();

    // Count digits
    if idx < params.count {
        let key = keys_in[idx];
        let digit = extract_digit(key, params.pass_idx);
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    // Simple scatter: for small arrays, use a basic counting approach.
    // For production, this would use a multi-pass prefix sum + scatter.
    // Here we do a serial scatter per workgroup for correctness.
    if idx < params.count {
        let key = keys_in[idx];
        let value = values_in[idx];
        let digit = extract_digit(key, params.pass_idx);

        // Count how many elements before us have the same digit
        var rank = 0u;
        for (var i = 0u; i < idx; i++) {
            if i < params.count {
                let other_digit = extract_digit(keys_in[i], params.pass_idx);
                if other_digit == digit {
                    rank++;
                }
            }
        }

        // Count elements with smaller digits
        var base = 0u;
        for (var d = 0u; d < digit; d++) {
            base += atomicLoad(&local_histogram[d]);
        }

        let dest = base + rank;
        if dest < params.count {
            keys_out[dest] = key;
            values_out[dest] = value;
        }
    }
}
