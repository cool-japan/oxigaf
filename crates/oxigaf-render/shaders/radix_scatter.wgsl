// Radix sort scatter pass: place elements at their sorted positions.
//
// Purpose
// ───────
// Second pass of GPU radix sort.  Using the prefix-summed histogram from
// the histogram pass, each thread writes its key and value to the globally
// correct output position, producing a stably sorted output array for the
// selected 4-bit digit.
//
// Bindings
// ────────
// See struct/binding declarations below.  Typically:
//   input keys+values, prefix-summed histogram, output keys+values, bit-shift.
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(num_elements / workgroup_size) workgroups.
//
// Math
// ────
// pos = histogram_prefix[digit * num_wg + wid] + local_rank.
// out_key[pos] = in_key[i];  out_val[pos] = in_val[i].
// The starting position for (digit, wg_id) = prefix[d*nwg + wg] - count[d*nwg + wg].

struct SortParams {
    count: u32,
    pass_idx: u32,
    radix_bits: u32,
    num_workgroups: u32,
};

@group(0) @binding(0) var<storage, read> keys_in: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> values_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(4) var<storage, read> histogram: array<u32>;
@group(0) @binding(5) var<storage, read> histogram_prefix: array<u32>;
@group(0) @binding(6) var<uniform> params: SortParams;

fn extract_digit(key: vec2<u32>, pass_idx: u32) -> u32 {
    let bit_offset = pass_idx * 4u;
    var word: u32;
    if bit_offset < 32u {
        word = key.y;
    } else {
        word = key.x;
    }
    let shift = bit_offset % 32u;
    return (word >> shift) & 0xFu;
}

// Shared memory for local rank computation
var<workgroup> wg_digits: array<u32, 256>;

@compute @workgroup_size(256)
fn radix_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let idx = gid.x;
    let nwg = params.num_workgroups;

    // Each thread loads its digit into shared memory
    var digit = 16u; // sentinel for out-of-bounds
    if idx < params.count {
        digit = extract_digit(keys_in[idx], params.pass_idx);
    }
    wg_digits[lid.x] = digit;
    workgroupBarrier();

    if idx >= params.count {
        return;
    }

    // Compute local rank: count how many earlier threads in this workgroup
    // have the same digit (stable sort: preserve relative order)
    var local_rank = 0u;
    for (var i = 0u; i < lid.x; i++) {
        if wg_digits[i] == digit {
            local_rank++;
        }
    }

    // Look up global starting position for this (digit, workgroup)
    let hist_idx = digit * nwg + wid.x;
    let inclusive_prefix = histogram_prefix[hist_idx];
    let count_here = histogram[hist_idx];
    let global_start = inclusive_prefix - count_here;

    let dest = global_start + local_rank;
    if dest < params.count {
        keys_out[dest] = keys_in[idx];
        values_out[dest] = values_in[idx];
    }
}
