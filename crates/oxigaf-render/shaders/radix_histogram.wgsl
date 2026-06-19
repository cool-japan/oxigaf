// Radix sort histogram pass: compute per-workgroup digit counts.
//
// Purpose
// ───────
// First pass of GPU radix sort.  For a given bit-shift (selecting 4 bits),
// each workgroup histograms its slice of keys into 16 digit buckets and writes
// the counts to histogram[], which is later prefix-summed and used by the
// scatter pass to place elements at their correct sorted positions.
//
// Bindings
// ────────
// See struct/binding declarations below.  Typically:
//   input keys, histogram output, bit-shift uniform.
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(num_elements / workgroup_size) workgroups.
//
// Math
// ────
// digit = (key >> bit_shift) & 0xF.
// histogram[digit * num_workgroups + wid] += 1  for each key in slice.

struct SortParams {
    count: u32,
    pass_idx: u32,
    radix_bits: u32,
    num_workgroups: u32,
};

@group(0) @binding(0) var<storage, read> keys_in: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> histogram: array<u32>;
@group(0) @binding(2) var<uniform> params: SortParams;

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

var<workgroup> local_hist: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn radix_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let idx = gid.x;

    // Clear local histogram
    if lid.x < 16u {
        atomicStore(&local_hist[lid.x], 0u);
    }
    workgroupBarrier();

    // Count digits for this workgroup's elements
    if idx < params.count {
        let digit = extract_digit(keys_in[idx], params.pass_idx);
        atomicAdd(&local_hist[digit], 1u);
    }
    workgroupBarrier();

    // Thread 0-15 write local histogram to global buffer
    if lid.x < 16u {
        histogram[lid.x * params.num_workgroups + wid.x] = atomicLoad(&local_hist[lid.x]);
    }
}
