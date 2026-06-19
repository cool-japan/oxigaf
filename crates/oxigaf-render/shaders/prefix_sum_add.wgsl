// Add scanned block offsets back to each element (hierarchical prefix-sum phase 3).
//
// Purpose
// ───────
// After prefix_sum.wgsl scans each block independently and writes totals to
// block_sums[], and block_sums[] itself is scanned, this shader adds the
// appropriate block offset to every element so the full array is consistent.
//
// Bindings
// ────────
// group:binding  type              description
//    0:0         storage (rw)      data          — partially-scanned array (in-place)
//    0:1         storage (ro)      block_offsets — scanned block_sums from phase 2
//    0:2         uniform (vec4u)   params        — params.x = element count
//
// Dispatch dimensions
// ───────────────────
// 1D: same grid as prefix_sum phase 1. Each workgroup adds block_offsets[wid.x]
// to its 512-element slice.
//
// Math
// ────
// data[i] += block_offsets[wid.x]   for all i in [wid.x*512, (wid.x+1)*512).

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = count

@compute @workgroup_size(256)
fn prefix_sum_add(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = params.x;
    let offset_base = wid.x * 512u;
    let block_offset = block_offsets[wid.x];

    // Each thread handles two elements (matching the scan's 512 per workgroup)
    let ai = offset_base + lid.x;
    let bi = offset_base + lid.x + 256u;

    if ai < n {
        data[ai] += block_offset;
    }
    if bi < n {
        data[bi] += block_offset;
    }
}
