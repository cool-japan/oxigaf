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
// 1D: same grid as prefix_sum phase 1. Each workgroup adds the offset of all
// *preceding* blocks to its 512-element slice.
//
// Math
// ────
// `block_offsets` is the **inclusive** scan of the per-block totals produced by
// phase 2 (prefix_sum.wgsl writes an inclusive scan). The amount that has to be
// added in front of block `b` is therefore the total of blocks `0..b-1`, i.e.
// the *exclusive* prefix — which is `block_offsets[b - 1]`, and 0 for block 0.
// Adding `block_offsets[b]` instead would inflate every block by its own total.
//
// data[i] += (wid.x == 0 ? 0 : block_offsets[wid.x - 1])
//     for all i in [wid.x*512, (wid.x+1)*512).

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

    // Exclusive prefix of the block totals: block 0 gets nothing, block b gets
    // the inclusive scan value of block b-1.  Indexing `block_offsets[wid.x]`
    // here would add each block's own total to itself.
    var block_offset = 0u;
    if wid.x > 0u {
        block_offset = block_offsets[wid.x - 1u];
    }

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
