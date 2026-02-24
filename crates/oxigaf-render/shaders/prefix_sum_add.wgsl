// Add scanned block offsets back to each element.
// This is the third phase of hierarchical prefix sum.
// For each element in workgroup wid.x, add the scanned block_sums[wid.x].

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
