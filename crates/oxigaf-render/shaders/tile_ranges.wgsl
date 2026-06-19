// Compute per-tile start/end ranges in the sorted key array.
//
// Purpose
// ───────
// After key-value radix sort, the sort_keys array holds (tile_id, depth) pairs
// sorted by tile_id.  This shader scans adjacent entries and marks the [start,
// end) index range for each tile, enabling the rasterize_fwd kernel to loop
// only over the Gaussians assigned to its tile.
//
// Bindings
// ────────
// group:binding  type              description
//    0:0         storage (ro)      sort_keys   — sorted vec2<u32> (tile_id, depth)
//    0:1         storage (rw)      tile_ranges — vec2<u32> (start, end) per tile
//    0:2         uniform (vec4u)   params      — x=num_elements, y=num_tiles
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(num_elements / 256) workgroups × 256 threads.
//
// Math
// ────
// Thread i examines sort_keys[i] and sort_keys[i-1].
// When the tile_id changes, it writes tile_ranges[prev_tile].y = i  (end)
// and tile_ranges[curr_tile].x = i  (start).

@group(0) @binding(0) var<storage, read> sort_keys: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> tile_ranges: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = total_pairs, y = num_tiles

@compute @workgroup_size(256)
fn tile_ranges_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_pairs = params.x;
    let num_tiles = params.y;

    if idx >= total_pairs {
        return;
    }

    let tile_id = sort_keys[idx].x;

    // Check if this is the start of a new tile
    if idx == 0u || sort_keys[idx - 1u].x != tile_id {
        if tile_id < num_tiles {
            tile_ranges[tile_id].x = idx;
        }
    }

    // Check if this is the end of a tile
    if idx == total_pairs - 1u || sort_keys[idx + 1u].x != tile_id {
        if tile_id < num_tiles {
            tile_ranges[tile_id].y = idx + 1u;
        }
    }
}
