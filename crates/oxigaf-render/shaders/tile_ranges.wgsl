// Compute per-tile start/end ranges in the sorted key array.

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
