// Tile assignment: write (tile_id, depth) sort keys for each Gaussian-tile pair.
//
// Purpose
// ───────
// For each visible Gaussian, determines which tiles it overlaps (based on its
// screen-space bounding box derived from mean2d and radius) and writes a
// sort key of (tile_id << 32 | depth_bits) into the sort_keys buffer.  The
// matching sort_values entry holds the Gaussian index.  After this pass,
// a radix sort orders entries by tile then by depth for the rasterizer.
//
// Bindings
// ────────
// group:binding  type              description
//    0:0         uniform           Camera/viewport/render uniforms
//    0:1         storage (ro)      means2d      — projected 2D centres
//    0:2         storage (ro)      depths       — view-space depths
//    0:3         storage (ro)      radii        — screen-space radii (pixels)
//    0:4         storage (ro)      tile_offsets — prefix-summed per-Gaussian tile counts
//    0:5         storage (rw)      sort_keys    — output (tile,depth) keys
//    0:6         storage (rw)      sort_values  — output Gaussian indices
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(num_gaussians / 256) workgroups × 256 threads.
//
// Math
// ────
// Bounding box: [mean2d − radius, mean2d + radius] converted to tile coordinates.
// For each covered tile: key = (tile_y * tile_grid_x + tile_x, reinterpret_f32(depth)).
//
// Capacity
// ────────
// The tile bounding box MUST be derived with the same `uniforms.tile_size` that
// preprocess.wgsl used to fill `tile_counts`, otherwise the prefix-summed write
// offsets and the number of entries written here disagree and every Gaussian
// overruns its neighbour's slot range.
//
// `sort_keys` / `sort_values` are allocated for `max_pairs` entries, which is
// only an estimate (`IntermediateBuffers::estimate_max_pairs`). `arrayLength`
// gives the real capacity, so the write loop stops instead of letting WGSL
// bounds-clamping collapse every overflowing pair onto the last slot. The host
// separately validates the exact pair total against the capacity
// (`IntermediateBuffers::verify_pair_capacity`) and reports
// `RenderError::TooManyTilePairs`; this guard is the in-shader safety net.

struct Uniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    cam_pos: vec3<f32>,
    _pad0: f32,
    focal: vec2<f32>,
    viewport: vec2<f32>,
    tile_grid: vec2<u32>,
    num_gaussians: u32,
    sh_degree: u32,
    near_plane: f32,
    far_plane: f32,
    _pad_bg: vec2<f32>,
    background: vec3<f32>,
    output_flags: u32,
    transmittance_threshold: f32,
    tile_size: u32,
    _pad1: vec2<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> means2d: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> depths: array<f32>;
@group(0) @binding(3) var<storage, read> radii: array<i32>;
@group(0) @binding(4) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> sort_keys: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> sort_values: array<u32>;

@compute @workgroup_size(256)
fn tile_assign(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.num_gaussians {
        return;
    }

    let radius = radii[idx];
    if radius <= 0 {
        return;
    }

    let mean = means2d[idx];
    let depth = depths[idx];
    let r = f32(radius);
    // Must match preprocess.wgsl's `f32(uniforms.tile_size)`; a hardcoded 16.0
    // here silently desyncs from the prefix-summed tile counts.
    let tile_size = f32(uniforms.tile_size);

    let tile_min_x = u32(max(0, i32(floor((mean.x - r) / tile_size))));
    let tile_max_x = u32(min(i32(uniforms.tile_grid.x) - 1, i32(floor((mean.x + r) / tile_size))));
    let tile_min_y = u32(max(0, i32(floor((mean.y - r) / tile_size))));
    let tile_max_y = u32(min(i32(uniforms.tile_grid.y) - 1, i32(floor((mean.y + r) / tile_size))));

    // Depth as u32 for sorting (preserves order for positive depths)
    let depth_bits = bitcast<u32>(depth);

    // Get the write offset for this Gaussian
    var base_offset: u32;
    if idx == 0u {
        base_offset = 0u;
    } else {
        // tile_offsets is inclusive prefix sum, so previous Gaussian's offset
        // gives our starting point (previous sum = items before us)
        base_offset = tile_offsets[idx - 1u];
    }

    // Allocated pair capacity: both buffers are sized to max_pairs, so take the
    // smaller of the two lengths and never write past it.
    let max_pairs = min(arrayLength(&sort_keys), arrayLength(&sort_values));

    var write_idx = base_offset;
    for (var ty = tile_min_y; ty <= tile_max_y; ty++) {
        for (var tx = tile_min_x; tx <= tile_max_x; tx++) {
            if write_idx >= max_pairs {
                // Out of capacity: stop rather than let bounds-clamping fold
                // every remaining pair onto the final slot. There is no
                // workgroup barrier in this kernel, so returning early from a
                // subset of threads is safe.
                return;
            }
            let tile_id = ty * uniforms.tile_grid.x + tx;
            // Key: high 32 bits = tile_id, low 32 bits = depth
            sort_keys[write_idx] = vec2<u32>(tile_id, depth_bits);
            sort_values[write_idx] = idx;
            write_idx++;
        }
    }
}
