// Tile assignment: write (tile_id, depth) sort keys for each Gaussian-tile pair.

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
    let tile_size = 16.0;

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

    var write_idx = base_offset;
    for (var ty = tile_min_y; ty <= tile_max_y; ty++) {
        for (var tx = tile_min_x; tx <= tile_max_x; tx++) {
            let tile_id = ty * uniforms.tile_grid.x + tx;
            // Key: high 32 bits = tile_id, low 32 bits = depth
            sort_keys[write_idx] = vec2<u32>(tile_id, depth_bits);
            sort_values[write_idx] = idx;
            write_idx++;
        }
    }
}
