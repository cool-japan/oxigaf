// Forward rasterization: per-tile alpha-blending (front-to-back).

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
@group(0) @binding(2) var<storage, read> conics: array<vec4<f32>>;  // vec3 with padding
@group(0) @binding(3) var<storage, read> colors: array<vec4<f32>>;  // vec3 with padding
@group(0) @binding(4) var<storage, read> opacities: array<f32>;
@group(0) @binding(5) var<storage, read> depths: array<f32>;
@group(0) @binding(6) var<storage, read> tile_ranges: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> sort_values: array<u32>;
@group(0) @binding(8) var<storage, read_write> out_color: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> out_depth: array<f32>;
@group(0) @binding(10) var<storage, read_write> out_transmittance: array<f32>;
@group(0) @binding(11) var<storage, read_write> out_n_contrib: array<u32>;
@group(0) @binding(12) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> out_normals: array<vec4<f32>>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(16, 16)
fn rasterize_forward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let px = gid.x;
    let py = gid.y;
    let W = u32(uniforms.viewport.x);
    let H = u32(uniforms.viewport.y);

    if px >= W || py >= H {
        return;
    }

    let pixel_idx = py * W + px;
    let pixel_f = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

    // Determine tile
    let tile_x = wid.x;
    let tile_y = wid.y;
    let tile_id = tile_y * uniforms.tile_grid.x + tile_x;

    let range = tile_ranges[tile_id];
    let range_start = range.x;
    let range_end = range.y;

    // Alpha blending (front-to-back)
    var T = 1.0f; // transmittance
    var color = vec3<f32>(0.0);
    var depth_acc = 0.0f;
    var normal_acc = vec3<f32>(0.0);
    var n_contrib = 0u;
    // Track the absolute stopping index so the backward pass knows
    // exactly which sort entries were actually blended.
    var k_stop = range_end;

    for (var k = range_start; k < range_end; k++) {
        if T < 1.0 / 255.0 {
            k_stop = k;
            break;
        }

        let gaussian_idx = sort_values[k];
        let mean = means2d[gaussian_idx];
        let conic = conics[gaussian_idx];

        // Evaluate 2D Gaussian
        let d = pixel_f - mean;
        let power = -0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y);

        if power > 0.0 || power < -4.0 {
            continue;
        }

        let alpha_raw = sigmoid(opacities[gaussian_idx]) * exp(power);
        let alpha = min(alpha_raw, 0.99);

        if alpha < 1.0 / 255.0 {
            continue;
        }

        let c = colors[gaussian_idx].xyz;
        let weight = T * alpha;
        color += weight * c;
        depth_acc += weight * depths[gaussian_idx];

        // Accumulate normals if output is enabled (bit 1 of output_flags)
        if (uniforms.output_flags & 2u) != 0u {
            normal_acc += weight * normals[gaussian_idx].xyz;
        }

        T *= (1.0 - alpha);
        n_contrib++;
    }

    // Add background
    color += T * uniforms.background;

    out_color[pixel_idx] = vec4<f32>(color, 1.0 - T);
    out_depth[pixel_idx] = depth_acc;
    out_transmittance[pixel_idx] = T;
    // Store the stopping sort index (not count) so backward can limit its range.
    // k_stop == range_end when no early termination; k_stop < range_end otherwise.
    out_n_contrib[pixel_idx] = k_stop;

    // Write normals if enabled
    if (uniforms.output_flags & 2u) != 0u {
        // Normalize accumulated normal (weighted sum)
        let normal_len = length(normal_acc);
        let final_normal = select(vec3<f32>(0.0, 0.0, 1.0), normal_acc / normal_len, normal_len > 0.0001);
        out_normals[pixel_idx] = vec4<f32>(final_normal, 0.0);
    }
}
