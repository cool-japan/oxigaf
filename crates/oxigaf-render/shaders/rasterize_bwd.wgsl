// Backward rasterization: reverse-order tile traversal for gradient computation.
// Computes dL/d(color), dL/d(opacity), dL/d(mean2d), dL/d(conic).

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
@group(0) @binding(2) var<storage, read> conics: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> opacities: array<f32>;
@group(0) @binding(5) var<storage, read> out_color: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> out_transmittance: array<f32>;
@group(0) @binding(7) var<storage, read> out_n_contrib: array<u32>;
@group(0) @binding(8) var<storage, read> tile_ranges: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read> sort_values: array<u32>;
@group(0) @binding(10) var<storage, read> grad_output: array<vec4<f32>>;
// Gradient accumulators (atomic u32 for CAS-based f32 addition)
@group(0) @binding(11) var<storage, read_write> grad_colors: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> grad_opacities: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> grad_means2d: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> grad_conics: array<atomic<u32>>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(16, 16)
fn rasterize_backward(
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

    let tile_x = wid.x;
    let tile_y = wid.y;
    let tile_id = tile_y * uniforms.tile_grid.x + tile_x;

    let range = tile_ranges[tile_id];
    let range_start = range.x;
    let range_end = range.y;

    let dL_dpixel = grad_output[pixel_idx];
    let dL_dcolor_out = dL_dpixel.xyz;

    let final_T = out_transmittance[pixel_idx];
    let n_contrib_total = out_n_contrib[pixel_idx];

    // Reconstruct transmittance in reverse order
    var T = final_T;
    var accum_color = vec3<f32>(0.0);

    // Reverse traversal – only iterate over entries the forward pass actually
    // processed. n_contrib_total now stores the absolute stopping sort index
    // (k_stop) written by the forward shader, NOT a count.
    let effective_end = n_contrib_total;
    if effective_end <= range_start {
        return;
    }

    for (var k_rev = 0u; k_rev < (effective_end - range_start); k_rev++) {
        let k = effective_end - 1u - k_rev;
        let gaussian_idx = sort_values[k];
        let mean = means2d[gaussian_idx];
        let conic = conics[gaussian_idx];

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

        // Recover T before this Gaussian was blended
        T /= (1.0 - alpha);

        let c = colors[gaussian_idx].xyz;

        // dL/d(alpha) for this Gaussian
        let dL_dalpha = dot(dL_dcolor_out, T * c - accum_color / (1.0 - alpha + 1e-7));

        // dL/d(color) for this Gaussian
        let dL_dc = dL_dcolor_out * T * alpha;

        // Accumulate color seen after this point (T is T_before_this_gaussian)
        accum_color += alpha * T * c;

        // dL/d(opacity) through sigmoid
        // When alpha is capped at 0.99 (alpha_raw >= 0.99), the cap has zero
        // derivative so both dL_dpower and dL_dopacity must be zeroed.
        let sig = sigmoid(opacities[gaussian_idx]);
        let dsig = sig * (1.0 - sig);
        let dL_dopacity = select(dL_dalpha * exp(power) * dsig, 0.0, alpha_raw >= 0.99);

        // dL/d(power) - true gradient of loss w.r.t. power
        let dL_dpower = select(dL_dalpha * alpha_raw, 0.0, alpha_raw >= 0.99);

        // dL/d(mean2d): d(power)/d(mean.x) = conic.x * d.x + conic.y * d.y
        // (derivation: power = -0.5 * Q, d(Q)/d(d.x) = 2*(conic.x*d.x + conic.y*d.y),
        //  d(power)/d(d.x) = -(conic.x*d.x + conic.y*d.y), d(d.x)/d(mean.x) = -1)
        let dL_dx = dL_dpower * (conic.x * d.x + conic.y * d.y);
        let dL_dy = dL_dpower * (conic.y * d.x + conic.z * d.y);

        // dL/d(conic): d(power)/d(conic.x) = -0.5 * d.x^2
        let dL_dconic_a = dL_dpower * (-0.5) * d.x * d.x;
        let dL_dconic_b = dL_dpower * (-1.0) * d.x * d.y;  // -0.5 * 2.0 = -1.0
        let dL_dconic_c = dL_dpower * (-0.5) * d.y * d.y;

        // Atomic accumulate (CAS loop inlined for WebGPU compatibility)
        // grad_colors[gaussian_idx * 3 + 0] += dL_dc.x
        {
            var old = atomicLoad(&grad_colors[gaussian_idx * 3u + 0u]);
            loop {
                let new_val = bitcast<f32>(old) + dL_dc.x;
                let result = atomicCompareExchangeWeak(&grad_colors[gaussian_idx * 3u + 0u], old, bitcast<u32>(new_val));
                if result.exchanged { break; }
                old = result.old_value;
            }
        }
        // grad_colors[gaussian_idx * 3 + 1] += dL_dc.y
        {
            var old = atomicLoad(&grad_colors[gaussian_idx * 3u + 1u]);
            loop {
                let new_val = bitcast<f32>(old) + dL_dc.y;
                let result = atomicCompareExchangeWeak(&grad_colors[gaussian_idx * 3u + 1u], old, bitcast<u32>(new_val));
                if result.exchanged { break; }
                old = result.old_value;
            }
        }
        // grad_colors[gaussian_idx * 3 + 2] += dL_dc.z
        {
            var old = atomicLoad(&grad_colors[gaussian_idx * 3u + 2u]);
            loop {
                let new_val = bitcast<f32>(old) + dL_dc.z;
                let result = atomicCompareExchangeWeak(&grad_colors[gaussian_idx * 3u + 2u], old, bitcast<u32>(new_val));
                if result.exchanged { break; }
                old = result.old_value;
            }
        }
        // grad_opacities[gaussian_idx] += dL_dopacity
        {
            var old = atomicLoad(&grad_opacities[gaussian_idx]);
            loop {
                let new_val = bitcast<f32>(old) + dL_dopacity;
                let result = atomicCompareExchangeWeak(&grad_opacities[gaussian_idx], old, bitcast<u32>(new_val));
                if result.exchanged { break; }
                old = result.old_value;
            }
        }
        // grad_means2d[gaussian_idx * 2 + 0] += dL_dx
        {
            var old = atomicLoad(&grad_means2d[gaussian_idx * 2u + 0u]);
            loop {
                let new_val = bitcast<f32>(old) + dL_dx;
                let result = atomicCompareExchangeWeak(&grad_means2d[gaussian_idx * 2u + 0u], old, bitcast<u32>(new_val));
                if result.exchanged { break; }
                old = result.old_value;
            }
        }
        // grad_means2d[gaussian_idx * 2 + 1] += dL_dy
        {
            var old = atomicLoad(&grad_means2d[gaussian_idx * 2u + 1u]);
            loop {
                let new_val = bitcast<f32>(old) + dL_dy;
                let result = atomicCompareExchangeWeak(&grad_means2d[gaussian_idx * 2u + 1u], old, bitcast<u32>(new_val));
                if result.exchanged { break; }
                old = result.old_value;
            }
        }
        // grad_conics[gaussian_idx * 3 + 0] += dL_dconic_a
        {
            var old = atomicLoad(&grad_conics[gaussian_idx * 3u + 0u]);
            loop {
                let new_val = bitcast<f32>(old) + dL_dconic_a;
                let result = atomicCompareExchangeWeak(&grad_conics[gaussian_idx * 3u + 0u], old, bitcast<u32>(new_val));
                if result.exchanged { break; }
                old = result.old_value;
            }
        }
        // grad_conics[gaussian_idx * 3 + 1] += dL_dconic_b
        {
            var old = atomicLoad(&grad_conics[gaussian_idx * 3u + 1u]);
            loop {
                let new_val = bitcast<f32>(old) + dL_dconic_b;
                let result = atomicCompareExchangeWeak(&grad_conics[gaussian_idx * 3u + 1u], old, bitcast<u32>(new_val));
                if result.exchanged { break; }
                old = result.old_value;
            }
        }
        // grad_conics[gaussian_idx * 3 + 2] += dL_dconic_c
        {
            var old = atomicLoad(&grad_conics[gaussian_idx * 3u + 2u]);
            loop {
                let new_val = bitcast<f32>(old) + dL_dconic_c;
                let result = atomicCompareExchangeWeak(&grad_conics[gaussian_idx * 3u + 2u], old, bitcast<u32>(new_val));
                if result.exchanged { break; }
                old = result.old_value;
            }
        }
    }
}
