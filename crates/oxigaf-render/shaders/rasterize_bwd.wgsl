// Backward rasterization: reverse-order tile traversal for gradient computation.
// Computes dL/d(color), dL/d(opacity), dL/d(mean2d), dL/d(conic).
//
// Purpose
// ───────
// Each workgroup (one tile, 16×16 threads) traverses the sorted Gaussian list
// in reverse depth order, accumulating gradients for each Gaussian.  To reduce
// global-memory atomic contention, per-Gaussian gradient contributions from all
// 256 threads are first summed into workgroup shared-memory atomics, then a
// single elected thread (local_invocation_index == 0) performs one global CAS
// write per gradient slot per Gaussian.  This reduces DRAM CAS operations by
// up to 256× compared to every thread writing to global memory directly.
//
// Bindings
// ────────
// group:binding  type                description
//    0:0         uniform             Camera/viewport/render uniforms
//    0:1         storage (ro)        means2d   — projected 2D centres
//    0:2         storage (ro)        conics    — inverse-cov2D
//    0:3         storage (ro)        colors    — per-Gaussian RGB
//    0:4         storage (ro)        opacities — pre-sigmoid opacities
//    0:5         storage (ro)        out_color — forward render output
//    0:6         storage (ro)        out_transmittance — forward T per pixel
//    0:7         storage (ro)        out_n_contrib — stopping sort index per pixel
//    0:8         storage (ro)        tile_ranges — [start,end) per tile
//    0:9         storage (ro)        sort_values — sorted Gaussian indices
//   0:10         storage (ro)        grad_output — ∂L/∂pixel_color
//   0:11         storage (rw)        grad_colors   — atomic<u32>, bitcast f32
//   0:12         storage (rw)        grad_opacities — atomic<u32>, bitcast f32
//   0:13         storage (rw)        grad_means2d  — atomic<u32>, bitcast f32
//   0:14         storage (rw)        grad_conics   — atomic<u32>, bitcast f32
//
// Dispatch dimensions
// ───────────────────
// One workgroup per tile: ceil(W/16) × ceil(H/16).
// Workgroup size: 16×16 = 256 threads.
//
// Math
// ────
// Reverse traversal computes:
//   dL/dα = dL/dcolor · (T·c − accum/(1−α))
//   T_prev = T / (1 − α)   (recovering pre-blend transmittance)
//   dL/dc   = dL/dcolor · T · α
//   dL/d(power) = dL/dα · α   (when alpha < 0.99 cap)
//   dL/d(conic) from d(power)/d(conic)
//   dL/d(mean2d) from d(power)/d(mean)

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

// ── Workgroup shared-memory gradient accumulators ─────────────────────────────
//
// Layout of wg_grad (9 slots, bitcast f32 CAS accumulation):
//   [0] = grad_colors[gidx*3+0]   (∂L/∂color.r)
//   [1] = grad_colors[gidx*3+1]   (∂L/∂color.g)
//   [2] = grad_colors[gidx*3+2]   (∂L/∂color.b)
//   [3] = grad_opacities[gidx]    (∂L/∂opacity)
//   [4] = grad_means2d[gidx*2+0]  (∂L/∂mean.x)
//   [5] = grad_means2d[gidx*2+1]  (∂L/∂mean.y)
//   [6] = grad_conics[gidx*3+0]   (∂L/∂conic.a)
//   [7] = grad_conics[gidx*3+1]   (∂L/∂conic.b)
//   [8] = grad_conics[gidx*3+2]   (∂L/∂conic.c)
//
// Protocol per Gaussian:
//   1. Thread 0 resets all 9 slots to 0.
//   2. workgroupBarrier() — all threads see the cleared values.
//   3. Every thread that has a nonzero contribution does a CAS-loop into
//      workgroup atomic slots (fast: no DRAM round-trip).
//   4. workgroupBarrier() — all CAS loops complete.
//   5. Thread 0 issues 9 CAS-loops into global atomic buffers (one per slot).
//   6. workgroupBarrier() — global writes complete before thread 0 resets for
//      the next Gaussian.
//
// This reduces global DRAM CAS from 256×9 to 9 per Gaussian.
var<workgroup> wg_grad: array<atomic<u32>, 9u>;

/// CAS-loop accumulation into a workgroup atomic slot (bitcast f32 trick).
fn wg_atomic_add_f32(slot: u32, val: f32) {
    var old = atomicLoad(&wg_grad[slot]);
    loop {
        let nv  = bitcast<f32>(old) + val;
        let res = atomicCompareExchangeWeak(&wg_grad[slot], old, bitcast<u32>(nv));
        if res.exchanged { break; }
        old = res.old_value;
    }
}

// NOTE: WGSL prohibits passing ptr<storage,...> into functions.
// Global atomic f32 accumulation is inlined at the call sites below using
// the same CAS-loop trick as wg_atomic_add_f32.

@compute @workgroup_size(16, 16)
fn rasterize_backward(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(workgroup_id)           wid: vec3<u32>,
    @builtin(local_invocation_index) lid_flat: u32,
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
            // ── Workgroup sync for Gaussians skipped by all threads ───────
            // We still need barriers for the per-Gaussian accumulation protocol.
            // Reset and flush with zero contribution from this thread.
            if lid_flat == 0u {
                atomicStore(&wg_grad[0], 0u);
                atomicStore(&wg_grad[1], 0u);
                atomicStore(&wg_grad[2], 0u);
                atomicStore(&wg_grad[3], 0u);
                atomicStore(&wg_grad[4], 0u);
                atomicStore(&wg_grad[5], 0u);
                atomicStore(&wg_grad[6], 0u);
                atomicStore(&wg_grad[7], 0u);
                atomicStore(&wg_grad[8], 0u);
            }
            workgroupBarrier();
            // (no thread writes anything — all contributions are zero)
            workgroupBarrier();
            // (thread 0 writes nothing to global — slots are zero)
            workgroupBarrier();
            continue;
        }

        let alpha_raw = sigmoid(opacities[gaussian_idx]) * exp(power);
        let alpha = min(alpha_raw, 0.99);

        if alpha < 1.0 / 255.0 {
            // Same barrier protocol for skipped Gaussians.
            if lid_flat == 0u {
                atomicStore(&wg_grad[0], 0u);
                atomicStore(&wg_grad[1], 0u);
                atomicStore(&wg_grad[2], 0u);
                atomicStore(&wg_grad[3], 0u);
                atomicStore(&wg_grad[4], 0u);
                atomicStore(&wg_grad[5], 0u);
                atomicStore(&wg_grad[6], 0u);
                atomicStore(&wg_grad[7], 0u);
                atomicStore(&wg_grad[8], 0u);
            }
            workgroupBarrier();
            workgroupBarrier();
            workgroupBarrier();
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

        // ── Phase 1: Reset workgroup accumulators (thread 0) ──────────────
        if lid_flat == 0u {
            atomicStore(&wg_grad[0], 0u);
            atomicStore(&wg_grad[1], 0u);
            atomicStore(&wg_grad[2], 0u);
            atomicStore(&wg_grad[3], 0u);
            atomicStore(&wg_grad[4], 0u);
            atomicStore(&wg_grad[5], 0u);
            atomicStore(&wg_grad[6], 0u);
            atomicStore(&wg_grad[7], 0u);
            atomicStore(&wg_grad[8], 0u);
        }
        workgroupBarrier();

        // ── Phase 2: All threads accumulate into workgroup atomics ────────
        wg_atomic_add_f32(0u, dL_dc.x);
        wg_atomic_add_f32(1u, dL_dc.y);
        wg_atomic_add_f32(2u, dL_dc.z);
        wg_atomic_add_f32(3u, dL_dopacity);
        wg_atomic_add_f32(4u, dL_dx);
        wg_atomic_add_f32(5u, dL_dy);
        wg_atomic_add_f32(6u, dL_dconic_a);
        wg_atomic_add_f32(7u, dL_dconic_b);
        wg_atomic_add_f32(8u, dL_dconic_c);
        workgroupBarrier();

        // ── Phase 3: Thread 0 flushes workgroup totals to global atomics ──
        // All 256 pixel contributions for this Gaussian are now summed in
        // wg_grad; inlined CAS loops (WGSL forbids ptr<storage> as fn args).
        if lid_flat == 0u {
            // slot 0 → grad_colors[gaussian_idx*3+0]
            {
                let wg_val = bitcast<f32>(atomicLoad(&wg_grad[0]));
                let gidx = gaussian_idx * 3u + 0u;
                var old0 = atomicLoad(&grad_colors[gidx]);
                loop { let nv = bitcast<f32>(old0) + wg_val; let res = atomicCompareExchangeWeak(&grad_colors[gidx], old0, bitcast<u32>(nv)); if res.exchanged { break; } old0 = res.old_value; }
            }
            // slot 1 → grad_colors[gaussian_idx*3+1]
            {
                let wg_val = bitcast<f32>(atomicLoad(&wg_grad[1]));
                let gidx = gaussian_idx * 3u + 1u;
                var old1 = atomicLoad(&grad_colors[gidx]);
                loop { let nv = bitcast<f32>(old1) + wg_val; let res = atomicCompareExchangeWeak(&grad_colors[gidx], old1, bitcast<u32>(nv)); if res.exchanged { break; } old1 = res.old_value; }
            }
            // slot 2 → grad_colors[gaussian_idx*3+2]
            {
                let wg_val = bitcast<f32>(atomicLoad(&wg_grad[2]));
                let gidx = gaussian_idx * 3u + 2u;
                var old2 = atomicLoad(&grad_colors[gidx]);
                loop { let nv = bitcast<f32>(old2) + wg_val; let res = atomicCompareExchangeWeak(&grad_colors[gidx], old2, bitcast<u32>(nv)); if res.exchanged { break; } old2 = res.old_value; }
            }
            // slot 3 → grad_opacities[gaussian_idx]
            {
                let wg_val = bitcast<f32>(atomicLoad(&wg_grad[3]));
                let gidx = gaussian_idx;
                var old3 = atomicLoad(&grad_opacities[gidx]);
                loop { let nv = bitcast<f32>(old3) + wg_val; let res = atomicCompareExchangeWeak(&grad_opacities[gidx], old3, bitcast<u32>(nv)); if res.exchanged { break; } old3 = res.old_value; }
            }
            // slot 4 → grad_means2d[gaussian_idx*2+0]
            {
                let wg_val = bitcast<f32>(atomicLoad(&wg_grad[4]));
                let gidx = gaussian_idx * 2u + 0u;
                var old4 = atomicLoad(&grad_means2d[gidx]);
                loop { let nv = bitcast<f32>(old4) + wg_val; let res = atomicCompareExchangeWeak(&grad_means2d[gidx], old4, bitcast<u32>(nv)); if res.exchanged { break; } old4 = res.old_value; }
            }
            // slot 5 → grad_means2d[gaussian_idx*2+1]
            {
                let wg_val = bitcast<f32>(atomicLoad(&wg_grad[5]));
                let gidx = gaussian_idx * 2u + 1u;
                var old5 = atomicLoad(&grad_means2d[gidx]);
                loop { let nv = bitcast<f32>(old5) + wg_val; let res = atomicCompareExchangeWeak(&grad_means2d[gidx], old5, bitcast<u32>(nv)); if res.exchanged { break; } old5 = res.old_value; }
            }
            // slot 6 → grad_conics[gaussian_idx*3+0]
            {
                let wg_val = bitcast<f32>(atomicLoad(&wg_grad[6]));
                let gidx = gaussian_idx * 3u + 0u;
                var old6 = atomicLoad(&grad_conics[gidx]);
                loop { let nv = bitcast<f32>(old6) + wg_val; let res = atomicCompareExchangeWeak(&grad_conics[gidx], old6, bitcast<u32>(nv)); if res.exchanged { break; } old6 = res.old_value; }
            }
            // slot 7 → grad_conics[gaussian_idx*3+1]
            {
                let wg_val = bitcast<f32>(atomicLoad(&wg_grad[7]));
                let gidx = gaussian_idx * 3u + 1u;
                var old7 = atomicLoad(&grad_conics[gidx]);
                loop { let nv = bitcast<f32>(old7) + wg_val; let res = atomicCompareExchangeWeak(&grad_conics[gidx], old7, bitcast<u32>(nv)); if res.exchanged { break; } old7 = res.old_value; }
            }
            // slot 8 → grad_conics[gaussian_idx*3+2]
            {
                let wg_val = bitcast<f32>(atomicLoad(&wg_grad[8]));
                let gidx = gaussian_idx * 3u + 2u;
                var old8 = atomicLoad(&grad_conics[gidx]);
                loop { let nv = bitcast<f32>(old8) + wg_val; let res = atomicCompareExchangeWeak(&grad_conics[gidx], old8, bitcast<u32>(nv)); if res.exchanged { break; } old8 = res.old_value; }
            }
        }
        workgroupBarrier();
    }
}
