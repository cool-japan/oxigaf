// Backward rasterization: reverse-order tile traversal for gradient computation.
// Computes dL/d(color), dL/d(opacity), dL/d(mean2d), dL/d(conic).
//
// Purpose
// ───────
// Each workgroup owns one tile (16×16 = 256 threads) and walks that tile's
// sorted Gaussian list in reverse depth order.  Every thread visits the SAME
// Gaussian at the same loop iteration; threads whose pixel did not blend that
// Gaussian in the forward pass simply contribute zero.  Per Gaussian the 256
// per-pixel contributions are summed by a tree reduction in workgroup memory
// and a single elected thread (local_invocation_index == 0) issues the global
// CAS writes.
//
// Barrier / attribution uniformity  (do not "optimise" this away)
// ───────────────────────────────────────────────────────────────
// rasterize_fwd.wgsl stores a PER-PIXEL stopping sort index (k_stop) in
// out_n_contrib.  Driving this loop with *that per-thread value* would give
// every thread a different trip count, which
//   (a) is undefined behaviour for the workgroupBarrier() calls in the loop
//       body — WGSL requires barriers in workgroup-uniform control flow, and
//       violating it hangs tile-based/mobile GPUs, and
//   (b) would make the elected thread flush the workgroup totals onto whatever
//       Gaussian *it* happened to be visiting, silently attributing the other
//       255 threads' gradients to the wrong Gaussian.
// Per-thread validity is therefore expressed as a mask (`contributes`), never
// as a trip count or an early return.  For the same reason out-of-image threads
// on edge tiles (resolutions that are not a multiple of 16) must not return:
// they stay alive, hit every barrier, and contribute zero.
//
// What the loop bound IS allowed to be is the workgroup-wide MAXIMUM of those
// per-pixel k_stop values: one number, identical in all 256 threads.  It is
// computed by a max tree-reduction into wg_end and then read back through
// `workgroupUniformLoad`, the WGSL builtin whose whole purpose is to yield a
// value the uniformity analysis accepts as workgroup-uniform (a plain
// `wg_end[0]` read is classified non-uniform and Naga rejects using it around a
// barrier).  Skipping the tail is exactly equivalent to walking it: above the
// tile-wide max, `contributes` is false in all 256 threads, so T, accum_color
// and the reduction totals are all untouched and the flush is skipped anyway.
//
// Gradient reduction
// ──────────────────
// The 9 gradient components of all 256 threads are summed by a log2(256) = 8
// step tree reduction over plain (non-atomic) workgroup memory, after which
// thread 0 issues exactly 9 global CAS loops.  That is 9 DRAM CAS per Gaussian
// per tile instead of 256×9.  The previous revision used 9 workgroup atomics
// instead of the tree: 256-way same-address CAS contention is O(n²) in retries
// (~2304 contended CAS per Gaussian), which the tree replaces with 8 barriers
// and 8 conflict-free shared-memory steps.  Where subgroup operations are
// available, `subgroupAdd` would shrink this further.
//
// Remaining cost, deliberately left in place: entries BELOW the tile-wide max
// k_stop that this particular pixel skipped (its own k_stop is lower, or the
// power/alpha tests rejected the Gaussian) still pay a full 8-step reduction.
// Only a per-thread trip count could avoid that, and per (a)/(b) above it is not
// available.  Such iterations flush nothing — Phase 3's all-zero test elides the
// 9 global CAS round-trips — so the residual cost is shared-memory traffic, not
// DRAM traffic.
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
// Reverse traversal computes, for each Gaussian k that the pixel blended:
//   T_before = T_after / (1 − α)      (recovering pre-blend transmittance)
//   dL/dα    = dL/dcolor · (T_before·c − C_behind)
//              + (−T_final / (1 − α)) · (background · dL/dcolor)
//   dL/dc    = dL/dcolor · T_before · α
//   dL/d(power)  = dL/dα · α_raw      (zero when α is capped at 0.99)
//   dL/d(conic), dL/d(mean2d) from d(power)/d(conic), d(power)/d(mean)
// The background term is required because the forward pass finishes with
// `color += T_final · background` (rasterize_fwd.wgsl), so every α influences
// the loss through T_final as well as through its own blend weight.

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
    // Declared to match the buffer layout the forward pass writes, and
    // deliberately never read here: the early-termination decision was already
    // taken by rasterize_fwd.wgsl using this same value, and its outcome
    // reaches us as the per-pixel stopping index in out_n_contrib.  Re-testing
    // `T < transmittance_threshold` during the reverse walk would double-apply
    // the cut-off against a reconstructed T, and would silently disagree with
    // the forward pass on the boundary Gaussian.
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

// ── Workgroup gradient partials (one entry per thread, no atomics) ────────────
//
// Packed so the tree reduction below is 2 vec4 adds + 1 scalar add per step:
//   wg_grad_a[i] = (∂L/∂color.r, ∂L/∂color.g, ∂L/∂color.b, ∂L/∂opacity)
//   wg_grad_b[i] = (∂L/∂mean.x,  ∂L/∂mean.y,  ∂L/∂conic.a, ∂L/∂conic.b)
//   wg_grad_c[i] =  ∂L/∂conic.c
//
// Storage cost: 2 × 256 × 16 B + 256 × 4 B = 9216 B, plus wg_end's 1024 B for
// a total of 10240 B — within the 16384 B
// `max_compute_workgroup_storage_size` of the default WebGPU limits.
var<workgroup> wg_grad_a: array<vec4<f32>, 256>;
var<workgroup> wg_grad_b: array<vec4<f32>, 256>;
var<workgroup> wg_grad_c: array<f32, 256>;

// Scratch for the one-off max-reduction of the per-pixel stopping indices.
// Written once before the main loop and never touched again, so it cannot race
// with the per-Gaussian gradient reduction that reuses the same barriers.
var<workgroup> wg_end: array<u32, 256>;

// NOTE: WGSL prohibits passing ptr<storage,...> into functions, so the global
// atomic f32 accumulation is inlined at the call sites below using a CAS loop.

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

    // Do NOT return for out-of-image threads: the loop below contains
    // workgroupBarrier() calls that all 256 threads must execute the same
    // number of times.  Mask instead.
    let in_bounds = px < W && py < H;
    let pixel_idx = select(0u, py * W + px, in_bounds);
    let pixel_f = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

    let tile_x = wid.x;
    let tile_y = wid.y;
    let tile_id = tile_y * uniforms.tile_grid.x + tile_x;

    // Workgroup-uniform: derived from `wid` only.  `range_start` is the loop's
    // lower bound directly; `range_end` bounds the per-thread `effective_end`
    // that the max-reduction below turns into the uniform upper bound.
    let range = tile_ranges[tile_id];
    let range_start = range.x;
    let range_end = range.y;

    let dL_dcolor_out = grad_output[pixel_idx].xyz;
    let final_T = out_transmittance[pixel_idx];
    // Forward adds `T_final * background`; every α feeds the loss through it.
    let bg_dot_dpixel = dot(uniforms.background, dL_dcolor_out);

    // `out_n_contrib` holds the absolute stopping sort index (k_stop) written by
    // the forward shader, NOT a count. Entries at or beyond it were never
    // blended by this pixel, so this thread contributes zero for them.
    let effective_end = min(out_n_contrib[pixel_idx], range_end);
    let has_work = in_bounds && effective_end > range_start;

    // ── Workgroup-uniform upper bound for the reverse walk ────────────────
    // Each thread proposes its own exclusive end (range_start when it has no
    // work at all, so it can never raise the max); the maximum over the
    // workgroup is the highest sort index ANY pixel of this tile blended.
    // Every entry above it was skipped by all 256 threads, so walking it would
    // leave T, accum_color and the reduction totals untouched and flush
    // nothing — dropping those iterations is observationally identical.
    wg_end[lid_flat] = select(range_start, effective_end, has_work);
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if lid_flat < stride {
            wg_end[lid_flat] = max(wg_end[lid_flat], wg_end[lid_flat + stride]);
        }
        workgroupBarrier();
    }
    // `workgroupUniformLoad` both synchronises and returns a value that WGSL's
    // uniformity analysis treats as workgroup-uniform — the prerequisite for
    // using it as the trip count of a loop whose body contains barriers.
    let tile_end = workgroupUniformLoad(&wg_end[0]);

    // Reconstruct transmittance in reverse order.
    var T = final_T;
    var accum_color = vec3<f32>(0.0);

    var k = tile_end;
    while k > range_start {
        k -= 1u;

        let gaussian_idx = sort_values[k];
        let mean = means2d[gaussian_idx];
        let conic = conics[gaussian_idx];
        let raw_opacity = opacities[gaussian_idx];

        let d = pixel_f - mean;
        let power = -0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y);
        let sig = sigmoid(raw_opacity);
        let alpha_raw = sig * exp(power);
        let alpha = min(alpha_raw, 0.99);

        // Same rejection tests the forward shader applied, plus this thread's
        // own early-termination bound.  Written as an accept-test rather than
        // the forward's `power > 0.0 || power < -4.0` reject-test: the two agree
        // on every finite `power`, and differ only when `power` is NaN, where
        // both of the forward's comparisons are false so it does NOT skip.  A
        // NaN here contributes zero instead of poisoning the whole tile's
        // reduction with a NaN that the elected thread would then CAS into the
        // global gradient buffers.
        let contributes = has_work
            && k < effective_end
            && power <= 0.0
            && power >= -4.0
            && alpha >= 1.0 / 255.0;

        var dL_dc = vec3<f32>(0.0);
        var dL_dopacity = 0.0;
        var dL_dmean = vec2<f32>(0.0);
        var dL_dconic = vec3<f32>(0.0);

        if contributes {
            // Recover T before this Gaussian was blended.
            T /= (1.0 - alpha);

            let c = colors[gaussian_idx].xyz;

            // dL/dα: blend term + background term (through T_final).
            let dL_dalpha = dot(dL_dcolor_out, T * c - accum_color / (1.0 - alpha + 1e-7))
                + (-final_T / (1.0 - alpha)) * bg_dot_dpixel;

            dL_dc = dL_dcolor_out * T * alpha;

            // Colour seen behind this Gaussian (T is now T_before).
            accum_color += alpha * T * c;

            // When α is capped at 0.99 the cap has zero derivative, so both
            // dL/d(power) and dL/d(opacity) must be zeroed.
            let capped = alpha_raw >= 0.99;
            let dsig = sig * (1.0 - sig);
            dL_dopacity = select(dL_dalpha * exp(power) * dsig, 0.0, capped);
            let dL_dpower = select(dL_dalpha * alpha_raw, 0.0, capped);

            // d(power)/d(mean.x) = conic.x*d.x + conic.y*d.y
            // (power = -0.5·Q, dQ/d(d.x) = 2·(conic.x·d.x + conic.y·d.y),
            //  d(d.x)/d(mean.x) = -1, so the two sign flips cancel.)
            dL_dmean = vec2<f32>(
                dL_dpower * (conic.x * d.x + conic.y * d.y),
                dL_dpower * (conic.y * d.x + conic.z * d.y),
            );

            // d(power)/d(conic.x) = -0.5·d.x², d/d(conic.y) = -d.x·d.y,
            // d/d(conic.z) = -0.5·d.y².
            dL_dconic = vec3<f32>(
                dL_dpower * (-0.5) * d.x * d.x,
                dL_dpower * (-1.0) * d.x * d.y,
                dL_dpower * (-0.5) * d.y * d.y,
            );
        }

        // ── Phase 1: every thread publishes its partial (overwrite, no reset) ──
        wg_grad_a[lid_flat] = vec4<f32>(dL_dc, dL_dopacity);
        wg_grad_b[lid_flat] = vec4<f32>(dL_dmean, dL_dconic.x, dL_dconic.y);
        wg_grad_c[lid_flat] = dL_dconic.z;
        workgroupBarrier();

        // ── Phase 2: tree reduction, 256 → 1 in log2(256) = 8 conflict-free
        //             steps. `stride` is workgroup-uniform, so the barrier at
        //             the end of the body stays in uniform control flow. ──────
        for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
            if lid_flat < stride {
                wg_grad_a[lid_flat] += wg_grad_a[lid_flat + stride];
                wg_grad_b[lid_flat] += wg_grad_b[lid_flat + stride];
                wg_grad_c[lid_flat] += wg_grad_c[lid_flat + stride];
            }
            workgroupBarrier();
        }

        // ── Phase 3: thread 0 flushes the tile totals to global atomics ───────
        // All 256 pixel contributions for THIS Gaussian (the same Gaussian in
        // every thread, which is what makes the flush valid) are now in slot 0.
        // Every thread reads it — a shared-memory broadcast, and slot 0 is not
        // written again until the next iteration — so the cheap all-zero test
        // below stays branch-uniform. The adds are inlined CAS loops because
        // WGSL forbids ptr<storage> as a function argument.
        let tot_a = wg_grad_a[0u];
        let tot_b = wg_grad_b[0u];
        let tot_c = wg_grad_c[0u];
        // Gaussians past every pixel's k_stop, or rejected by the power/alpha
        // tests in all 256 threads, sum to exactly zero — skip their 9 global
        // CAS round-trips entirely.
        let any_nonzero = any(tot_a != vec4<f32>(0.0))
            || any(tot_b != vec4<f32>(0.0))
            || tot_c != 0.0;

        if lid_flat == 0u && any_nonzero {
            // grad_colors[gaussian_idx*3 + 0..2]
            {
                let gidx = gaussian_idx * 3u + 0u;
                var old0 = atomicLoad(&grad_colors[gidx]);
                loop { let nv = bitcast<f32>(old0) + tot_a.x; let res = atomicCompareExchangeWeak(&grad_colors[gidx], old0, bitcast<u32>(nv)); if res.exchanged { break; } old0 = res.old_value; }
            }
            {
                let gidx = gaussian_idx * 3u + 1u;
                var old1 = atomicLoad(&grad_colors[gidx]);
                loop { let nv = bitcast<f32>(old1) + tot_a.y; let res = atomicCompareExchangeWeak(&grad_colors[gidx], old1, bitcast<u32>(nv)); if res.exchanged { break; } old1 = res.old_value; }
            }
            {
                let gidx = gaussian_idx * 3u + 2u;
                var old2 = atomicLoad(&grad_colors[gidx]);
                loop { let nv = bitcast<f32>(old2) + tot_a.z; let res = atomicCompareExchangeWeak(&grad_colors[gidx], old2, bitcast<u32>(nv)); if res.exchanged { break; } old2 = res.old_value; }
            }
            // grad_opacities[gaussian_idx]
            {
                let gidx = gaussian_idx;
                var old3 = atomicLoad(&grad_opacities[gidx]);
                loop { let nv = bitcast<f32>(old3) + tot_a.w; let res = atomicCompareExchangeWeak(&grad_opacities[gidx], old3, bitcast<u32>(nv)); if res.exchanged { break; } old3 = res.old_value; }
            }
            // grad_means2d[gaussian_idx*2 + 0..1]
            {
                let gidx = gaussian_idx * 2u + 0u;
                var old4 = atomicLoad(&grad_means2d[gidx]);
                loop { let nv = bitcast<f32>(old4) + tot_b.x; let res = atomicCompareExchangeWeak(&grad_means2d[gidx], old4, bitcast<u32>(nv)); if res.exchanged { break; } old4 = res.old_value; }
            }
            {
                let gidx = gaussian_idx * 2u + 1u;
                var old5 = atomicLoad(&grad_means2d[gidx]);
                loop { let nv = bitcast<f32>(old5) + tot_b.y; let res = atomicCompareExchangeWeak(&grad_means2d[gidx], old5, bitcast<u32>(nv)); if res.exchanged { break; } old5 = res.old_value; }
            }
            // grad_conics[gaussian_idx*3 + 0..2]
            {
                let gidx = gaussian_idx * 3u + 0u;
                var old6 = atomicLoad(&grad_conics[gidx]);
                loop { let nv = bitcast<f32>(old6) + tot_b.z; let res = atomicCompareExchangeWeak(&grad_conics[gidx], old6, bitcast<u32>(nv)); if res.exchanged { break; } old6 = res.old_value; }
            }
            {
                let gidx = gaussian_idx * 3u + 1u;
                var old7 = atomicLoad(&grad_conics[gidx]);
                loop { let nv = bitcast<f32>(old7) + tot_b.w; let res = atomicCompareExchangeWeak(&grad_conics[gidx], old7, bitcast<u32>(nv)); if res.exchanged { break; } old7 = res.old_value; }
            }
            {
                let gidx = gaussian_idx * 3u + 2u;
                var old8 = atomicLoad(&grad_conics[gidx]);
                loop { let nv = bitcast<f32>(old8) + tot_c; let res = atomicCompareExchangeWeak(&grad_conics[gidx], old8, bitcast<u32>(nv)); if res.exchanged { break; } old8 = res.old_value; }
            }
        }
        // Keeps the next iteration's Phase 1 stores from racing thread 0's
        // reads of slot 0.
        workgroupBarrier();
    }
}
