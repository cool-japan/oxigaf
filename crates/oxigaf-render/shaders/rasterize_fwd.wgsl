// Forward rasterization: per-tile alpha-blending (front-to-back).
//
// Purpose
// ───────
// Rasterizes pre-projected 3D Gaussians onto the screen using sorted, tile-based
// alpha-blending. Each workgroup processes one tile (16×16 pixels); threads within
// the workgroup collaboratively load Gaussian data into workgroup shared memory
// (BATCH_SIZE=256 Gaussians at a time) before each thread independently blends
// its pixel's contribution from shared memory.
//
// Bindings
// ────────
// group:binding  type              description
//    0:0         uniform           Camera/viewport/render uniforms
//    0:1         storage (ro)      means2d  — projected 2D Gaussian centres
//    0:2         storage (ro)      conics   — inverse-cov2D (packed vec3 + pad)
//    0:3         storage (ro)      colors   — per-Gaussian RGB (packed vec3 + pad)
//    0:4         storage (ro)      opacities — pre-sigmoid opacities
//    0:5         storage (ro)      depths   — view-space depth per Gaussian
//    0:6         storage (ro)      tile_ranges — [start,end) sort index per tile
//    0:7         storage (ro)      sort_values — Gaussian indices sorted by (tile,depth)
//    0:8         storage (rw)      out_color   — RGBA output image
//    0:9         storage (rw)      out_depth   — depth output image
//   0:10         storage (rw)      out_transmittance — transmittance T per pixel
//   0:11         storage (rw)      out_n_contrib — absolute stopping sort index per pixel
//   0:12         storage (ro)      normals  — per-Gaussian normal (optional)
//   0:13         storage (rw)      out_normals — normal output image (optional)
//
// Dispatch dimensions
// ───────────────────
// Each workgroup covers one 16×16 tile.
// Grid: ceil(W/16) × ceil(H/16) workgroups.
//
// Math
// ────
// For each pixel (px,py) in the tile, iterates the sorted Gaussian list in
// depth order (front-to-back). For each Gaussian:
//   power = −½ · (conic.x·dx² + 2·conic.y·dx·dy + conic.z·dy²)
//   alpha = min(sigmoid(opacity) · exp(power), 0.99)
//   color += T · alpha · c;  T *= (1 − alpha)
// Loop stops when T < uniforms.transmittance_threshold (RasterConfig::
// transmittance_threshold, default 1/255), and the stopping index is published
// as out_n_contrib so rasterize_bwd.wgsl can bound its reverse traversal by
// exactly the entries this pixel blended (it clamps to `out_n_contrib` rather
// than re-testing a threshold of its own, so raising the configured threshold
// stays consistent between the two passes).
//
// The `alpha < 1/255` rejection below is a different, deliberately fixed
// constant — the 8-bit quantisation floor of a colour contribution, not a
// transmittance budget — and rasterize_bwd.wgsl repeats it verbatim so the two
// passes accept the same set of Gaussians.
//
// Barrier uniformity
// ──────────────────
// Threads whose pixel lies outside the image (edge tiles at resolutions that
// are not a multiple of 16) must NOT return early: the batch loop below
// contains workgroupBarrier() calls that every thread of the workgroup has to
// execute the same number of times.  They stay alive, take part in the
// cooperative load, and only their final stores are suppressed.

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

// ── Workgroup shared memory for cooperative Gaussian data loading ─────────────
//
// The inner alpha-blending loop accesses up to thousands of Gaussians from
// global storage. To amortise memory-bandwidth cost, all 256 threads in the
// workgroup collaboratively prefetch BATCH_SIZE Gaussians into shared memory
// before processing.  Each batch requires two workgroupBarrier() calls:
//   1. After the load phase  — ensures all data is visible before reads.
//   2. After the process phase — prevents the next load from overwriting
//      data that slower threads may still be consuming.
//
// Data cached per batch: sort index (→ gaussian_idx), mean2d, conic, color,
// opacity, depth, normal.

const BATCH_SIZE: u32 = 256u;

var<workgroup> wg_gidx:    array<u32,         256>;
var<workgroup> wg_mean:    array<vec2<f32>,   256>;
var<workgroup> wg_conic:   array<vec4<f32>,   256>;
var<workgroup> wg_color:   array<vec4<f32>,   256>;
var<workgroup> wg_opacity: array<f32,         256>;
var<workgroup> wg_depth:   array<f32,         256>;
var<workgroup> wg_normal:  array<vec4<f32>,   256>;

@compute @workgroup_size(16, 16)
fn rasterize_forward(
    @builtin(global_invocation_id)    gid: vec3<u32>,
    @builtin(workgroup_id)            wid: vec3<u32>,
    @builtin(local_invocation_id)     lid: vec3<u32>,
) {
    let px = gid.x;
    let py = gid.y;
    let W = u32(uniforms.viewport.x);
    let H = u32(uniforms.viewport.y);

    // Do NOT return here. `range_start`/`range_end` below come from
    // `tile_ranges[tile_id]` with `tile_id` derived from `wid`, so the batch
    // loop's trip count is workgroup-uniform and every thread must reach both
    // workgroupBarrier() calls. Returning early from the out-of-image threads
    // of an edge tile is undefined behaviour, and in practice leaves the shared
    // slots owned by those threads holding stale data that the surviving
    // threads then blend as phantom Gaussians.
    let in_bounds = px < W && py < H;

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
    // Track the absolute stopping index so the backward pass knows
    // exactly which sort entries were actually blended.  This is an INDEX into
    // the tile's sort range, not a count of blended Gaussians — a count would
    // not let the backward pass identify *which* entries were skipped, because
    // the power/alpha rejections below are per-pixel and leave gaps.  The
    // buffer it lands in is named `out_n_contrib` for historical reasons.
    var k_stop = range_end;

    // Flat thread index within the 16×16 workgroup; used to assign cooperative
    // load slots.  Values 0..255 correspond to unique slots in the shared arrays.
    let local_idx = lid.y * 16u + lid.x;

    // ── Batched tile loop with shared-memory caching ──────────────────────
    //
    // We iterate over the Gaussian list in chunks of BATCH_SIZE. All 256
    // threads cooperatively load one chunk, then each thread blends its pixel
    // against that chunk from shared memory.
    //
    // Threads that have already crossed the transmittance threshold (done==true)
    // still participate in the load phase and barriers to keep the workgroup in
    // sync — they just skip the inner processing loop.  Out-of-image threads
    // start out "done" for exactly the same reason: they must keep hitting the
    // barriers, but there is no pixel for them to blend.
    var done = !in_bounds;
    var k_base = range_start;

    while k_base < range_end {
        let batch_end  = min(k_base + BATCH_SIZE, range_end);
        let batch_size = batch_end - k_base;

        // ── Cooperative load phase ────────────────────────────────────────
        if local_idx < batch_size {
            let sort_k = k_base + local_idx;
            let g_idx  = sort_values[sort_k];
            wg_gidx[local_idx]    = g_idx;
            wg_mean[local_idx]    = means2d[g_idx];
            wg_conic[local_idx]   = conics[g_idx];
            wg_color[local_idx]   = colors[g_idx];
            wg_opacity[local_idx] = opacities[g_idx];
            wg_depth[local_idx]   = depths[g_idx];
            wg_normal[local_idx]  = normals[g_idx];
        }
        workgroupBarrier();

        // ── Per-pixel processing phase (read from shared memory) ──────────
        if !done {
            for (var b = 0u; b < batch_size; b++) {
                if T < uniforms.transmittance_threshold {
                    k_stop = k_base + b;
                    done = true;
                    break;
                }

                let mean  = wg_mean[b];
                let conic = wg_conic[b];

                // Evaluate 2D Gaussian
                let d     = pixel_f - mean;
                let power = -0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y);

                if power > 0.0 || power < -4.0 {
                    continue;
                }

                let alpha_raw = sigmoid(wg_opacity[b]) * exp(power);
                let alpha     = min(alpha_raw, 0.99);

                if alpha < 1.0 / 255.0 {
                    continue;
                }

                let c      = wg_color[b].xyz;
                let weight = T * alpha;
                color     += weight * c;
                depth_acc += weight * wg_depth[b];

                // Accumulate normals if output is enabled (bit 1 of output_flags)
                if (uniforms.output_flags & 2u) != 0u {
                    normal_acc += weight * wg_normal[b].xyz;
                }

                T *= (1.0 - alpha);
            }
        }

        // Barrier before next batch overwrites shared memory.
        workgroupBarrier();
        k_base += BATCH_SIZE;
    }

    // Add background
    color += T * uniforms.background;

    // Only the threads that own a real pixel store anything; the rest exist
    // purely to keep the barriers above workgroup-uniform.
    if in_bounds {
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
}
