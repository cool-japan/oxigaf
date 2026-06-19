// Inclusive prefix sum (Blelloch-style work-efficient scan).
//
// Purpose
// ───────
// Computes an inclusive prefix sum (scan) over a u32 array in two phases:
//   Phase 1 — each workgroup scans its 512-element chunk and writes its total
//             to block_sums[].
//   Phase 2 — block offsets are propagated back via prefix_sum_add.wgsl.
//
// Bindings
// ────────
// group:binding  type              description
//    0:0         storage (ro)      input       — source u32 array
//    0:1         storage (rw)      output      — scanned u32 array
//    0:2         uniform (vec4u)   params      — params.x = element count
//    0:3         storage (rw)      block_sums  — per-workgroup totals
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(count / 512) workgroups × 256 threads each.
// Each workgroup processes 512 elements (2 per thread).
//
// Math
// ────
// Blelloch two-phase scan in shared memory (512-element tile):
//   Up-sweep:   reduce pairs into a binary tree of partial sums.
//   Down-sweep: propagate partial sums back through the tree.
// Result: output[i] = sum(input[0..=i]).

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = count
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn prefix_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = params.x;
    let local_id = lid.x;
    let global_id = gid.x;
    let offset_base = wid.x * 512u;

    // Load two elements per thread
    let ai = local_id;
    let bi = local_id + 256u;

    var val_a = 0u;
    var val_b = 0u;
    if offset_base + ai < n {
        val_a = input[offset_base + ai];
        temp[ai] = val_a;
    } else {
        temp[ai] = 0u;
    }
    if offset_base + bi < n {
        val_b = input[offset_base + bi];
        temp[bi] = val_b;
    } else {
        temp[bi] = 0u;
    }

    // Up-sweep (reduce)
    var stride = 1u;
    for (var d = 256u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if local_id < d {
            let a_idx = stride * (2u * local_id + 1u) - 1u;
            let b_idx = stride * (2u * local_id + 2u) - 1u;
            if b_idx < 512u {
                temp[b_idx] += temp[a_idx];
            }
        }
        stride *= 2u;
    }

    // Save the total for this workgroup before clearing root
    workgroupBarrier();
    if local_id == 0u {
        let total = temp[511u];
        block_sums[wid.x] = total;
        temp[511u] = 0u;
    }

    // Down-sweep
    for (var d = 1u; d < 512u; d *= 2u) {
        stride >>= 1u;
        workgroupBarrier();
        if local_id < d {
            let a_idx = stride * (2u * local_id + 1u) - 1u;
            let b_idx = stride * (2u * local_id + 2u) - 1u;
            if b_idx < 512u {
                let t = temp[a_idx];
                temp[a_idx] = temp[b_idx];
                temp[b_idx] += t;
            }
        }
    }

    workgroupBarrier();

    // Write output (convert exclusive to inclusive by adding input)
    if offset_base + ai < n {
        output[offset_base + ai] = temp[ai] + val_a;
    }
    if offset_base + bi < n {
        output[offset_base + bi] = temp[bi] + val_b;
    }
}
