// Radix sort scatter pass: place elements at their sorted positions.
//
// Purpose
// ───────
// Second pass of GPU radix sort.  Using the prefix-summed histogram from
// the histogram pass, each thread writes its key and value to the globally
// correct output position, producing a stably sorted output array for the
// selected 4-bit digit.
//
// Bindings
// ────────
// See struct/binding declarations below.  Typically:
//   input keys+values, prefix-summed histogram, output keys+values, bit-shift.
//
// Dispatch dimensions
// ───────────────────
// 1D: ceil(num_elements / workgroup_size) workgroups.
//
// Math
// ────
// pos = histogram_prefix[digit * num_wg + wid] + local_rank.
// out_key[pos] = in_key[i];  out_val[pos] = in_val[i].
// The starting position for (digit, wg_id) = prefix[d*nwg + wg] - count[d*nwg + wg].
//
// Local rank via digit ballot
// ───────────────────────────
// `local_rank` is the number of LOWER-INDEXED lanes in this workgroup that
// carry the same digit; adding it to the workgroup's global start is what makes
// the sort stable.  The obvious implementation — scan `wg_digits[0 .. lane]`
// and count matches — costs O(256) shared-memory reads per lane, i.e. ~32768
// reads per workgroup per pass, and it is the slowest part of the scatter.
//
// Instead each lane sets one bit in a per-digit 256-bit ballot mask
// (16 digits × 8 words of 32 bits = 128 u32 = 512 B of workgroup storage — less
// than the 1 KiB the old `wg_digits` array needed), and the rank is then the
// population count of the bits strictly below this lane in its own digit's
// mask.  That is at most 8 loads + 8 `countOneBits` per lane regardless of
// workgroup size: O(256) → O(8).
//
// The result is bit-for-bit identical to the serial scan, including the
// treatment of out-of-range lanes: those never set a bit, so they can never be
// counted as predecessors of a real element.
//
// Barrier discipline
// ──────────────────
// Both workgroupBarrier() calls must be executed by every lane, so the
// out-of-range early return has to come AFTER the second one.  Lanes that are
// past `params.count` still zero their slice of the ballot table and still hit
// both barriers; they simply skip the atomicOr and then return.

struct SortParams {
    count: u32,
    pass_idx: u32,
    radix_bits: u32,
    num_workgroups: u32,
};

@group(0) @binding(0) var<storage, read> keys_in: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> values_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(4) var<storage, read> histogram: array<u32>;
@group(0) @binding(5) var<storage, read> histogram_prefix: array<u32>;
@group(0) @binding(6) var<uniform> params: SortParams;

fn extract_digit(key: vec2<u32>, pass_idx: u32) -> u32 {
    let bit_offset = pass_idx * 4u;
    var word: u32;
    if bit_offset < 32u {
        word = key.y;
    } else {
        word = key.x;
    }
    let shift = bit_offset % 32u;
    return (word >> shift) & 0xFu;
}

// Number of distinct digits produced by `extract_digit` (4-bit radix).
const RADIX_DIGITS: u32 = 16u;
// 256 lanes / 32 bits per word = 8 words to hold one bit per lane.
const BALLOT_WORDS: u32 = 8u;

// Per-digit lane ballot: wg_ballot[d * BALLOT_WORDS + w] bit b is set iff lane
// (w * 32 + b) of this workgroup carries digit d.  Written with atomicOr
// because up to 32 lanes contend for the same word.
var<workgroup> wg_ballot: array<atomic<u32>, 128>; // RADIX_DIGITS * BALLOT_WORDS

@compute @workgroup_size(256)
fn radix_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let idx = gid.x;
    let nwg = params.num_workgroups;
    let lane = lid.x;

    // ── Clear the ballot table ────────────────────────────────────────────
    // 128 words, one per lane for the first 128 lanes.  Every lane reaches the
    // barrier, including the ones this dispatch has no element for.
    if lane < RADIX_DIGITS * BALLOT_WORDS {
        atomicStore(&wg_ballot[lane], 0u);
    }
    workgroupBarrier();

    // ── Cast this lane's ballot ───────────────────────────────────────────
    // Out-of-range lanes must NOT write: `extract_digit` was never called for
    // them, and a sentinel digit of 16 would index past the end of the table.
    // Leaving their bit clear is exactly the right semantics — they are not
    // predecessors of anything.
    let in_range = idx < params.count;
    var digit = 0u;
    if in_range {
        digit = extract_digit(keys_in[idx], params.pass_idx);
        atomicOr(&wg_ballot[digit * BALLOT_WORDS + (lane >> 5u)], 1u << (lane & 31u));
    }
    workgroupBarrier();

    // Safe to leave only now: both barriers above are behind every lane.
    if !in_range {
        return;
    }

    // ── Local rank = popcount of same-digit lanes strictly below this one ──
    // Whole words below this lane's word, then the partial word masked to the
    // bits under this lane.  `lane & 31u` is at most 31, so the shift below is
    // always well defined and `below_mask` is never the full word.
    let base = digit * BALLOT_WORDS;
    let my_word = lane >> 5u;
    var local_rank = 0u;
    for (var w = 0u; w < my_word; w++) {
        local_rank += countOneBits(atomicLoad(&wg_ballot[base + w]));
    }
    let below_mask = (1u << (lane & 31u)) - 1u;
    local_rank += countOneBits(atomicLoad(&wg_ballot[base + my_word]) & below_mask);

    // Look up global starting position for this (digit, workgroup)
    let hist_idx = digit * nwg + wid.x;
    let inclusive_prefix = histogram_prefix[hist_idx];
    let count_here = histogram[hist_idx];
    let global_start = inclusive_prefix - count_here;

    let dest = global_start + local_rank;
    if dest < params.count {
        keys_out[dest] = keys_in[idx];
        values_out[dest] = values_in[idx];
    }
}
