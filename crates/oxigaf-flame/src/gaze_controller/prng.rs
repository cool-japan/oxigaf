//! Internal xorshift64 PRNG primitives used by blink synthesis
//! ([`super::functions::gz_synthesize_blinks`]).

// ---------------------------------------------------------------------------
// Internal PRNG (xorshift64)
// ---------------------------------------------------------------------------

/// Advance the xorshift64 state and return the next pseudo-random `u64`.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    if s == 0 {
        s = 1;
    }
    *state = s;
    s
}

/// Hash a seed with a splitmix64-style step to produce a well-distributed
/// initial xorshift64 state even for small integer seeds.
#[inline]
pub(super) fn gz_seed_hash(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9_u64);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB_u64);
    x = x ^ (x >> 31);
    if x == 0 {
        1
    } else {
        x
    }
}

/// Map a `u64` to `[0, 1)` uniformly using the full 64-bit range.
#[inline]
pub(super) fn xorshift64_f32(state: &mut u64) -> f32 {
    let raw = xorshift64(state);
    // Divide by 2^64 to get uniform [0, 1).
    // Use f64 intermediate for precision, then cast to f32.
    #[allow(clippy::cast_precision_loss)]
    let v = (raw as f64) * (1.0_f64 / u64::MAX as f64);
    v as f32
}
