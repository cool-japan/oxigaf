//! Error type and small private helpers shared across the domain-adaptation
//! submodules: a dependency-free xorshift64 PRNG, a sigmoid, and the shared
//! feature-matrix shape check.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by domain-adaptation routines.
#[derive(Debug, Error)]
pub enum DomainAdaptationError {
    #[error("Empty features")]
    EmptyFeatures,
    #[error("Dimension mismatch: source has {src}, target has {tgt}")]
    DimensionMismatch { src: usize, tgt: usize },
    #[error("Invalid kernel bandwidth: {bw}")]
    InvalidBandwidth { bw: f32 },
    #[error("Batch size mismatch: source={src}, target={tgt}")]
    BatchMismatch { src: usize, tgt: usize },
    #[error("Invalid config: {reason}")]
    InvalidConfig { reason: String },
}

// ---------------------------------------------------------------------------
// Internal PRNG (xorshift64) — no `rand` crate
// ---------------------------------------------------------------------------

/// Advance the xorshift64 state by one step. Returns new state.
/// Invariant: state must never be 0 on entry; if 0 is produced it is reset to 1.
#[inline]
pub(super) fn da_xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Sample a uniform f32 in [0, 1) from xorshift64 state.
#[inline]
pub(super) fn da_xorshift_f32(state: &mut u64) -> f32 {
    let raw = da_xorshift64(state);
    // use the top 24 bits for the mantissa of an f32
    (raw >> 40) as f32 / (1u64 << 24) as f32
}

/// Sample a uniform f32 in [lo, hi) from xorshift64 state.
#[inline]
pub(super) fn da_xorshift_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + da_xorshift_f32(state) * (hi - lo)
}

// ---------------------------------------------------------------------------
// Private sigmoid helper (avoids re-exporting a bare `sigmoid`)
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn da_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Validate that `buf` has exactly `n * d` elements.
///
/// Every function below that takes a raw `&[f32]` feature matrix alongside
/// separate `n`/`d` parameters (rather than a validated [`DomainBatch`])
/// calls this first, so a caller-supplied `n`/`d` that does not match the
/// buffer produces a [`DomainAdaptationError::DimensionMismatch`] instead of
/// an out-of-bounds slice panic.
#[inline]
pub(super) fn da_check_matrix(
    buf: &[f32],
    n: usize,
    d: usize,
) -> Result<(), DomainAdaptationError> {
    let expected = n * d;
    if buf.len() != expected {
        return Err(DomainAdaptationError::DimensionMismatch {
            src: buf.len(),
            tgt: expected,
        });
    }
    Ok(())
}
