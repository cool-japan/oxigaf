//! Error type for light probe operations.

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by light probe operations.
#[derive(Debug, thiserror::Error)]
pub enum LightProbeError {
    /// Invalid cubemap resolution (must be power of 2 and >= 4).
    #[error("Invalid cubemap resolution {res}: must be power of 2 and >= 4")]
    InvalidResolution { res: u32 },

    /// Buffer length mismatch.
    #[error("Buffer length mismatch: expected {expected}, got {got}")]
    BufferMismatch { expected: usize, got: usize },

    /// Empty probe list supplied.
    #[error("Empty probe list")]
    EmptyProbeList,

    /// SH order not in {1, 2, 3}.
    #[error("Invalid SH order {order}: must be 1, 2, or 3")]
    InvalidOrder { order: usize },

    /// Zero-length direction vector.
    #[error("Invalid direction vector (zero length)")]
    ZeroDirection,

    /// Zero-width or zero-height image supplied to a sampling function.
    #[error("Invalid image dimensions {width}x{height}: both must be > 0")]
    InvalidImageDimensions { width: u32, height: u32 },

    /// Too many probes for the configured limit.
    #[error("Too many probes: {count} exceeds LightProbeConfig::max_probes ({max})")]
    TooManyProbes { count: usize, max: usize },
}
