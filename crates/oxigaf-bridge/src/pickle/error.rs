//! Errors raised while reading pickle streams and the containers around them.

use thiserror::Error;

/// An error encountered decoding a pickle stream or a `.pt`/`.pkl`
/// container.
///
/// Every variant names a concrete defect in the *input*: this reader never
/// panics on malformed data, so a corrupt or hostile checkpoint always
/// surfaces here rather than aborting the process.
#[derive(Debug, Error)]
pub enum PickleError {
    /// The stream ended in the middle of an opcode or its payload.
    #[error("truncated pickle stream at byte offset {offset}")]
    Truncated {
        /// Byte offset at which the stream ran out.
        offset: usize,
    },

    /// An opcode popped more values than the stack held.
    #[error("pickle stack underflow at byte offset {offset}")]
    StackUnderflow {
        /// Byte offset of the offending opcode.
        offset: usize,
    },

    /// `STOP` was reached with more than one value left on the stack.
    #[error("pickle ended with {remaining} extra value(s) on the stack")]
    UnbalancedStack {
        /// How many values were left behind.
        remaining: usize,
    },

    /// A `GET`-family opcode referenced a memo key that was never `PUT`.
    #[error("pickle referenced memo key {key}, which was never stored")]
    MemoMiss {
        /// The missing key.
        key: u32,
    },

    /// An opcode this reader does not implement.
    #[error("unsupported pickle opcode 0x{opcode:02x} at byte offset {offset}")]
    UnsupportedOpcode {
        /// The raw opcode byte.
        opcode: u8,
        /// Byte offset at which it appeared.
        offset: usize,
    },

    /// A `PROTO` opcode declared a protocol newer than 5.
    #[error("unsupported pickle protocol version {version} (this reader supports 0-5)")]
    UnsupportedProtocol {
        /// The declared version.
        version: u8,
    },

    /// A protocol-0 text literal did not parse.
    #[error("malformed {opcode} literal at byte offset {offset}")]
    MalformedLiteral {
        /// Opcode name, e.g. `INT`.
        opcode: &'static str,
        /// Byte offset of the literal.
        offset: usize,
    },

    /// A `BINUNICODE`-family payload was not valid UTF-8.
    #[error("invalid UTF-8 in pickle string at byte offset {offset}: {source}")]
    InvalidUtf8 {
        /// Byte offset of the string.
        offset: usize,
        /// The underlying decoding error.
        source: std::str::Utf8Error,
    },

    /// An opcode was applied to a value of the wrong kind.
    #[error("{opcode} applied to {found} at byte offset {offset}")]
    WrongTarget {
        /// Opcode name.
        opcode: &'static str,
        /// What was found instead.
        found: &'static str,
        /// Byte offset of the opcode.
        offset: usize,
    },

    /// A structural limit was exceeded, which a well-formed checkpoint never
    /// does; see the limits in [`super::vm`].
    #[error("pickle exceeded the {what} limit of {limit}")]
    LimitExceeded {
        /// What was exceeded, e.g. `MARK nesting depth`.
        what: &'static str,
        /// The limit that was hit.
        limit: usize,
    },

    /// The pickle decoded successfully but did not describe what the caller
    /// asked for -- e.g. a `.pt` whose top level is not a state dict, or a
    /// tensor built from an unrecognized rebuild function.
    #[error("unsupported checkpoint structure: {0}")]
    Structure(String),

    /// A tensor referenced a dtype this crate cannot represent.
    #[error("unsupported tensor dtype: {0}")]
    UnsupportedDtype(String),

    /// The referenced storage was missing from the container, or too small
    /// for the tensor's declared shape and offset.
    #[error("storage '{key}' {problem}")]
    Storage {
        /// The storage key the tensor referenced.
        key: String,
        /// What was wrong with it.
        problem: String,
    },

    /// The `.pt` ZIP container could not be read.
    #[error("failed to read checkpoint archive: {0}")]
    Archive(String),

    /// An underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result alias for pickle operations.
pub type Result<T> = std::result::Result<T, PickleError>;

impl From<PickleError> for crate::BridgeError {
    fn from(error: PickleError) -> Self {
        match error {
            PickleError::Io(source) => Self::Io(source),
            other => Self::Conversion(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_error_maps_to_bridge_io_variant() {
        // An I/O failure must not be flattened into a stringly-typed
        // conversion error: callers matching on `BridgeError::Io` (e.g. to
        // distinguish "file missing" from "file corrupt") depend on it.
        let error = PickleError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no such file",
        ));
        assert!(matches!(
            crate::BridgeError::from(error),
            crate::BridgeError::Io(_)
        ));
    }

    #[test]
    fn test_structural_error_maps_to_conversion() {
        let error = PickleError::Structure("top level is a list".into());
        let bridge = crate::BridgeError::from(error);
        assert!(matches!(bridge, crate::BridgeError::Conversion(_)));
        assert!(bridge.to_string().contains("top level is a list"));
    }
}
