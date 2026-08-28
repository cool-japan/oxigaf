//! Head tracking over video sequences.
//!
//! Maintains pose history, applies temporal filters, detects anomalies,
//! and provides trajectory analysis for 6-DOF head pose sequences.

pub mod functions;
pub mod one_euro;
pub mod tracker;
pub mod types;

pub use functions::*;
pub use tracker::*;
pub use types::*;

#[cfg(test)]
mod tests;
