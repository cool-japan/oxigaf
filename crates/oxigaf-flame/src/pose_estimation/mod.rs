//! Head pose estimation from 2D facial landmark observations.
//!
//! Recovers FLAME head pose parameters (rotation and translation) from 2D
//! landmark observations using a weak-perspective (scaled orthographic) `PnP`
//! solve with optional RANSAC robustification and quaternion-based temporal
//! smoothing.  The solver recovers the two scaled rotation rows by linear least
//! squares, projects them onto `SO(3)` and derives the translation from the
//! centroid correspondence — rotation **and** translation are estimated, not
//! assumed.

pub mod functions;
pub mod math;
pub mod types;

pub use functions::*;
pub use types::*;

#[cfg(test)]
#[allow(clippy::doc_markdown)]
mod tests;
