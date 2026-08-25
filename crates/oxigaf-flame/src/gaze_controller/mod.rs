//! # Eye Gaze Estimation and Control
//!
//! Provides a comprehensive gaze control system for the `OxiGAF` FLAME head model,
//! including Listing's law quaternion computation, I-VT/I-DT saccade/fixation
//! detection, natural blink synthesis, vergence estimation, and statistics.
//!
//! ## Overview
//!
//! - [`GazeDirection`] — azimuth / elevation / vergence representation
//! - [`GazeFrame`]     — per-frame binocular gaze + blink data
//! - [`GazeController`] — stateful ring-buffer gaze manager
//! - [`gz_listing_rotation`] — Listing's law quaternion computation
//! - [`gz_detect_saccades`] / [`gz_detect_fixations`] — I-VT classifier
//! - [`gz_synthesize_blinks`] — natural blink generation (xorshift64 + exponential ISI)

pub mod functions;
pub mod prng;
pub mod types;

pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
