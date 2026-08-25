//! Temporal Anti-Aliasing (TAA) with Halton jitter, variance clipping, and adaptive blending.
//!
//! TAA reduces aliasing by accumulating and blending multiple frames over time. Each frame
//! uses slightly different subpixel sample offsets (jitter from a Halton sequence), and the
//! history buffer is blended with the current frame. This produces high-quality anti-aliasing
//! without the memory overhead of MSAA.
//!
//! ## Key Features
//!
//! - **Halton jitter**: Quasi-random subpixel offsets with better distribution than uniform random.
//! - **Variance clipping**: Reduces ghosting by clamping history to the local color neighborhood.
//! - **Adaptive blending**: Dynamically adjusts blend factor based on per-pixel motion magnitude.
//! - **Unsharp mask sharpening**: Post-accumulation sharpening to counteract blur from blending.
//! - **Stateful accumulator**: [`TaaAccumulator`] manages jitter sequencing and history state.
//!
//! ## Relationship to [`crate::temporal`]
//!
//! Two temporal-accumulation modules coexist in this crate and are *not*
//! interchangeable:
//!
//! | Module | History reprojection | Ghosting control | Extras |
//! |---|---|---|---|
//! | [`crate::temporal_aa`] (this one) | none — history is aligned by construction | variance clipping (local mean ± σ) | Halton jitter, unsharp sharpening, RGB only |
//! | [`crate::temporal`] | motion-vector warping with bilinear resampling | 3×3 neighbourhood min/max clamp + disocclusion blend | arbitrary channel count |
//!
//! Pick this module for a static or jitter-only camera; pick
//! [`crate::temporal`] when you have a motion-vector field. Note that both
//! modules define their own `TaaConfig` and `TaaError` with different fields
//! and variants — the crate root re-exports [`crate::temporal`]'s pair, so
//! this module's must be reached through its full path.
//!
//! # Module layout
//!
//! The implementation is split by concern; every public item is re-exported
//! here, so `oxigaf_render::temporal_aa::<item>` paths are unchanged:
//!
//! - `config` — [`TaaError`] and [`TaaConfig`]
//! - `jitter` — Halton sequence and per-frame subpixel offsets
//! - `history` — the [`TaaHistory`] accumulation buffer
//! - `clipping` — local mean/σ statistics and variance clipping
//! - `accumulate` — sharpening, [`accumulate_taa`], [`TaaAccumulator`]
//! - `stats` — [`TaaStats`] and [`compute_taa_stats`]

mod accumulate;
mod clipping;
mod config;
mod history;
mod jitter;
mod stats;

pub use accumulate::{accumulate_taa, sharpen_image, TaaAccumulator};
pub use clipping::{clip_to_variance, local_color_stats};
pub use config::{TaaConfig, TaaError};
pub use history::TaaHistory;
pub use jitter::{halton, jitter_offset};
pub use stats::{compute_taa_stats, TaaStats};
