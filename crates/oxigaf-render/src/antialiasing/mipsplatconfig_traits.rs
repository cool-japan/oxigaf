//! # `MipSplatConfig` - Trait Implementations
//!
//! This module contains trait implementations for `MipSplatConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MipSplatConfig;

impl Default for MipSplatConfig {
    fn default() -> Self {
        Self {
            min_2d_radius_px: 0.3,
            max_distance_scale: 1.0,
            opacity_ramp_min_px: 0.5,
            opacity_ramp_max_px: 2.0,
            use_distance_lod: false,
            reference_distance: 1.0,
        }
    }
}
