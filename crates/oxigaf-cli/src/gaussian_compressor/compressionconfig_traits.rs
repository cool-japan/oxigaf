//! # `CompressionConfig` - Trait Implementations
//!
//! This module contains trait implementations for `CompressionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CompressionConfig, KMeansConfig, QuantizationPrecision, ScenePruningConfig};

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            position_precision: QuantizationPrecision::Half,
            rotation_precision: QuantizationPrecision::Half,
            scale_precision: QuantizationPrecision::Half,
            opacity_precision: QuantizationPrecision::Half,
            sh_dc_precision: QuantizationPrecision::Half,
            sh_rest_precision: QuantizationPrecision::Half,
            pruning: ScenePruningConfig::default(),
            use_position_clustering: false,
            kmeans: KMeansConfig::default(),
        }
    }
}
