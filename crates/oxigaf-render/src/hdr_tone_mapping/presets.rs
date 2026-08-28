//! Ready-made `ToneMappingConfig` presets.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::config::ToneMappingConfig;
use super::operator::ToneMappingOperator;

/// Simple Reinhard tone mapping with neutral gamma.
pub fn preset_reinhard() -> ToneMappingConfig {
    ToneMappingConfig {
        operator: ToneMappingOperator::Reinhard,
        exposure_stops: 0.0,
        gamma: 1.0,
        use_srgb_gamma: false,
        saturation: 1.0,
    }
}

/// ACES filmic tone mapping with proper sRGB gamma encoding.
pub fn preset_aces() -> ToneMappingConfig {
    ToneMappingConfig {
        operator: ToneMappingOperator::AcesFilmic,
        exposure_stops: 0.0,
        gamma: 1.0,
        use_srgb_gamma: true,
        saturation: 1.0,
    }
}

/// Simplified filmic curve with neutral gamma.
pub fn preset_filmic() -> ToneMappingConfig {
    ToneMappingConfig {
        operator: ToneMappingOperator::Filmic,
        exposure_stops: 0.0,
        gamma: 1.0,
        use_srgb_gamma: false,
        saturation: 1.0,
    }
}

/// Hable "Uncharted 2" filmic curve with a slight saturation boost.
///
/// Simulates a photographic look suitable for outdoor 3DGS scenes.
pub fn preset_photography() -> ToneMappingConfig {
    ToneMappingConfig {
        operator: ToneMappingOperator::Hable { exposure: 2.0 },
        exposure_stops: 0.0,
        gamma: 1.0,
        use_srgb_gamma: true,
        saturation: 1.1,
    }
}
