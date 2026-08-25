//! Tone mapping pipeline configuration types.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::error::ToneMappingError;
use super::operator::{ToneMapOperator, ToneMappingOperator};

/// Configuration for the new [`crate::tone_map`] / [`crate::tone_map_inplace`] pipeline.
#[derive(Debug, Clone)]
pub struct ToneMapConfig {
    /// The tone mapping operator.
    pub operator: ToneMapOperator,
    /// Display gamma exponent (default 2.2).
    pub gamma: f32,
    /// Whether to apply gamma correction after tone mapping.
    pub apply_gamma: bool,
    /// Whether to clip output to `[0, 1]`.
    pub clip: bool,
}
impl Default for ToneMapConfig {
    fn default() -> Self {
        Self {
            operator: ToneMapOperator::Aces,
            gamma: 2.2,
            apply_gamma: true,
            clip: true,
        }
    }
}
/// Full tone mapping pipeline configuration.
#[derive(Debug, Clone)]
pub struct ToneMappingConfig {
    /// The tone mapping operator to apply.
    pub operator: ToneMappingOperator,
    /// Pre-tone-mapping exposure in EV stops.
    pub exposure_stops: f32,
    /// Post-tone-mapping gamma correction exponent.
    ///
    /// `1.0` means no correction; `2.2` approximates sRGB.
    /// This is ignored when `use_srgb_gamma` is `true`.
    pub gamma: f32,
    /// If `true`, apply the proper piecewise sRGB curve instead of simple
    /// power-law gamma.
    pub use_srgb_gamma: bool,
    /// Channel saturation multiplier applied after tone mapping.
    ///
    /// `1.0` = neutral; `0.0` = greyscale; `> 1.0` = boosted saturation.
    pub saturation: f32,
}
impl ToneMappingConfig {
    /// Validate configuration parameters.
    ///
    /// Returns [`ToneMappingError::InvalidConfig`] if any parameter is out of
    /// its valid range.
    pub fn validate(&self) -> Result<(), ToneMappingError> {
        if self.gamma <= 0.0 {
            return Err(ToneMappingError::InvalidConfig(format!(
                "gamma must be > 0, got {}",
                self.gamma
            )));
        }
        if self.saturation < 0.0 {
            return Err(ToneMappingError::InvalidConfig(format!(
                "saturation must be >= 0, got {}",
                self.saturation
            )));
        }
        if let ToneMappingOperator::Custom {
            shadow_gamma,
            midtone_scale,
            highlight_rolloff,
        } = &self.operator
        {
            if *shadow_gamma <= 0.0 {
                return Err(ToneMappingError::InvalidConfig(format!(
                    "Custom operator shadow_gamma must be > 0, got {shadow_gamma}"
                )));
            }
            if *midtone_scale <= 0.0 {
                return Err(ToneMappingError::InvalidConfig(format!(
                    "Custom operator midtone_scale must be > 0, got {midtone_scale}"
                )));
            }
            if *highlight_rolloff <= 0.0 {
                return Err(ToneMappingError::InvalidConfig(format!(
                    "Custom operator highlight_rolloff must be > 0, got {highlight_rolloff}"
                )));
            }
        }
        Ok(())
    }
}
impl Default for ToneMappingConfig {
    fn default() -> Self {
        Self {
            operator: ToneMappingOperator::AcesFilmic,
            exposure_stops: 0.0,
            gamma: 1.0,
            use_srgb_gamma: false,
            saturation: 1.0,
        }
    }
}
/// Format a [`ToneMapConfig`] as a human-readable string.
pub fn format_tone_config(config: &ToneMapConfig) -> String {
    let op = match &config.operator {
        ToneMapOperator::Reinhard => "reinhard".to_string(),
        ToneMapOperator::ReinhardExtended { max_luminance } => {
            format!("reinhard_extended(max_lum={max_luminance})")
        }
        ToneMapOperator::Filmic => "filmic".to_string(),
        ToneMapOperator::Aces => "aces".to_string(),
        ToneMapOperator::Lottes(params) => format!(
            "lottes(contrast={},shoulder={},hdr_max={},mid_in={},mid_out={})",
            params.contrast, params.shoulder, params.hdr_max, params.mid_in, params.mid_out
        ),
        ToneMapOperator::Exposure { stops } => format!("exposure(stops={stops})"),
        ToneMapOperator::Linear { min, max } => format!("linear(min={min},max={max})"),
    };
    format!(
        "ToneMapConfig {{ op={op}, gamma={}, apply_gamma={}, clip={} }}",
        config.gamma, config.apply_gamma, config.clip
    )
}
