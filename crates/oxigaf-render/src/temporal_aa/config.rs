//! TAA error type and pipeline configuration.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by temporal anti-aliasing operations.
#[derive(Debug, Error, PartialEq)]
pub enum TaaError {
    /// Invalid configuration parameter.
    #[error("Invalid TAA configuration: {0}")]
    InvalidConfig(String),

    /// Pixel count does not match declared dimensions.
    #[error("Invalid image: pixel count mismatch — {0}")]
    InvalidImage(String),

    /// History buffer has no frames yet.
    #[error("Empty TAA history — accumulate at least one frame first")]
    EmptyHistory,

    /// Dimension mismatch between two images.
    #[error("Dimension mismatch: expected {expected} pixels, got {got}")]
    DimensionMismatch {
        /// Expected number of pixels.
        expected: usize,
        /// Actual number of pixels.
        got: usize,
    },
}

// ---------------------------------------------------------------------------
// TAA configuration
// ---------------------------------------------------------------------------

/// Configuration for the TAA accumulation pipeline.
#[derive(Debug, Clone)]
pub struct TaaConfig {
    /// Blend factor: weight of the current frame vs. accumulated history.
    ///
    /// - `0.0` = pure history (no current frame contribution).
    /// - `1.0` = pure current frame (no temporal accumulation).
    ///
    /// Typical value: `0.1` (10% current, 90% history).
    pub blend_factor: f32,

    /// Number of frames in the Halton jitter sequence before wrapping.
    ///
    /// A power-of-two value such as `8` or `16` is common.
    pub jitter_sequence_length: usize,

    /// When `true`, clamp history color to the local variance neighborhood of the current
    /// frame before blending. Reduces ghosting at the cost of some sharpness.
    pub variance_clipping: bool,

    /// Radius in pixels of the local neighborhood used for variance clipping.
    ///
    /// A radius of `1` uses a `3×3` pixel window.
    pub clip_window_radius: usize,

    /// Strength of unsharp mask sharpening applied after accumulation.
    ///
    /// - `0.0` = no sharpening.
    /// - `1.0` = strong sharpening.
    pub sharpen_strength: f32,

    /// When `true`, the blend factor adapts per-pixel based on the luminance difference
    /// between the current frame and history. High motion → more current frame weight.
    pub adaptive_blend: bool,

    /// Minimum blend factor when adaptive blending is active. Must be ≤ `blend_factor`.
    pub adaptive_blend_min: f32,
}

impl Default for TaaConfig {
    fn default() -> Self {
        Self {
            blend_factor: 0.1,
            jitter_sequence_length: 8,
            variance_clipping: true,
            clip_window_radius: 1,
            sharpen_strength: 0.2,
            adaptive_blend: false,
            adaptive_blend_min: 0.05,
        }
    }
}

impl TaaConfig {
    /// Validate all configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns [`TaaError::InvalidConfig`] if any parameter is out of range.
    pub fn validate(&self) -> Result<(), TaaError> {
        if self.blend_factor <= 0.0 || self.blend_factor > 1.0 {
            return Err(TaaError::InvalidConfig(format!(
                "blend_factor must be in (0, 1], got {}",
                self.blend_factor
            )));
        }
        if self.jitter_sequence_length < 1 {
            return Err(TaaError::InvalidConfig(
                "jitter_sequence_length must be >= 1".to_string(),
            ));
        }
        if self.clip_window_radius < 1 {
            return Err(TaaError::InvalidConfig(
                "clip_window_radius must be >= 1".to_string(),
            ));
        }
        if self.sharpen_strength < 0.0 {
            return Err(TaaError::InvalidConfig(format!(
                "sharpen_strength must be >= 0, got {}",
                self.sharpen_strength
            )));
        }
        if self.adaptive_blend_min < 0.0 || self.adaptive_blend_min > self.blend_factor {
            return Err(TaaError::InvalidConfig(format!(
                "adaptive_blend_min must be in [0, blend_factor={}], got {}",
                self.blend_factor, self.adaptive_blend_min
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── TaaConfig ──────────────────────────────────────────────────────────────

    #[test]
    fn test_taaconfig_default_validates() {
        assert!(TaaConfig::default().validate().is_ok());
    }

    #[test]
    fn test_taaconfig_invalid_blend_factor_zero() {
        let cfg = TaaConfig {
            blend_factor: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_invalid_blend_factor_over_one() {
        let cfg = TaaConfig {
            blend_factor: 1.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_blend_factor_exactly_one_is_valid() {
        let cfg = TaaConfig {
            blend_factor: 1.0,
            adaptive_blend_min: 0.05_f32.min(1.0),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_taaconfig_invalid_jitter_length_zero() {
        let cfg = TaaConfig {
            jitter_sequence_length: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_invalid_clip_window_zero() {
        let cfg = TaaConfig {
            clip_window_radius: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_invalid_sharpen_negative() {
        let cfg = TaaConfig {
            sharpen_strength: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_adaptive_blend_min_exceeds_blend_factor() {
        let mut cfg = TaaConfig::default();
        cfg.adaptive_blend_min = cfg.blend_factor + 0.1;
        assert!(cfg.validate().is_err());
    }
}
