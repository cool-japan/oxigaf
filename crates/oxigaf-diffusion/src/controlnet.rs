//! ControlNet-style conditioning for the diffusion pipeline.
//!
//! Injects conditioning signals (edge maps, depth maps, normal maps) into the
//! U-Net at multiple resolution levels, following the ControlNet paper.

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// ControlSignalType
// ---------------------------------------------------------------------------

/// Type of conditioning signal fed to ControlNet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlSignalType {
    /// Canny edge detection output (binary edges, single channel).
    CannyEdge,
    /// Depth map normalized to [0, 1], single channel.
    DepthMap,
    /// Surface normal map, RGB-encoded (3 channels).
    NormalMap,
    /// Soft edge (HED/PIDI-style), single channel.
    SoftEdge,
    /// OpenPose keypoints rendered as heatmaps (3 channels).
    Pose,
}

impl ControlSignalType {
    /// Returns the number of image channels expected for this signal type.
    pub fn expected_channels(&self) -> usize {
        match self {
            ControlSignalType::CannyEdge
            | ControlSignalType::DepthMap
            | ControlSignalType::SoftEdge => 1,
            ControlSignalType::NormalMap | ControlSignalType::Pose => 3,
        }
    }

    /// Returns a human-readable display name for this signal type.
    pub fn display_name(&self) -> &'static str {
        match self {
            ControlSignalType::CannyEdge => "Canny Edge",
            ControlSignalType::DepthMap => "Depth Map",
            ControlSignalType::NormalMap => "Normal Map",
            ControlSignalType::SoftEdge => "Soft Edge",
            ControlSignalType::Pose => "Pose",
        }
    }
}

// ---------------------------------------------------------------------------
// ControlNetCondition
// ---------------------------------------------------------------------------

/// A single control signal image for conditioning.
#[derive(Debug, Clone)]
pub struct ControlNetCondition {
    /// Type of the conditioning signal.
    pub signal_type: ControlSignalType,
    /// Flattened pixel data: `[channels * height * width]` f32, values in [0, 1].
    pub data: Vec<f32>,
    /// Number of channels (derived from `signal_type`).
    pub channels: usize,
    /// Height of the control image.
    pub height: usize,
    /// Width of the control image.
    pub width: usize,
    /// Scale factor applied during feature injection (default 1.0, higher = stronger).
    pub conditioning_scale: f32,
}

impl ControlNetCondition {
    /// Creates a new `ControlNetCondition`, inferring channel count from the signal type.
    ///
    /// # Errors
    /// - `DiffusionError::ShapeMismatch` if `data.len() != channels * height * width`.
    /// - `DiffusionError::InvalidConfig` if `conditioning_scale <= 0.0`.
    pub fn new(
        signal_type: ControlSignalType,
        data: Vec<f32>,
        height: usize,
        width: usize,
        conditioning_scale: f32,
    ) -> Result<Self, DiffusionError> {
        if conditioning_scale <= 0.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "conditioning_scale must be > 0.0, got {conditioning_scale}"
            )));
        }

        let channels = signal_type.expected_channels();
        let expected_len = channels * height * width;
        if data.len() != expected_len {
            return Err(DiffusionError::ShapeMismatch {
                op: "ControlNetCondition::new".to_string(),
                expected: vec![channels, height, width],
                got: vec![data.len()],
            });
        }

        Ok(Self {
            signal_type,
            data,
            channels,
            height,
            width,
            conditioning_scale,
        })
    }

    /// Convenience constructor for a Canny edge map with default scale 1.0.
    pub fn edge_map(data: Vec<f32>, height: usize, width: usize) -> Result<Self, DiffusionError> {
        Self::new(ControlSignalType::CannyEdge, data, height, width, 1.0)
    }

    /// Convenience constructor for a depth map with default scale 1.0.
    pub fn depth_map(data: Vec<f32>, height: usize, width: usize) -> Result<Self, DiffusionError> {
        Self::new(ControlSignalType::DepthMap, data, height, width, 1.0)
    }

    /// Convenience constructor for a normal map with default scale 1.0.
    pub fn normal_map(data: Vec<f32>, height: usize, width: usize) -> Result<Self, DiffusionError> {
        Self::new(ControlSignalType::NormalMap, data, height, width, 1.0)
    }

    /// Returns the pixel value at channel `c`, row `y`, column `x`.
    ///
    /// Returns `0.0` for any out-of-bounds index.
    pub fn pixel_at(&self, c: usize, y: usize, x: usize) -> f32 {
        if c >= self.channels || y >= self.height || x >= self.width {
            return 0.0;
        }
        let idx = c * self.height * self.width + y * self.width + x;
        // idx is always in bounds by the guard above, but use get for safety
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Downsamples the condition image to `(target_h, target_w)` using nearest-neighbor.
    ///
    /// Returns a clone of `self` if the size already matches.
    pub fn downsample_to(&self, target_h: usize, target_w: usize) -> Self {
        if self.height == target_h && self.width == target_w {
            return Self {
                signal_type: self.signal_type,
                data: self.data.clone(),
                channels: self.channels,
                height: self.height,
                width: self.width,
                conditioning_scale: self.conditioning_scale,
            };
        }

        // Avoid division by zero for degenerate sizes
        let src_h = self.height.max(1);
        let src_w = self.width.max(1);
        let tgt_h = target_h.max(1);
        let tgt_w = target_w.max(1);

        let mut out = vec![0.0f32; self.channels * tgt_h * tgt_w];

        for c in 0..self.channels {
            for ty in 0..tgt_h {
                // Map target pixel back to source: nearest-neighbor
                let sy = (ty * src_h / tgt_h).min(src_h - 1);
                for tx in 0..tgt_w {
                    let sx = (tx * src_w / tgt_w).min(src_w - 1);
                    let src_idx = c * src_h * src_w + sy * src_w + sx;
                    let dst_idx = c * tgt_h * tgt_w + ty * tgt_w + tx;
                    out[dst_idx] = self.data.get(src_idx).copied().unwrap_or(0.0);
                }
            }
        }

        Self {
            signal_type: self.signal_type,
            data: out,
            channels: self.channels,
            height: tgt_h,
            width: tgt_w,
            conditioning_scale: self.conditioning_scale,
        }
    }

    /// Normalizes pixel data in-place to [0, 1] by dividing by the maximum value.
    ///
    /// Does nothing if all values are zero.
    pub fn normalize(&mut self) {
        let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if max_val > 0.0 && max_val.is_finite() {
            for v in &mut self.data {
                *v /= max_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ControlNetConfig
// ---------------------------------------------------------------------------

/// Configuration for ControlNet conditioning.
#[derive(Debug, Clone)]
pub struct ControlNetConfig {
    /// Whether ControlNet conditioning is active.
    pub enabled: bool,
    /// Maximum number of simultaneous conditions.
    pub max_conditions: usize,
    /// U-Net layer indices at which features will be injected.
    pub injection_layers: Vec<usize>,
    /// Default conditioning scale applied when no per-condition scale is set.
    pub default_scale: f32,
    /// If `true`, inject only at later (coarser) layers; if `false` inject at all listed layers.
    pub late_injection: bool,
}

impl ControlNetConfig {
    /// Returns the default (enabled) configuration.
    pub fn default_config() -> Self {
        Self {
            enabled: true,
            max_conditions: 4,
            injection_layers: vec![0, 1, 2, 3],
            default_scale: 1.0,
            late_injection: false,
        }
    }

    /// Returns a disabled configuration (no conditioning applied).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            max_conditions: 4,
            injection_layers: vec![0, 1, 2, 3],
            default_scale: 1.0,
            late_injection: false,
        }
    }

    /// Validates configuration parameters.
    ///
    /// # Errors
    /// - `DiffusionError::InvalidConfig` if `max_conditions == 0` or `default_scale <= 0.0`.
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if self.max_conditions == 0 {
            return Err(DiffusionError::InvalidConfig(
                "max_conditions must be > 0".to_string(),
            ));
        }
        if self.default_scale <= 0.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "default_scale must be > 0.0, got {}",
                self.default_scale
            )));
        }
        Ok(())
    }
}

impl Default for ControlNetConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

// ---------------------------------------------------------------------------
// ControlFeature
// ---------------------------------------------------------------------------

/// Control features injected at one U-Net layer.
#[derive(Debug, Clone)]
pub struct ControlFeature {
    /// Index of the U-Net layer this feature belongs to.
    pub layer_idx: usize,
    /// Weighted feature map: `[batch * channels * height * width]` f32.
    pub features: Vec<f32>,
    /// Batch size (number of images).
    pub batch_size: usize,
    /// Number of feature channels.
    pub channels: usize,
    /// Spatial height of the feature map.
    pub height: usize,
    /// Spatial width of the feature map.
    pub width: usize,
}

// ---------------------------------------------------------------------------
// ControlNetProcessor
// ---------------------------------------------------------------------------

/// Manages multiple ControlNet conditions and orchestrates feature injection.
#[derive(Debug)]
pub struct ControlNetProcessor {
    /// Active configuration.
    pub config: ControlNetConfig,
    /// Registered conditioning signals.
    conditions: Vec<ControlNetCondition>,
}

impl ControlNetProcessor {
    /// Creates a new `ControlNetProcessor` with the given configuration.
    pub fn new(config: ControlNetConfig) -> Self {
        Self {
            config,
            conditions: Vec::new(),
        }
    }

    /// Adds a conditioning signal to the processor.
    ///
    /// # Errors
    /// - `DiffusionError::InvalidConfig` if ControlNet is disabled.
    /// - `DiffusionError::InvalidConfig` if `max_conditions` would be exceeded.
    pub fn add_condition(&mut self, cond: ControlNetCondition) -> Result<(), DiffusionError> {
        if !self.config.enabled {
            return Err(DiffusionError::InvalidConfig(
                "ControlNet is disabled; cannot add conditions".to_string(),
            ));
        }
        if self.conditions.len() >= self.config.max_conditions {
            return Err(DiffusionError::InvalidConfig(format!(
                "Cannot add more than {} conditions",
                self.config.max_conditions
            )));
        }
        self.conditions.push(cond);
        Ok(())
    }

    /// Returns a slice of all registered conditions.
    pub fn conditions(&self) -> &[ControlNetCondition] {
        &self.conditions
    }

    /// Injects conditioning features into a mutable feature buffer for a given U-Net layer.
    ///
    /// If `layer_idx` is not in `config.injection_layers`, the buffer is left unchanged.
    /// Each condition is downsampled to `(feature_h, feature_w)` and its values are added
    /// element-wise, broadcast across all channels. After injection, features are clamped
    /// to `[-10.0, 10.0]` to prevent explosion.
    ///
    /// `features` is expected to have length `channels * feature_h * feature_w` where
    /// `channels` is implied by the total length divided by `feature_h * feature_w`.
    pub fn apply_to_features(
        &self,
        features: &mut [f32],
        layer_idx: usize,
        feature_h: usize,
        feature_w: usize,
    ) {
        if !self.config.enabled {
            return;
        }

        // Only inject at specified layers
        if !self.config.injection_layers.contains(&layer_idx) {
            return;
        }

        let spatial = feature_h * feature_w;
        if spatial == 0 || features.is_empty() {
            return;
        }

        // Number of channels implied by feature buffer size
        let feature_channels = features.len().checked_div(spatial).unwrap_or(0);
        if feature_channels == 0 {
            return;
        }

        // Accumulate contributions from all conditions
        for cond in &self.conditions {
            let downsampled = cond.downsample_to(feature_h, feature_w);
            let scale = downsampled.conditioning_scale;

            // For each condition channel, add scaled values to all feature channels
            for fy in 0..feature_h {
                for fx in 0..feature_w {
                    // Sum across condition channels to produce a scalar per spatial position
                    let mut cond_val = 0.0f32;
                    for cc in 0..downsampled.channels {
                        cond_val += downsampled.pixel_at(cc, fy, fx);
                    }
                    // Normalize by condition channel count to keep magnitude stable
                    if downsampled.channels > 0 {
                        cond_val /= downsampled.channels as f32;
                    }
                    cond_val *= scale;

                    // Broadcast this scalar to all feature channels at this spatial position
                    for fc in 0..feature_channels {
                        let idx = fc * spatial + fy * feature_w + fx;
                        if let Some(v) = features.get_mut(idx) {
                            *v += cond_val;
                        }
                    }
                }
            }
        }

        // Clamp to prevent gradient explosion
        for v in features.iter_mut() {
            *v = v.clamp(-10.0, 10.0);
        }
    }

    /// Generates `ControlFeature` objects for every injection layer by summing downsampled conditions.
    ///
    /// Output features have `channels = 1` (scalar aggregation).
    pub fn process_conditions(&self, image_h: usize, image_w: usize) -> Vec<ControlFeature> {
        if !self.config.enabled || self.conditions.is_empty() {
            return Vec::new();
        }

        self.config
            .injection_layers
            .iter()
            .map(|&layer_idx| {
                // Halve resolution at each deeper layer (approximate U-Net scale)
                let scale_factor = 1usize << layer_idx;
                let feat_h = (image_h / scale_factor).max(1);
                let feat_w = (image_w / scale_factor).max(1);
                let spatial = feat_h * feat_w;

                // Sum all condition contributions at this resolution
                let mut data = vec![0.0f32; spatial];

                for cond in &self.conditions {
                    let downsampled = cond.downsample_to(feat_h, feat_w);
                    let scale = downsampled.conditioning_scale;

                    for y in 0..feat_h {
                        for x in 0..feat_w {
                            let mut cond_val = 0.0f32;
                            for cc in 0..downsampled.channels {
                                cond_val += downsampled.pixel_at(cc, y, x);
                            }
                            if downsampled.channels > 0 {
                                cond_val /= downsampled.channels as f32;
                            }
                            let idx = y * feat_w + x;
                            data[idx] += cond_val * scale;
                        }
                    }
                }

                // Clamp to prevent explosion
                for v in &mut data {
                    *v = v.clamp(-10.0, 10.0);
                }

                ControlFeature {
                    layer_idx,
                    features: data,
                    batch_size: 1,
                    channels: 1,
                    height: feat_h,
                    width: feat_w,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Applies a 3×3 Sobel edge-detection filter to a single-channel image.
///
/// Border pixels are set to 0. Output is normalized to [0, 1].
///
/// `image` must have length `height * width`. If the input is empty, an empty
/// vector is returned.
pub fn apply_edge_enhancement(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    if image.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }

    // Sobel kernels
    // Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    // Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
    let gx: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let gy: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    let mut out = vec![0.0f32; height * width];

    for y in 1..(height.saturating_sub(1)) {
        for x in 1..(width.saturating_sub(1)) {
            let mut vx = 0.0f32;
            let mut vy = 0.0f32;

            for ky in 0..3usize {
                for kx in 0..3usize {
                    let sy = y + ky - 1;
                    let sx = x + kx - 1;
                    let idx = sy * width + sx;
                    let pixel = image.get(idx).copied().unwrap_or(0.0);
                    vx += gx[ky][kx] * pixel;
                    vy += gy[ky][kx] * pixel;
                }
            }

            let magnitude = (vx * vx + vy * vy).sqrt();
            out[y * width + x] = magnitude;
        }
    }

    // Normalize to [0, 1]
    let max_val = out.iter().cloned().fold(0.0f32, f32::max);
    if max_val > 0.0 {
        for v in &mut out {
            *v /= max_val;
        }
    }

    out
}

/// Converts a batch of normal maps (3-channel, \[0,1\] values) into `ControlNetCondition` objects.
///
/// Invalid maps (wrong data length) are silently skipped.
pub fn condition_images_from_multi_view(
    normal_maps: &[Vec<f32>],
    height: usize,
    width: usize,
) -> Vec<ControlNetCondition> {
    normal_maps
        .iter()
        .filter_map(|data| ControlNetCondition::normal_map(data.clone(), height, width).ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ControlSignalType
    // -----------------------------------------------------------------------

    #[test]
    fn test_signal_type_channels() {
        assert_eq!(ControlSignalType::CannyEdge.expected_channels(), 1);
        assert_eq!(ControlSignalType::DepthMap.expected_channels(), 1);
        assert_eq!(ControlSignalType::SoftEdge.expected_channels(), 1);
        assert_eq!(ControlSignalType::NormalMap.expected_channels(), 3);
        assert_eq!(ControlSignalType::Pose.expected_channels(), 3);
    }

    #[test]
    fn test_signal_type_display_name() {
        assert_eq!(ControlSignalType::CannyEdge.display_name(), "Canny Edge");
        assert_eq!(ControlSignalType::DepthMap.display_name(), "Depth Map");
        assert_eq!(ControlSignalType::NormalMap.display_name(), "Normal Map");
        assert_eq!(ControlSignalType::SoftEdge.display_name(), "Soft Edge");
        assert_eq!(ControlSignalType::Pose.display_name(), "Pose");
    }

    // -----------------------------------------------------------------------
    // ControlNetCondition construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_condition_new_valid() {
        let data = vec![0.5f32; 4 * 4]; // 1 channel, 4×4
        let cond = ControlNetCondition::new(ControlSignalType::DepthMap, data.clone(), 4, 4, 0.8);
        assert!(cond.is_ok());
        let c = cond.unwrap();
        assert_eq!(c.channels, 1);
        assert_eq!(c.height, 4);
        assert_eq!(c.width, 4);
        assert_eq!(c.conditioning_scale, 0.8);
    }

    #[test]
    fn test_condition_new_invalid_data_size() {
        // NormalMap expects 3 channels, but we supply 1 channel worth of data
        let data = vec![0.5f32; 4 * 4];
        let result = ControlNetCondition::new(ControlSignalType::NormalMap, data, 4, 4, 1.0);
        assert!(result.is_err());
        match result {
            Err(DiffusionError::ShapeMismatch { op, .. }) => {
                assert!(op.contains("ControlNetCondition::new"));
            }
            other => panic!("Expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_condition_new_invalid_scale() {
        let data = vec![0.5f32; 4 * 4];
        let result = ControlNetCondition::new(ControlSignalType::CannyEdge, data, 4, 4, -1.0);
        assert!(result.is_err());
        match result {
            Err(DiffusionError::InvalidConfig(_)) => {}
            other => panic!("Expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_condition_edge_map_convenience() {
        let data = vec![0.0f32; 8 * 8];
        let cond = ControlNetCondition::edge_map(data, 8, 8);
        assert!(cond.is_ok());
        let c = cond.unwrap();
        assert_eq!(c.signal_type, ControlSignalType::CannyEdge);
        assert_eq!(c.channels, 1);
        assert_eq!(c.conditioning_scale, 1.0);
    }

    #[test]
    fn test_condition_depth_map_convenience() {
        let data = vec![0.3f32; 6 * 6];
        let cond = ControlNetCondition::depth_map(data, 6, 6);
        assert!(cond.is_ok());
        let c = cond.unwrap();
        assert_eq!(c.signal_type, ControlSignalType::DepthMap);
    }

    #[test]
    fn test_condition_normal_map_convenience() {
        let data = vec![0.7f32; 3 * 5 * 5]; // 3 channels, 5×5
        let cond = ControlNetCondition::normal_map(data, 5, 5);
        assert!(cond.is_ok());
        let c = cond.unwrap();
        assert_eq!(c.signal_type, ControlSignalType::NormalMap);
        assert_eq!(c.channels, 3);
    }

    // -----------------------------------------------------------------------
    // pixel_at
    // -----------------------------------------------------------------------

    #[test]
    fn test_condition_pixel_at_valid() {
        // 1-channel 2×3 image, row-major values 0..6
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let cond = ControlNetCondition::edge_map(data, 2, 3).unwrap();
        // pixel_at(c=0, y=1, x=2) => index 1*3+2 = 5
        assert_eq!(cond.pixel_at(0, 1, 2), 5.0);
        assert_eq!(cond.pixel_at(0, 0, 0), 0.0);
    }

    #[test]
    fn test_condition_pixel_at_oob_returns_zero() {
        let data = vec![1.0f32; 4];
        let cond = ControlNetCondition::edge_map(data, 2, 2).unwrap();
        assert_eq!(cond.pixel_at(1, 0, 0), 0.0); // channel OOB
        assert_eq!(cond.pixel_at(0, 5, 0), 0.0); // y OOB
        assert_eq!(cond.pixel_at(0, 0, 5), 0.0); // x OOB
    }

    // -----------------------------------------------------------------------
    // downsample_to
    // -----------------------------------------------------------------------

    #[test]
    fn test_condition_downsample_to() {
        // 4×4 single-channel image, all ones
        let data = vec![1.0f32; 16];
        let cond = ControlNetCondition::depth_map(data, 4, 4).unwrap();
        let ds = cond.downsample_to(2, 2);
        assert_eq!(ds.height, 2);
        assert_eq!(ds.width, 2);
        assert_eq!(ds.data.len(), 4);
        // Nearest-neighbor from uniform source should still be all-ones
        for v in &ds.data {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0 got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // normalize
    // -----------------------------------------------------------------------

    #[test]
    fn test_condition_normalize() {
        let data = vec![0.0f32, 2.0, 4.0, 8.0];
        let mut cond = ControlNetCondition::depth_map(data, 2, 2).unwrap();
        cond.normalize();
        // max was 8.0 → all values divided by 8
        assert!((cond.data[3] - 1.0).abs() < 1e-6);
        assert!((cond.data[2] - 0.5).abs() < 1e-6);
        assert!(cond.data[0].abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // ControlNetConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = ControlNetConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.max_conditions, 4);
        assert_eq!(cfg.injection_layers, vec![0, 1, 2, 3]);
        assert_eq!(cfg.default_scale, 1.0);
        assert!(!cfg.late_injection);
    }

    #[test]
    fn test_config_disabled() {
        let cfg = ControlNetConfig::disabled();
        assert!(!cfg.enabled);
    }

    #[test]
    fn test_config_validate() {
        let mut cfg = ControlNetConfig::default();
        assert!(cfg.validate().is_ok());

        cfg.max_conditions = 0;
        assert!(cfg.validate().is_err());

        cfg.max_conditions = 4;
        cfg.default_scale = -1.0;
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // ControlNetProcessor
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_add_condition() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        let data = vec![0.5f32; 8 * 8];
        let cond = ControlNetCondition::depth_map(data, 8, 8).unwrap();
        assert!(proc.add_condition(cond).is_ok());
        assert_eq!(proc.conditions().len(), 1);
    }

    #[test]
    fn test_processor_add_too_many_conditions() {
        let cfg = ControlNetConfig {
            max_conditions: 2,
            ..ControlNetConfig::default()
        };
        let mut proc = ControlNetProcessor::new(cfg);

        for _ in 0..2 {
            let data = vec![0.1f32; 4 * 4];
            let cond = ControlNetCondition::depth_map(data, 4, 4).unwrap();
            assert!(proc.add_condition(cond).is_ok());
        }

        // Third one should fail
        let data = vec![0.1f32; 4 * 4];
        let cond = ControlNetCondition::depth_map(data, 4, 4).unwrap();
        let result = proc.add_condition(cond);
        assert!(result.is_err());
        match result {
            Err(DiffusionError::InvalidConfig(_)) => {}
            other => panic!("Expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_processor_add_to_disabled_error() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::disabled());
        let data = vec![0.5f32; 4 * 4];
        let cond = ControlNetCondition::depth_map(data, 4, 4).unwrap();
        let result = proc.add_condition(cond);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // apply_to_features
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_to_features_at_injection_layer() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        // Add a depth map that is all 1.0 at 4×4
        let data = vec![1.0f32; 16];
        let cond = ControlNetCondition::depth_map(data, 4, 4).unwrap();
        proc.add_condition(cond).unwrap();

        // feature buffer: 1 channel, 4×4 = 16 elements, all zeros
        let mut features = vec![0.0f32; 16];
        proc.apply_to_features(&mut features, 0, 4, 4);

        // After injection with scale=1.0, all positions should be 1.0
        for v in &features {
            assert!((*v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_apply_to_features_not_injection_layer() {
        let cfg = ControlNetConfig {
            injection_layers: vec![0, 1], // layer 5 is NOT in list
            ..ControlNetConfig::default()
        };
        let mut proc = ControlNetProcessor::new(cfg);
        let data = vec![1.0f32; 16];
        let cond = ControlNetCondition::depth_map(data, 4, 4).unwrap();
        proc.add_condition(cond).unwrap();

        let mut features = vec![0.0f32; 16];
        proc.apply_to_features(&mut features, 5, 4, 4); // layer 5 not in injection_layers

        // Features should remain unchanged (all zeros)
        for v in &features {
            assert!(v.abs() < 1e-10, "expected 0.0, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // apply_edge_enhancement
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_edge_enhancement_shape() {
        // Uniform image → no edges → output all zeros (then normalization is no-op)
        let image = vec![0.5f32; 8 * 8];
        let edges = apply_edge_enhancement(&image, 8, 8);
        assert_eq!(edges.len(), 64);

        // Image with a clear vertical edge: left half 0.0, right half 1.0
        let mut stepped = vec![0.0f32; 8 * 8];
        for y in 0..8usize {
            for x in 4..8usize {
                stepped[y * 8 + x] = 1.0;
            }
        }
        let edges2 = apply_edge_enhancement(&stepped, 8, 8);
        assert_eq!(edges2.len(), 64);
        // Check values are in [0, 1]
        for v in &edges2 {
            assert!(*v >= 0.0 && *v <= 1.0, "out of range: {v}");
        }
        // The column at x=4 (the edge column) should have the highest response
        let max_col4: f32 = (1..7)
            .map(|y| edges2[y * 8 + 4])
            .fold(f32::NEG_INFINITY, f32::max);
        let max_col0: f32 = (1..7)
            .map(|y| edges2[y * 8])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_col4 > max_col0,
            "edge column should have higher response"
        );
    }

    // -----------------------------------------------------------------------
    // condition_images_from_multi_view
    // -----------------------------------------------------------------------

    #[test]
    fn test_condition_from_multi_view() {
        // Three valid normal maps (3 ch × 4×4 = 48 floats)
        let maps: Vec<Vec<f32>> = (0..3).map(|_| vec![0.5f32; 3 * 4 * 4]).collect();
        let conds = condition_images_from_multi_view(&maps, 4, 4);
        assert_eq!(conds.len(), 3);

        // One valid, one invalid (wrong size)
        let mixed: Vec<Vec<f32>> = vec![
            vec![0.5f32; 3 * 4 * 4], // valid
            vec![0.5f32; 4 * 4],     // invalid (only 1 channel worth)
        ];
        let conds2 = condition_images_from_multi_view(&mixed, 4, 4);
        assert_eq!(conds2.len(), 1, "invalid map should be silently skipped");
    }
}
