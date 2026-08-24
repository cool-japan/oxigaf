//! ControlNet-style conditioning for the diffusion pipeline.
//!
//! Conditioning signals (edge maps, depth maps, normal maps, pose heatmaps) are
//! resampled to each U-Net feature resolution and projected into the feature
//! space through zero-initialised 1×1 convolutions ([`ZeroConv`]) — the output
//! projections described by the ControlNet paper.
//!
//! # What this module does and does not provide
//!
//! * **Provided**: condition management ([`ControlNetProcessor`]), cached
//!   per-resolution resampling, layer selection
//!   ([`ControlNetConfig::injects_at`]) and the zero-convolution output
//!   projection ([`ZeroConv`], [`ControlNetProcessor::set_zero_conv`]).
//! * **Not provided**: the trainable copy of the U-Net encoder that produces
//!   ControlNet's control features. That requires trained weights which this
//!   crate does not ship. Until a [`ZeroConv`] is registered for a layer,
//!   [`ControlNetProcessor::apply_to_features`] uses a documented fallback that
//!   adds a channel-constant spatial bias — useful for smoke-testing the
//!   plumbing, but *not* learned control features.
//! * **Not wired**: [`crate::unet::MultiViewUNet::forward`] does not call
//!   [`ControlNetProcessor::apply_to_features`] itself; callers that want
//!   conditioning must invoke it on the feature buffers between U-Net stages.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// Maximum number of U-Net stages that can receive ControlNet conditioning.
///
/// [`ControlNetConfig::injection_layers`] entries must be strictly below this
/// value; [`ControlNetConfig::validate`] rejects anything larger. The injection
/// routines additionally clamp to `MAX_INJECTION_LAYERS - 1` so a hand-built
/// configuration that skipped validation can never overflow the `1 << layer`
/// resolution divisor.
pub const MAX_INJECTION_LAYERS: usize = 8;

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
    /// Flattened pixel data: `[channels * height * width]` f32.
    ///
    /// Conditioning maps are conventionally in `[0, 1]`, but nothing enforces
    /// that: signed data (surface normals encoded in `[-1, 1]`, for instance) is
    /// accepted as-is. Call [`ControlNetCondition::normalize`] to rescale an
    /// arbitrary range into `[0, 1]`.
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

    /// Rescales pixel data in-place to `[0, 1]` using min-max normalization.
    ///
    /// Every value becomes `(v - min) / (max - min)`, so the smallest sample
    /// maps to `0.0` and the largest to `1.0` — including for data that spans
    /// negative values (e.g. `[-5.0, 10.0]` becomes `[0.0, 1.0]`).
    ///
    /// The data is left **unchanged** when a meaningful range cannot be
    /// determined, namely when it is empty, when every value is equal (which
    /// covers the all-zero case), or when `min`/`max` are not finite.
    pub fn normalize(&mut self) {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in &self.data {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        if !min_val.is_finite() || !max_val.is_finite() {
            return;
        }
        let range = max_val - min_val;
        if !range.is_finite() || range <= 0.0 {
            return;
        }

        for v in &mut self.data {
            *v = (*v - min_val) / range;
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
    ///
    /// Every entry must be `< `[`MAX_INJECTION_LAYERS`]; larger indices are
    /// rejected by [`ControlNetConfig::validate`] and clamped by the injection
    /// routines.
    pub injection_layers: Vec<usize>,
    /// Global conditioning scale multiplied into every injected contribution.
    ///
    /// This stacks with each condition's own
    /// [`ControlNetCondition::conditioning_scale`]: the value added to a feature
    /// is `default_scale * conditioning_scale * <condition value>`. Set it to
    /// `1.0` to let per-condition scales act alone.
    pub default_scale: f32,
    /// If `true`, inject only at later (coarser) layers; if `false` inject at all listed layers.
    ///
    /// "Later" means the upper half of the distinct entries of
    /// [`ControlNetConfig::injection_layers`] once sorted — see
    /// [`ControlNetConfig::injects_at`].
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
    /// - `DiffusionError::InvalidConfig` if `max_conditions == 0`.
    /// - `DiffusionError::InvalidConfig` if `default_scale` is not a finite value `> 0.0`.
    /// - `DiffusionError::InvalidConfig` if any [`ControlNetConfig::injection_layers`]
    ///   entry is `>= `[`MAX_INJECTION_LAYERS`].
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if self.max_conditions == 0 {
            return Err(DiffusionError::InvalidConfig(
                "max_conditions must be > 0".to_string(),
            ));
        }
        if !self.default_scale.is_finite() || self.default_scale <= 0.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "default_scale must be a finite value > 0.0, got {}",
                self.default_scale
            )));
        }
        if let Some(&bad) = self
            .injection_layers
            .iter()
            .find(|&&layer| layer >= MAX_INJECTION_LAYERS)
        {
            return Err(DiffusionError::InvalidConfig(format!(
                "injection_layers entry {bad} is out of range: must be < {MAX_INJECTION_LAYERS}"
            )));
        }
        Ok(())
    }

    /// Returns `true` when `layer_idx` should receive conditioning.
    ///
    /// A layer qualifies when it appears in
    /// [`ControlNetConfig::injection_layers`]. When
    /// [`ControlNetConfig::late_injection`] is set, only the coarser half of the
    /// distinct listed layers qualifies — for `[0, 1, 2, 3]` that is `{2, 3}`,
    /// for `[0, 1, 2]` it is `{1, 2}` (with an odd count the middle layer is
    /// kept).
    pub fn injects_at(&self, layer_idx: usize) -> bool {
        if !self.injection_layers.contains(&layer_idx) {
            return false;
        }
        if !self.late_injection {
            return true;
        }

        // Count distinct entries, and how many of them are strictly coarser-side
        // (smaller index) than `layer_idx`, without allocating.
        let mut total_distinct = 0usize;
        let mut smaller_distinct = 0usize;
        for (i, &layer) in self.injection_layers.iter().enumerate() {
            if self.injection_layers[..i].contains(&layer) {
                continue;
            }
            total_distinct += 1;
            if layer < layer_idx {
                smaller_distinct += 1;
            }
        }
        smaller_distinct >= total_distinct / 2
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
// ZeroConv
// ---------------------------------------------------------------------------

/// Zero-initialised 1×1 output projection, as used by ControlNet.
///
/// ControlNet feeds its control features through 1×1 convolutions whose weights
/// and biases start at exactly zero, so an untrained branch contributes nothing
/// and training can grow the contribution smoothly from the base model. This
/// struct is one such projection for one injection layer.
///
/// `weight` is row-major `[out_channels][in_channels]` (`out_channels *
/// in_channels` entries) and `bias` holds `out_channels` entries. `in_channels`
/// is the number of stacked condition channels registered on the processor
/// (the sum of [`ControlNetCondition::channels`] over all conditions) and
/// `out_channels` is the number of U-Net feature channels at that layer.
#[derive(Debug, Clone)]
pub struct ZeroConv {
    in_channels: usize,
    out_channels: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ZeroConv {
    /// Creates an all-zero projection — the ControlNet initialisation.
    ///
    /// Applying it adds exactly `0.0` to every feature.
    pub fn zeros(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            weight: vec![0.0; out_channels.saturating_mul(in_channels)],
            bias: vec![0.0; out_channels],
        }
    }

    /// Creates a projection from trained weights.
    ///
    /// # Errors
    /// - `DiffusionError::ShapeMismatch` if `weight.len() != out_channels * in_channels`
    ///   or `bias.len() != out_channels`.
    pub fn from_weights(
        in_channels: usize,
        out_channels: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, DiffusionError> {
        let expected = out_channels.saturating_mul(in_channels);
        if weight.len() != expected {
            return Err(DiffusionError::ShapeMismatch {
                op: "ZeroConv::from_weights (weight)".to_string(),
                expected: vec![out_channels, in_channels],
                got: vec![weight.len()],
            });
        }
        if bias.len() != out_channels {
            return Err(DiffusionError::ShapeMismatch {
                op: "ZeroConv::from_weights (bias)".to_string(),
                expected: vec![out_channels],
                got: vec![bias.len()],
            });
        }
        Ok(Self {
            in_channels,
            out_channels,
            weight,
            bias,
        })
    }

    /// Number of condition channels this projection consumes.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Number of feature channels this projection produces.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Row-major `[out_channels][in_channels]` weights.
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Per-output-channel bias.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Returns `true` while every weight and bias is exactly zero, i.e. the
    /// projection is still at its ControlNet initialisation and contributes
    /// nothing.
    pub fn is_zero(&self) -> bool {
        self.weight.iter().all(|&w| w == 0.0) && self.bias.iter().all(|&b| b == 0.0)
    }

    /// Projects one spatial position: `out[oc] = bias[oc] + Σ_ic weight[oc][ic] * input[ic]`.
    ///
    /// Values missing from `input` are treated as `0.0`, so a short slice is
    /// projected as if zero-padded rather than panicking.
    pub fn project(&self, input: &[f32]) -> Vec<f32> {
        (0..self.out_channels)
            .map(|oc| {
                let base = oc * self.in_channels;
                let mut sum = self.bias.get(oc).copied().unwrap_or(0.0);
                for ic in 0..self.in_channels {
                    let w = self.weight.get(base + ic).copied().unwrap_or(0.0);
                    sum += w * input.get(ic).copied().unwrap_or(0.0);
                }
                sum
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DownsampledConditions (internal cache entry)
// ---------------------------------------------------------------------------

/// All registered conditions resampled to one feature resolution.
///
/// Cached per `(height, width)` because the result is a pure function of the
/// conditions and the target resolution — it does not change between denoising
/// steps, layers with the same resolution, or repeated calls.
#[derive(Debug)]
struct DownsampledConditions {
    /// Stacked condition channels at the cached resolution, layout
    /// `[channel][spatial]`, with each condition's `conditioning_scale` already
    /// applied.
    channels: Vec<f32>,
    /// Number of stacked condition channels (`Σ cond.channels`).
    num_channels: usize,
    /// Fallback aggregation: per-condition channel mean × that condition's
    /// scale, summed over conditions. Length `spatial`.
    aggregate: Vec<f32>,
}

impl DownsampledConditions {
    fn build(conditions: &[ControlNetCondition], target_h: usize, target_w: usize) -> Self {
        let spatial = target_h.saturating_mul(target_w);
        let num_channels: usize = conditions.iter().map(|c| c.channels).sum();

        let mut channels = vec![0.0f32; num_channels.saturating_mul(spatial)];
        let mut aggregate = vec![0.0f32; spatial];

        let mut offset = 0usize;
        for cond in conditions {
            let downsampled = cond.downsample_to(target_h, target_w);
            let scale = downsampled.conditioning_scale;
            let cond_channels = downsampled.channels;
            let inv_channels = if cond_channels > 0 {
                1.0 / cond_channels as f32
            } else {
                0.0
            };

            for cc in 0..cond_channels {
                let src = match downsampled.data.get(cc * spatial..(cc + 1) * spatial) {
                    Some(src) => src,
                    // Defensive: a hand-built condition may carry a `data`
                    // vector shorter than `channels * height * width`.
                    None => continue,
                };
                let dst_start = (offset + cc) * spatial;
                if let Some(dst) = channels.get_mut(dst_start..dst_start + spatial) {
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d = s * scale;
                    }
                }
                let weight = scale * inv_channels;
                for (a, &s) in aggregate.iter_mut().zip(src.iter()) {
                    *a += s * weight;
                }
            }

            offset += cond_channels;
        }

        Self {
            channels,
            num_channels,
            aggregate,
        }
    }
}

/// Maximum number of distinct resolutions kept in the downsample cache.
///
/// A U-Net only visits a handful of feature resolutions, so this only ever
/// trips if a caller sweeps resolutions; the cache is then dropped wholesale
/// rather than growing without bound.
const DOWNSAMPLE_CACHE_CAPACITY: usize = 16;

// ---------------------------------------------------------------------------
// ControlNetProcessor
// ---------------------------------------------------------------------------

/// Manages multiple ControlNet conditions and orchestrates feature injection.
///
/// Conditions are resampled lazily and cached per feature resolution, so a
/// denoising loop pays the resampling cost once rather than once per step.
#[derive(Debug)]
pub struct ControlNetProcessor {
    /// Active configuration.
    pub config: ControlNetConfig,
    /// Registered conditioning signals.
    conditions: Vec<ControlNetCondition>,
    /// Zero-convolution output projections, keyed by U-Net layer index.
    zero_convs: HashMap<usize, ZeroConv>,
    /// Conditions resampled to a given `(height, width)`, computed on demand.
    downsample_cache: Mutex<HashMap<(usize, usize), Arc<DownsampledConditions>>>,
}

impl ControlNetProcessor {
    /// Creates a new `ControlNetProcessor` with the given configuration.
    ///
    /// The configuration is *not* validated here (that would change this
    /// constructor's signature); call [`ControlNetConfig::validate`] yourself if
    /// the configuration came from user input. The injection routines clamp
    /// out-of-range layer indices defensively either way.
    pub fn new(config: ControlNetConfig) -> Self {
        Self {
            config,
            conditions: Vec::new(),
            zero_convs: HashMap::new(),
            downsample_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Adds a conditioning signal to the processor.
    ///
    /// # Errors
    /// - `DiffusionError::InvalidConfig` if ControlNet is disabled.
    /// - `DiffusionError::InvalidConfig` if `max_conditions` would be exceeded.
    /// - `DiffusionError::ShapeMismatch` if the condition's `data` length does
    ///   not match `channels * height * width` (possible for conditions built
    ///   through the struct literal rather than [`ControlNetCondition::new`]).
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
        let expected_len = cond
            .channels
            .saturating_mul(cond.height)
            .saturating_mul(cond.width);
        if cond.data.len() != expected_len {
            return Err(DiffusionError::ShapeMismatch {
                op: "ControlNetProcessor::add_condition".to_string(),
                expected: vec![cond.channels, cond.height, cond.width],
                got: vec![cond.data.len()],
            });
        }

        self.conditions.push(cond);
        self.invalidate_cache();
        Ok(())
    }

    /// Removes every registered condition.
    pub fn clear_conditions(&mut self) {
        self.conditions.clear();
        self.invalidate_cache();
    }

    /// Returns a slice of all registered conditions.
    pub fn conditions(&self) -> &[ControlNetCondition] {
        &self.conditions
    }

    /// Total number of stacked condition channels (`Σ cond.channels`).
    ///
    /// This is the `in_channels` a [`ZeroConv`] must declare to be used.
    pub fn condition_channels(&self) -> usize {
        self.conditions.iter().map(|c| c.channels).sum()
    }

    /// Registers the zero-convolution output projection for one U-Net layer.
    ///
    /// Once registered (and shape-compatible with the conditions and the
    /// feature buffer), [`ControlNetProcessor::apply_to_features`] uses the real
    /// ControlNet projection path instead of the broadcast fallback. Returns the
    /// previously registered projection, if any.
    pub fn set_zero_conv(&mut self, layer_idx: usize, conv: ZeroConv) -> Option<ZeroConv> {
        self.zero_convs.insert(layer_idx, conv)
    }

    /// Returns the zero-convolution registered for `layer_idx`, if any.
    pub fn zero_conv(&self, layer_idx: usize) -> Option<&ZeroConv> {
        self.zero_convs.get(&layer_idx)
    }

    /// Drops the resampled-condition cache (called whenever conditions change).
    fn invalidate_cache(&mut self) {
        let mut cache = self
            .downsample_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cache.clear();
    }

    /// Returns all conditions resampled to `(height, width)`, computing and
    /// caching the result on first use.
    fn downsampled(&self, height: usize, width: usize) -> Arc<DownsampledConditions> {
        let key = (height, width);
        let mut cache = self
            .downsample_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = cache.get(&key) {
            return Arc::clone(entry);
        }
        if cache.len() >= DOWNSAMPLE_CACHE_CAPACITY {
            cache.clear();
        }
        let entry = Arc::new(DownsampledConditions::build(
            &self.conditions,
            height,
            width,
        ));
        cache.insert(key, Arc::clone(&entry));
        entry
    }

    /// Injects conditioning into a mutable U-Net feature buffer for one layer.
    ///
    /// The buffer is left completely unchanged unless ControlNet is enabled,
    /// at least one condition is registered, and
    /// [`ControlNetConfig::injects_at`] accepts `layer_idx` (which honours both
    /// [`ControlNetConfig::injection_layers`] and
    /// [`ControlNetConfig::late_injection`]).
    ///
    /// `features` is `[channels][feature_h * feature_w]` row-major, where
    /// `channels` is implied by `features.len() / (feature_h * feature_w)`; a
    /// trailing partial channel is ignored.
    ///
    /// # Injection paths
    ///
    /// * **ControlNet projection** — used when a [`ZeroConv`] is registered for
    ///   `layer_idx` whose `in_channels` equals
    ///   [`ControlNetProcessor::condition_channels`] and whose `out_channels`
    ///   equals the implied feature-channel count. Each spatial position's
    ///   stacked condition channels are projected through the 1×1 convolution
    ///   and added to the corresponding feature channels. With the default
    ///   all-zero weights this adds nothing, exactly as in ControlNet before
    ///   training.
    /// * **Broadcast fallback** — used when no compatible projection is
    ///   registered. Each condition is reduced to its channel mean, scaled, and
    ///   the resulting scalar is added to *every* feature channel at that
    ///   position. This is a channel-constant spatial bias, not learned control
    ///   features; it exists so the plumbing can be exercised without trained
    ///   ControlNet weights.
    ///
    /// Both paths multiply by [`ControlNetConfig::default_scale`] and clamp the
    /// whole buffer to `[-10.0, 10.0]` afterwards to prevent explosion.
    pub fn apply_to_features(
        &self,
        features: &mut [f32],
        layer_idx: usize,
        feature_h: usize,
        feature_w: usize,
    ) {
        if !self.config.enabled || !self.config.injects_at(layer_idx) {
            return;
        }

        let spatial = feature_h.saturating_mul(feature_w);
        if spatial == 0 || features.is_empty() || self.conditions.is_empty() {
            return;
        }

        // Number of channels implied by feature buffer size.
        let feature_channels = features.len() / spatial;
        if feature_channels == 0 {
            return;
        }

        let global_scale = self.config.default_scale;
        let cached = self.downsampled(feature_h, feature_w);

        let projection = self.zero_convs.get(&layer_idx).filter(|conv| {
            conv.in_channels == cached.num_channels && conv.out_channels == feature_channels
        });

        match projection {
            Some(conv) => {
                // Real ControlNet output projection. The inner loop walks one
                // output channel's contiguous spatial slice; the condition
                // channels form a handful of parallel streams.
                for (oc, chunk) in features
                    .chunks_mut(spatial)
                    .take(feature_channels)
                    .enumerate()
                {
                    let base = oc * conv.in_channels;
                    let bias = conv.bias.get(oc).copied().unwrap_or(0.0);
                    for (s, v) in chunk.iter_mut().enumerate() {
                        let mut sum = bias;
                        for ic in 0..conv.in_channels {
                            let w = conv.weight.get(base + ic).copied().unwrap_or(0.0);
                            let c = cached
                                .channels
                                .get(ic * spatial + s)
                                .copied()
                                .unwrap_or(0.0);
                            sum += w * c;
                        }
                        *v += sum * global_scale;
                    }
                }
            }
            None => {
                // Broadcast fallback: one contiguous pass per feature channel.
                for chunk in features.chunks_mut(spatial).take(feature_channels) {
                    for (v, &a) in chunk.iter_mut().zip(cached.aggregate.iter()) {
                        *v += a * global_scale;
                    }
                }
            }
        }

        // Clamp to prevent gradient explosion.
        for v in features.iter_mut() {
            *v = v.clamp(-10.0, 10.0);
        }
    }

    /// Generates a [`ControlFeature`] for every layer that
    /// [`ControlNetConfig::injects_at`] accepts, by aggregating the downsampled
    /// conditions.
    ///
    /// Layer `n` is evaluated at `image_h / 2^n × image_w / 2^n` (floored, at
    /// least `1 × 1`), matching the U-Net's per-stage halving. The exponent is
    /// clamped to `MAX_INJECTION_LAYERS - 1` so an unvalidated
    /// [`ControlNetConfig::injection_layers`] entry can never overflow the
    /// shift.
    ///
    /// Output features have `channels = 1` (scalar aggregation), scaled by
    /// [`ControlNetConfig::default_scale`] and clamped to `[-10.0, 10.0]`.
    pub fn process_conditions(&self, image_h: usize, image_w: usize) -> Vec<ControlFeature> {
        if !self.config.enabled || self.conditions.is_empty() {
            return Vec::new();
        }

        let global_scale = self.config.default_scale;

        self.config
            .injection_layers
            .iter()
            .copied()
            .filter(|&layer_idx| self.config.injects_at(layer_idx))
            .map(|layer_idx| {
                // Halve resolution at each deeper layer (approximate U-Net
                // scale). Clamping keeps `1 << stage` well inside `usize`.
                let stage = layer_idx.min(MAX_INJECTION_LAYERS - 1);
                let scale_factor = 1usize.checked_shl(stage as u32).unwrap_or(1);
                let feat_h = (image_h / scale_factor).max(1);
                let feat_w = (image_w / scale_factor).max(1);

                let cached = self.downsampled(feat_h, feat_w);
                let data: Vec<f32> = cached
                    .aggregate
                    .iter()
                    .map(|&v| (v * global_scale).clamp(-10.0, 10.0))
                    .collect();

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
        // min was 0.0 and max 8.0 → (v - 0) / 8
        assert!((cond.data[3] - 1.0).abs() < 1e-6);
        assert!((cond.data[2] - 0.5).abs() < 1e-6);
        assert!(cond.data[0].abs() < 1e-6);
    }

    // Regression: `normalize` used to divide by the maximum only, so signed data
    // such as [-5.0, 10.0] came back as [-0.5, 1.0] — outside the documented
    // [0, 1] range — and all-negative data was left untouched entirely.
    #[test]
    fn test_condition_normalize_signed_data_lands_in_unit_range() {
        let mut cond = ControlNetCondition::depth_map(vec![-5.0f32, 0.0, 2.5, 10.0], 2, 2).unwrap();
        cond.normalize();
        for v in &cond.data {
            assert!(
                (0.0..=1.0).contains(v),
                "normalize must produce [0, 1], got {v}"
            );
        }
        assert!(cond.data[0].abs() < 1e-6, "min must map to 0.0");
        assert!((cond.data[3] - 1.0).abs() < 1e-6, "max must map to 1.0");
        // (0 - (-5)) / 15 = 1/3
        assert!((cond.data[1] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_condition_normalize_all_negative_lands_in_unit_range() {
        let mut cond =
            ControlNetCondition::depth_map(vec![-8.0f32, -6.0, -4.0, -2.0], 2, 2).unwrap();
        cond.normalize();
        for v in &cond.data {
            assert!(
                (0.0..=1.0).contains(v),
                "normalize must produce [0, 1], got {v}"
            );
        }
        assert!(cond.data[0].abs() < 1e-6);
        assert!((cond.data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_condition_normalize_constant_data_is_left_unchanged() {
        let mut cond = ControlNetCondition::depth_map(vec![0.0f32; 4], 2, 2).unwrap();
        cond.normalize();
        assert!(cond.data.iter().all(|v| v.abs() < 1e-6));

        let mut uniform = ControlNetCondition::depth_map(vec![3.0f32; 4], 2, 2).unwrap();
        uniform.normalize();
        assert!(uniform.data.iter().all(|v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_condition_normalize_non_finite_data_is_left_unchanged() {
        let mut cond =
            ControlNetCondition::depth_map(vec![0.0f32, 1.0, f32::INFINITY, 2.0], 2, 2).unwrap();
        cond.normalize();
        assert!((cond.data[1] - 1.0).abs() < 1e-6);
        assert!(cond.data[2].is_infinite());
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

    // -----------------------------------------------------------------------
    // injection_layers range checking (regression: shift-left overflow)
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validate_rejects_out_of_range_injection_layer() {
        let cfg = ControlNetConfig {
            injection_layers: vec![0, MAX_INJECTION_LAYERS],
            ..ControlNetConfig::default()
        };
        let result = cfg.validate();
        assert!(
            result.is_err(),
            "layer >= MAX_INJECTION_LAYERS must be rejected"
        );
        match result {
            Err(DiffusionError::InvalidConfig(msg)) => {
                assert!(
                    msg.contains("injection_layers"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("Expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_config_validate_rejects_non_finite_default_scale() {
        let cfg = ControlNetConfig {
            default_scale: f32::NAN,
            ..ControlNetConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // Regression: `process_conditions` used `1usize << layer_idx` on an
    // unvalidated, publicly settable index — 64 and above panicked in debug
    // builds ("attempt to shift left with overflow") and silently wrapped in
    // release builds.
    #[test]
    fn test_process_conditions_huge_layer_index_does_not_overflow() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig {
            injection_layers: vec![64, 200, usize::MAX],
            ..ControlNetConfig::default()
        });
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 64], 8, 8).unwrap())
            .unwrap();

        let features = proc.process_conditions(64, 64);
        assert_eq!(features.len(), 3);
        for feature in &features {
            assert!(feature.height >= 1 && feature.width >= 1);
            assert_eq!(feature.features.len(), feature.height * feature.width);
        }
    }

    // -----------------------------------------------------------------------
    // default_scale / late_injection are honoured
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_to_features_honours_default_scale() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig {
            default_scale: 2.5,
            ..ControlNetConfig::default()
        });
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 16], 4, 4).unwrap())
            .unwrap();

        let mut features = vec![0.0f32; 16];
        proc.apply_to_features(&mut features, 0, 4, 4);
        for v in &features {
            assert!((*v - 2.5).abs() < 1e-5, "expected 2.5, got {v}");
        }
    }

    #[test]
    fn test_process_conditions_honours_default_scale() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig {
            injection_layers: vec![0],
            default_scale: 3.0,
            ..ControlNetConfig::default()
        });
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 16], 4, 4).unwrap())
            .unwrap();

        let features = proc.process_conditions(4, 4);
        assert_eq!(features.len(), 1);
        for v in &features[0].features {
            assert!((*v - 3.0).abs() < 1e-5, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn test_injects_at_late_injection_keeps_coarser_half() {
        let cfg = ControlNetConfig {
            injection_layers: vec![0, 1, 2, 3],
            late_injection: true,
            ..ControlNetConfig::default()
        };
        assert!(!cfg.injects_at(0));
        assert!(!cfg.injects_at(1));
        assert!(cfg.injects_at(2));
        assert!(cfg.injects_at(3));
        assert!(!cfg.injects_at(7), "unlisted layer never injects");
    }

    #[test]
    fn test_injects_at_odd_count_keeps_middle_layer() {
        let cfg = ControlNetConfig {
            injection_layers: vec![2, 0, 1],
            late_injection: true,
            ..ControlNetConfig::default()
        };
        assert!(!cfg.injects_at(0));
        assert!(cfg.injects_at(1));
        assert!(cfg.injects_at(2));
    }

    #[test]
    fn test_injects_at_all_layers_when_late_injection_off() {
        let cfg = ControlNetConfig::default();
        for layer in 0..4 {
            assert!(cfg.injects_at(layer));
        }
    }

    #[test]
    fn test_apply_to_features_late_injection_skips_early_layers() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig {
            late_injection: true,
            ..ControlNetConfig::default()
        });
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 16], 4, 4).unwrap())
            .unwrap();

        let mut early = vec![0.0f32; 16];
        proc.apply_to_features(&mut early, 0, 4, 4);
        assert!(
            early.iter().all(|v| v.abs() < 1e-10),
            "layer 0 must be skipped when late_injection is set"
        );

        let mut late = vec![0.0f32; 16];
        proc.apply_to_features(&mut late, 3, 4, 4);
        assert!(late.iter().all(|v| (*v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_process_conditions_late_injection_filters_layers() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig {
            injection_layers: vec![0, 1, 2, 3],
            late_injection: true,
            ..ControlNetConfig::default()
        });
        proc.add_condition(ControlNetCondition::depth_map(vec![0.5f32; 256], 16, 16).unwrap())
            .unwrap();

        let features = proc.process_conditions(16, 16);
        let layers: Vec<usize> = features.iter().map(|f| f.layer_idx).collect();
        assert_eq!(layers, vec![2, 3]);
    }

    // -----------------------------------------------------------------------
    // Multi-channel / multi-condition aggregation semantics
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_to_features_broadcasts_across_all_feature_channels() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 4], 2, 2).unwrap())
            .unwrap();

        // 3 feature channels over a 2×2 map.
        let mut features = vec![0.0f32; 3 * 4];
        proc.apply_to_features(&mut features, 0, 2, 2);
        assert_eq!(features.len(), 12);
        for v in &features {
            assert!((*v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_apply_to_features_sums_conditions_with_per_condition_scale() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        proc.add_condition(
            ControlNetCondition::new(ControlSignalType::DepthMap, vec![1.0f32; 4], 2, 2, 0.5)
                .unwrap(),
        )
        .unwrap();
        // A 3-channel normal map of all 2.0 has channel mean 2.0.
        proc.add_condition(
            ControlNetCondition::new(ControlSignalType::NormalMap, vec![2.0f32; 3 * 4], 2, 2, 1.0)
                .unwrap(),
        )
        .unwrap();

        let mut features = vec![0.0f32; 4];
        proc.apply_to_features(&mut features, 0, 2, 2);
        // 1.0 * 0.5 + 2.0 * 1.0 = 2.5
        for v in &features {
            assert!((*v - 2.5).abs() < 1e-5, "expected 2.5, got {v}");
        }
    }

    #[test]
    fn test_apply_to_features_repeated_calls_accumulate_identically() {
        // Guards the per-resolution downsample cache: the second call must see
        // exactly the same conditioning as the first.
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 16], 4, 4).unwrap())
            .unwrap();

        let mut features = vec![0.0f32; 16];
        proc.apply_to_features(&mut features, 0, 4, 4);
        proc.apply_to_features(&mut features, 0, 4, 4);
        for v in &features {
            assert!((*v - 2.0).abs() < 1e-5, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn test_apply_to_features_cache_invalidated_by_new_condition() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 16], 4, 4).unwrap())
            .unwrap();

        let mut first = vec![0.0f32; 16];
        proc.apply_to_features(&mut first, 0, 4, 4);
        assert!(first.iter().all(|v| (*v - 1.0).abs() < 1e-5));

        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 16], 4, 4).unwrap())
            .unwrap();

        let mut second = vec![0.0f32; 16];
        proc.apply_to_features(&mut second, 0, 4, 4);
        assert!(
            second.iter().all(|v| (*v - 2.0).abs() < 1e-5),
            "cache must be invalidated when a condition is added"
        );
    }

    #[test]
    fn test_apply_to_features_clamps_to_plus_minus_ten() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        proc.add_condition(
            ControlNetCondition::new(ControlSignalType::DepthMap, vec![1.0f32; 4], 2, 2, 100.0)
                .unwrap(),
        )
        .unwrap();

        let mut features = vec![0.0f32; 4];
        proc.apply_to_features(&mut features, 0, 2, 2);
        for v in &features {
            assert!((*v - 10.0).abs() < 1e-5, "expected clamp to 10.0, got {v}");
        }
    }

    #[test]
    fn test_add_condition_rejects_inconsistent_data_length() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        // Built via struct literal so it bypasses `ControlNetCondition::new`.
        let malformed = ControlNetCondition {
            signal_type: ControlSignalType::DepthMap,
            data: vec![1.0f32; 3],
            channels: 1,
            height: 4,
            width: 4,
            conditioning_scale: 1.0,
        };
        let result = proc.add_condition(malformed);
        assert!(result.is_err());
        match result {
            Err(DiffusionError::ShapeMismatch { op, .. }) => {
                assert!(op.contains("add_condition"), "unexpected op: {op}");
            }
            other => panic!("Expected ShapeMismatch, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // ZeroConv / ControlNet output projection
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_conv_zeros_is_zero() {
        let conv = ZeroConv::zeros(3, 8);
        assert_eq!(conv.in_channels(), 3);
        assert_eq!(conv.out_channels(), 8);
        assert_eq!(conv.weight().len(), 24);
        assert_eq!(conv.bias().len(), 8);
        assert!(conv.is_zero());
        assert_eq!(conv.project(&[1.0, 2.0, 3.0]), vec![0.0; 8]);
    }

    #[test]
    fn test_zero_conv_from_weights_shape_checked() {
        assert!(ZeroConv::from_weights(2, 3, vec![0.0; 6], vec![0.0; 3]).is_ok());
        assert!(ZeroConv::from_weights(2, 3, vec![0.0; 5], vec![0.0; 3]).is_err());
        assert!(ZeroConv::from_weights(2, 3, vec![0.0; 6], vec![0.0; 2]).is_err());
    }

    #[test]
    fn test_zero_conv_project() {
        // out0 = 1*a + 0*b + 1, out1 = 0*a + 2*b + 0
        let conv = ZeroConv::from_weights(2, 2, vec![1.0, 0.0, 0.0, 2.0], vec![1.0, 0.0])
            .expect("valid shapes");
        assert!(!conv.is_zero());
        let out = conv.project(&[3.0, 5.0]);
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_to_features_zero_conv_contributes_nothing_until_trained() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 4], 2, 2).unwrap())
            .unwrap();
        // 1 condition channel in, 2 feature channels out.
        proc.set_zero_conv(0, ZeroConv::zeros(1, 2));

        let mut features = vec![0.0f32; 2 * 4];
        proc.apply_to_features(&mut features, 0, 2, 2);
        assert!(
            features.iter().all(|v| v.abs() < 1e-10),
            "an untrained zero-conv must add exactly nothing"
        );
    }

    #[test]
    fn test_apply_to_features_uses_zero_conv_projection_when_trained() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        // Two 1-channel conditions → in_channels = 2.
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 4], 2, 2).unwrap())
            .unwrap();
        proc.add_condition(ControlNetCondition::edge_map(vec![3.0f32; 4], 2, 2).unwrap())
            .unwrap();
        assert_eq!(proc.condition_channels(), 2);

        // out0 = 1*c0 + 0*c1, out1 = 0*c0 + 2*c1
        let conv = ZeroConv::from_weights(2, 2, vec![1.0, 0.0, 0.0, 2.0], vec![0.0, 0.0])
            .expect("valid shapes");
        proc.set_zero_conv(0, conv);

        let mut features = vec![0.0f32; 2 * 4];
        proc.apply_to_features(&mut features, 0, 2, 2);
        // Channel 0 gets c0 = 1.0, channel 1 gets 2 * c1 = 6.0.
        for v in &features[..4] {
            assert!((*v - 1.0).abs() < 1e-5, "channel 0: expected 1.0, got {v}");
        }
        for v in &features[4..] {
            assert!((*v - 6.0).abs() < 1e-5, "channel 1: expected 6.0, got {v}");
        }
    }

    #[test]
    fn test_apply_to_features_incompatible_zero_conv_falls_back_to_broadcast() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 4], 2, 2).unwrap())
            .unwrap();
        // Declares 5 input channels but only 1 condition channel exists.
        proc.set_zero_conv(0, ZeroConv::zeros(5, 2));

        let mut features = vec![0.0f32; 2 * 4];
        proc.apply_to_features(&mut features, 0, 2, 2);
        for v in &features {
            assert!(
                (*v - 1.0).abs() < 1e-5,
                "expected broadcast fallback, got {v}"
            );
        }
    }

    #[test]
    fn test_zero_conv_accessor_round_trip() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        assert!(proc.zero_conv(1).is_none());
        assert!(proc.set_zero_conv(1, ZeroConv::zeros(3, 4)).is_none());
        let stored = proc.zero_conv(1).expect("just registered");
        assert_eq!(stored.in_channels(), 3);
        assert_eq!(stored.out_channels(), 4);
        let previous = proc.set_zero_conv(1, ZeroConv::zeros(3, 8));
        assert_eq!(previous.map(|c| c.out_channels()), Some(4));
    }

    #[test]
    fn test_clear_conditions_resets_injection() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 16], 4, 4).unwrap())
            .unwrap();
        let mut features = vec![0.0f32; 16];
        proc.apply_to_features(&mut features, 0, 4, 4);
        assert!(features.iter().all(|v| (*v - 1.0).abs() < 1e-5));

        proc.clear_conditions();
        assert_eq!(proc.conditions().len(), 0);
        let mut after = vec![7.0f32; 16];
        proc.apply_to_features(&mut after, 0, 4, 4);
        assert!(
            after.iter().all(|v| (*v - 7.0).abs() < 1e-5),
            "no conditions means no change at all"
        );
    }

    #[test]
    fn test_apply_to_features_partial_trailing_channel_is_ignored() {
        let mut proc = ControlNetProcessor::new(ControlNetConfig::default());
        proc.add_condition(ControlNetCondition::depth_map(vec![1.0f32; 4], 2, 2).unwrap())
            .unwrap();

        // 4 spatial + 2 trailing floats: one whole channel plus a remainder.
        let mut features = vec![0.0f32; 6];
        proc.apply_to_features(&mut features, 0, 2, 2);
        for v in &features[..4] {
            assert!((*v - 1.0).abs() < 1e-5);
        }
        for v in &features[4..] {
            assert!(
                v.abs() < 1e-10,
                "trailing partial channel must be untouched"
            );
        }
    }
}
